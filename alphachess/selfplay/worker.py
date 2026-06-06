"""Batched selfplay worker

A long-running process that plays N games concurrently against itself and
writes the resulting (state, pi, z) records to storage. Selfplay and training
are decoupled — the worker writes records and forgets; the trainer reads them
on its own schedule

@Details
--------

    Play stages:
        A. Descend each active tree to a leaf
        B. One batched NN call for all leaves
        C. Expand each leaf with its prior, back up its value
        D. plays a move, finalize games and replaced with a fresh game
        E. Check for new models, load new model and clear current running games


    Shard format (``selfplay.games_per_worker_batch`` games per shard):
        states:         float32, [plies, planes, 8, 8]
        policy_targets: float32, [plies, action_space]  (MCTS visit distribution)
        values:         float32, [plies]                (outcome from current 
                                                          player prospective)

    File path:
        selfplay/{YYYYMMDD}/{timestamp}-{hostname}-{pid}-{uuid}.npz
"""

from __future__ import annotations

import io
import logging
import os
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import chess
import numpy as np
import random
import signal
import torch

from alphachess.config import Config
from alphachess.mcts.tree import Tree
from alphachess.nn.inference import InferenceModel
from alphachess.storage import Storage
from alphachess.game.encoding import encode, index_to_move

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-game state held by the worker for one in-flight game
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Mutable per-game bookkeeping for one game inside a batched cycle."""

    game_id: int
    board: chess.Board
    # Tree for the search algorithm, this will be re-used
    tree: Tree
    # One entry per ply already played: (encoded_state, pi, player_to_move).
    # player_to_move is needed at finalize time to project the game outcome
    # into each position's player-to-move perspective
    history: list[tuple[np.ndarray, np.ndarray, chess.Color]] = field(default_factory=list)
    # % of games that never resign
    resign_disabled: bool = False
    # Number of consecutive recent root_values below resign_threshold
    consecutive_resign_hits: int = 0
    # Set when the game has ended (terminal, resignation, or ply cap)
    done: bool = False
    # Final outcome from White's perspective: +1 / 0 / -1. None until done
    outcome_white_pov: float | None = None
    # Number of simulatino to be done in each game
    sim_to_do: int = 0


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class BatchedSelfplayWorker:
    """Plays ``games_per_worker_batch`` games concurrently per cycle"""

    def __init__(
        self,
        model: InferenceModel,
        storage: Storage,
        config: Config,
    ) -> None:
        self._model = model
        self._storage = storage
        self._config = config
        self._shutdown = False

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Continuous actor loop: play games, shard finished ones to storage,
        hot-reload the model mid-stream, and flush cleanly on SIGTERM/SIGINT.

        Logging is configured by the module-level ``run`` entrypoint.
        """

        def _request_shutdown(signum, _frame):
            # Minimal handler: just request stop. The loop flushes at a safe
            # point. A second signal escalates to an immediate hard exit.
            if self._shutdown:
                raise SystemExit(1)
            log.info("signal %s received; finishing tick, then flushing", signum)
            self._shutdown = True

        signal.signal(signal.SIGTERM, _request_shutdown)
        signal.signal(signal.SIGINT, _request_shutdown)
        
        log.info("worker start: model gen=%d, %d concurrent games",
         self._model.current_generation(), self._config.selfplay.games_per_worker_batch)
        
        queue: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        games = self._init_batch()

        while not self._shutdown:
            leaves = self._descend_all(games)

            # No network leaves (every active game hit a terminal
            # leaf) still advances games — but there is nothing to expand.
            if leaves:
                priors, values = self._evaluate_leaves(leaves)
                self._expand_and_backup(games, leaves, priors, values)

            self._advance_games(games)

            # check for terminated games and refill the batch
            for i, game in enumerate(games):
                if game.done:
                    queue.append(self._finalize_game(game))
                    games[i] = self._new_game_state(i)

            # if queue is full save shard
            if len(queue) >= self._config.selfplay.records_per_shard:
                self._write_shard(queue)
                log.info("wrote shard: %d games, %d positions", 
                         len(queue), sum(len(s) for s, _, _ in queue))
                queue = []


                # now check for new generation model
                # auto swaps and if new model run new games
                reloaded = self._model.maybe_reload()
                if reloaded:
                    log.info("reloaded model -> gen=%d, discarding in-flight games",
                              self._model.current_generation())
                    games = self._init_batch()



        if self._config.selfplay.save_on_shutdown:
            log.info("shutdown: flushed %d buffered games", len(queue))
            self._write_shard(queue)


    def _new_game_state(self, id) -> GameState:
        """Create a new game state with game_id = id"""
        cfg = self._config
        # safe guard the num nodes so no overflow after re-root
        max_nodes = 2 * cfg.mcts.num_simulations + 1
        resign_pct = cfg.selfplay.resign_disable_pct
        return GameState(
                game_id=id,
                board=chess.Board(),
                tree=Tree(max_nodes, cfg.mcts, cfg.game.action_space),
                resign_disabled=random.random() < resign_pct,
                sim_to_do=cfg.mcts.num_simulations
            )


    def _init_batch(self) -> list[GameState]:
        """Allocate N fresh GameState objects (board, tree, flags)."""
        return [
            self._new_game_state(i)
            for i in range(self._config.selfplay.games_per_worker_batch)
        ]


    # ------------------------------------------------------------------
    # Phase A — descend each active tree to a leaf
    # ------------------------------------------------------------------

    def _descend_all(
        self,
        games: list[GameState],
    ) -> list[tuple[int, chess.Board, int]]:
        """For each active game, walk root → leaf via PUCT.

        Returns:
            list (game_index, leaf_board, leaf_node_id):  games that need network eval 
            (Terminal leaves are handled here with the game result value)
        """

        leaves: list[tuple[int, chess.Board, int]] = []
        for i, game in enumerate(games):
            if game.done or game.sim_to_do == 0:
                # skip game if it is finished
                # or simulations reached capped number
                continue
            # decrement the simulations
            game.sim_to_do -= 1
            leaf_board, leaf_node_id = game.tree.descend_to_leaf(game.board)
            leaf_value = self._terminal_value(leaf_board)
            # if node is terminal backup the value right away
            if leaf_value is not None:
                game.tree.backup(leaf_node_id, leaf_value)
                continue

            leaves.append((i, leaf_board, leaf_node_id))
        return leaves


    # ------------------------------------------------------------------
    # Phase B — one batched NN call for all leaves
    # ------------------------------------------------------------------

    def _evaluate_leaves(
        self,
        leaves: list[tuple[int, chess.Board, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack encoded leaf boards and run ``predict_batch`` once.

        Returns ``(priors [B, action_space], values [B])``.
        """

        encoded = np.stack([encode(board) for _, board, _ in leaves])
        return self._model.predict_batch(encoded)

    # ------------------------------------------------------------------
    # Phase C — expand + backup
    # ------------------------------------------------------------------

    def _expand_and_backup(
        self,
        games: list[GameState],
        leaves: list[tuple[int, chess.Board, int]],
        priors: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Expand each leaf with its prior and back up the value.

        Dirichlet noise is added to the current root's prior
        """

        for idx, (k, board, node_id) in enumerate(leaves):
            game = games[k]
            game.tree.expand(node_id, board, priors[idx, :])

            if node_id == 0:
                self._inject_root_noise(game.tree)

            game.tree.backup(node_id, float(values[idx]))


    # ------------------------------------------------------------------
    # Phase D — advance games whose sim budget is met
    # ------------------------------------------------------------------

    def _advance_games(self, games: list[GameState]) -> None:
        """For each active game whose root has enough visits, play a move.
        """
        for game in games:
            if game.done or game.sim_to_do > 0:
                continue
            pi = game.tree.root_visit_distribution()
            move = self._sample_move_index(pi, 
                                           self._temperature_for_ply(game.board.ply()))
            encoded_state = encode(game.board)
            game.history.append((encoded_state, pi, game.board.turn))

            game.board.push(index_to_move(move, game.board))
    
            resign = self._check_resignation(game, game.tree.root_value())
            terminal = self._terminal_value(game.board)
            if resign or terminal is not None or game.board.ply() >= self._config.selfplay.max_game_plies:
                game.done = True
                game.outcome_white_pov = self._compute_outcome(game, resign, terminal)
                continue

            game.tree.reroot(move)
            game.sim_to_do = max(0, self._config.mcts.num_simulations - 
                                 game.tree.root_visits())
            if game.tree.is_expanded[0]:
                self._inject_root_noise(game.tree)





    # ------------------------------------------------------------------
    # Finalize one game: project outcome to each ply, write .npz
    # ------------------------------------------------------------------

    def _finalize_game(self, game: GameState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project the game outcome to player-to-move perspective and write.

        Value target z at ply t is:
            +1 if the player to move at ply t eventually won
            -1 if they eventually lost
             0 for a draw / max-ply cap / forced resignation outcome
        """
        assert game.done, "finalize called on an unfinished game"
        assert game.outcome_white_pov is not None, "finished game has no outcome"
        assert game.history, "finished game has empty history"

        states = np.stack([s  for s, _, _   in game.history])
        policy_targets = np.stack([pi for _, pi, _  in game.history])
        z = game.outcome_white_pov
        values = np.array(
            [z if player == chess.WHITE else -z for _, _, player in game.history],
            dtype=np.float32,
        )
        return (states, policy_targets, values)


    def _write_shard(
        self,
        queue: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        """Serialize a shard of finished games to ``.npz`` and write to storage.

        Path: ``selfplay/{YYYYMMDD}/{timestamp}-{hostname}-{pid}-{uuid}.npz``.
        """
        if not queue:
            return

        states = np.concatenate([s for s, _, _ in queue])
        policy_targets = np.concatenate([p for _, p, _ in queue])
        values = np.concatenate([v for _, _, v in queue])

        bio = io.BytesIO()
        np.savez_compressed(
            bio,
            states=states,
            policy_targets=policy_targets,
            values=values,
        )
        # atomic_put writes to a ".tmp" sibling then renames, so a trainer
        # listing this directory concurrently never reads a half-written .npz
        # (the temp name ends in ".tmp", not ".npz", so it is filtered out).
        self._storage.atomic_put(self._record_relpath(), bio.getvalue())

    # ------------------------------------------------------------------
    # Move selection & game-end policies
    # ------------------------------------------------------------------

    def _temperature_for_ply(self, ply: int) -> float:
        """ Get the temperature to select the move index
        
            1.0 for the first ``temperature_moves`` plies, then 0 (argmax)
        
            Returns
            --------
            temperature: the temperature of the move
        """
        return 1.0 if ply < self._config.selfplay.temperature_moves else 0.0


    def _sample_move_index(self, pi: np.ndarray, temperature: float) -> int:
        """Sample an action index from pi^(1/T), or argmax if T == 0
        
            Returns
            -------
            index (int): index with the chosen move
        """
        if temperature == 0.0:
            return int(np.argmax(pi))
        if temperature == 1.0:
            return int(np.random.choice(pi.shape[0], p=pi))
        # general case (spareing a pow when temp == 1 above)
        p = pi ** (1.0 / temperature)
        p = p / p.sum()
        return int(np.random.choice(p.shape[0], p=p))


    def _check_resignation(self, game: GameState, root_value: float) -> bool:
        """Update the consecutive-hits counter; return True if the game
        should resign now. No-op on the ``resign_disabled`` games.
        """
        if game.resign_disabled:
            return False
        if root_value < self._config.selfplay.resign_threshold:
            game.consecutive_resign_hits += 1
        else:
            game.consecutive_resign_hits = 0
        # if conescutive_hits is ge then config resign
        return game.consecutive_resign_hits >= self._config.selfplay.resign_consecutive_hits


    def _terminal_value(self, board: chess.Board) -> float | None:
        """
        Return the game-theoretic value if board is terminal, else None.
        """
        if not board.is_game_over():
            return None
        if board.is_checkmate():
            return -1.0
        return 0.0              # stalemate, repetition, 50move ...


    def _compute_outcome(
        self,
        game: GameState,
        resigned: bool,
        terminal: float | None
    ) -> float:
        """Compute final game outcome from White's POV: +1 White win, -1 Black win, 0 draw.

        *Must* be called after the chosen move has been pushed onto ``game.board``

        Args:
            game: the game whose board already has the move pushed.
            resigned: True if ``player_to_move`` resigned on this ply.
            terminal: result of _terminal_value() function
        """
        player_to_move = game.board.turn
        # Resignation: the side to move at the root gives up and therefore loses.
        if resigned:
            return 1.0 if player_to_move == chess.WHITE else -1.0

        # Terminal position. _terminal_value is from the perspective of the
        # side to move in the resulting position (game.board.turn); project it
        # onto White's POV. Checkmate -> the side that just moved won; any
        # other game-over (stalemate, repetition, 50-move) -> 0.
        if terminal is not None:
            return terminal if game.board.turn == chess.WHITE else -terminal

        # No terminal, no resignation -> reached max_game_plies -> draw.
        return 0.0


    def _inject_root_noise(self, tree: Tree) -> None:
        legal = tree.legal_masks[0]
        n_legal = int(legal.sum())
        if n_legal == 0:
            return
        noise = np.random.dirichlet([self._config.mcts.dirichlet_alpha] * n_legal).astype(np.float32)
        eps = self._config.mcts.dirichlet_epsilon
        tree.P[0, legal] = (1 - eps) * tree.P[0, legal] + eps * noise


    # ------------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------------

    def _record_relpath(self) -> str:
        """``selfplay/{YYYYMMDD}/{ts}-{hostname}-{pid}-{uuid}.npz``

        ``{day}`` partitions records for cheap windowing/eviction; the
        ``{ts}-{hostname}-{pid}-{uuid}`` tail makes the name globally unique
        across concurrent workers without any shared counter. The uuid alone
        guarantees uniqueness; the timestamp keeps names roughly sortable
        """
        now = datetime.now(timezone.utc)
        day = now.strftime("%Y%m%d")
        ts = now.strftime("%Y%m%dT%H%M%S%f")
        host = socket.gethostname()
        pid = os.getpid()
        uid = uuid.uuid4().hex[:12]
        return f"selfplay/{day}/{ts}-{host}-{pid}-{uid}.npz"
    


# ---------------------------------------------------------------------------
# Module entrypoint (matches alphachess.pretrain.trainer.run signature)
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    # mirrors pretrain.trainer._select_device; duplicated rather than imported
    # so selfplay does not depend on the pretrain package
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run(config: Config) -> None:
    """Construct dependencies and start one selfplay worker loop
        
        this entrypoint mirrors ``alphachess.pretrain.trainer.run``"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    storage = Storage(config.storage.root_uri)
    device = _select_device()
    model = InferenceModel(storage, device, config)   # loads newest gen; raises if none
    worker = BatchedSelfplayWorker(model, storage, config)

    log.info("selfplay worker start: device=%s, config hash=%s", device, config.hash())
    worker.run()


if __name__ == "__main__":
    run(Config.from_env())
