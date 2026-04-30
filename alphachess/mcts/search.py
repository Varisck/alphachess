from __future__ import annotations

import chess
import numpy as np

from alphachess.config import MCTSConfig
from alphachess.game.encoding import encode, index_to_move
from alphachess.mcts.tree import Tree
from alphachess.nn.inference import InferenceModel


class MCTS:
    """Single-game MCTS.

    run() performs num_simulations iterations of:
        descend → evaluate → expand → backup
    and returns the visit-count distribution at the root.
    """

    def __init__(self, model: InferenceModel, config: MCTSConfig) -> None:
        self._model = model
        self._config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, root_board: chess.Board, add_root_noise: bool = True) -> np.ndarray:
        """Run MCTS from root_board and return the policy distribution.

        Args:
            root_board: board at the root of the search
            add_root_noise:  inject Dirichlet noise at root (True during
                             selfplay, False during evaluation/testing)

        Returns:
            dist (float32):  array of shape [action_space], root move distribution
        """
        tree = self._new_tree()

        for i in range(self._config.num_simulations):
            board, node_id = tree.descend_to_leaf(root_board)

            # if position is terminal no need to run network
            value = self._terminal_value(board)
            if value is not None:
                tree.backup(node_id, value)
                continue

            policy, values = self._model.predict_batch(encode(board)[np.newaxis])
            policy = policy[0]
            value = float(values[0])

            tree.expand(node_id, board, policy)     # save values in tree

            if add_root_noise and node_id == 0:
                # adding noise to root on self_play
                legal = tree.legal_masks[0]
                n_legal = int(legal.sum())
                noise = np.random.dirichlet([self._config.dirichlet_alpha] * n_legal).astype(np.float32)
                eps = self._config.dirichlet_epsilon
                tree.P[0, legal] = (1 - eps) * tree.P[0, legal] + eps * noise

            tree.backup(node_id, value)     # backup the infos to root

        return tree.root_visit_distribution()


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _terminal_value(self, board: chess.Board) -> float | None:
        """
        Return the game-theoretic value if board is terminal, else None.
        """
        if not board.is_game_over():
            return None
        if board.is_checkmate():
            return -1.0
        return 0.0              # stalemate, repetition, 50move ...

    def _new_tree(self) -> Tree:
        """Allocate a fresh Tree sized for one search."""
        max_nodes = self._config.num_simulations + 1
        return Tree(max_nodes, self._config)

