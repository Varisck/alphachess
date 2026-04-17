"""Chess game environment.

Values and rewards are always from the perspective of the player to move
- A position where the side to move is checkmated has value -1
- After applying a move, the value sign must flip (because players alternate)

Orientation:
All positions are encoded from the current player's perspective
If it's Black to move, the board is flipped vertically (rank → 7-rank)
and colors are swapped in the piece planes before encoding
Same geometric move from White and Black encodes to the same action index
"""

from __future__ import annotations

import chess
import numpy as np

from alphachess.game import encoding as _enc


class ChessEnv:
    # stateless — all behaviour lives in static methods

    @staticmethod
    def initial_state() -> chess.Board:
        # fresh game, standard starting position
        return chess.Board()

    @staticmethod
    def legal_action_mask(board: chess.Board) -> np.ndarray:
        # passthrough — canonical implementation is in encoding.py
        return _enc.legal_action_mask(board)

    @staticmethod
    def apply(board: chess.Board, action_index: int) -> None:
        """Mutate board in place. Use board.pop() to undo (for MCTS simulation).

        No copy is made — the caller owns the board and decides when to snapshot.
        Raises ValueError (from index_to_move) or IllegalMoveError (from push)
        if the action is not legal in this position.
        """
        move = _enc.index_to_move(action_index, board)
        board.push(move)

    @staticmethod
    def is_terminal(board: chess.Board) -> tuple[bool, float]:
        """Return (game_over, value_from_current_player's_perspective).

        Possible values:
          -1.0  the side to move is checkmated
           0.0  draw (stalemate, 75-move rule, 5-fold repetition, insufficient material)

        NOTE: +1.0 is impossible here. The side to move can never have
        already won — winning requires delivering mate on the *previous* move,
        making the mated player the one to move in this terminal state.
        The +1 only appears during training when the trajectory is relabelled
        from the opponent's earlier positions.
        """
        outcome = board.outcome()
        if outcome is None:
            return False, 0.0
        if outcome.winner is None:
            # stalemate, 75-move rule, 5-fold repetition, insufficient material
            return True, 0.0
        # someone won; by construction it is always the opponent of the side to move
        return True, -1.0

    @staticmethod
    def encode(board: chess.Board) -> np.ndarray:
        # canonical [18, 8, 8] float32 tensor
        return _enc.encode(board)
