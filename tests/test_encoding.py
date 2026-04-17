"""
Tests for alphachess/game/encoding.py.
"""

import chess
import numpy as np
import pytest

from alphachess.game.encoding import (
    NUM_ACTIONS,
    encode,
    index_to_move,
    legal_action_mask,
    move_to_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _board(*sans: str) -> chess.Board:
    """Build a board by replaying a sequence of SAN moves from the start."""
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


# ------------------------------------------------
#                Worked examples
# ------------------------------------------------


class TestWorkedExamples:
    def test_e4_index(self):
        board = chess.Board()
        move = board.parse_san("e4")
        assert move_to_index(move, board) == 877

    def test_e5_index(self):
        # 1...e5 — Black's e7→e5 must map to the same index as 1.e4
        board = _board("e4")
        move = board.parse_san("e5")
        assert move_to_index(move, board) == 877

    def test_nf3_index(self):
        board = chess.Board()
        move = board.parse_san("Nf3")
        assert move_to_index(move, board) == 501

    def test_white_kingside_castling(self):
        # O-O White: e1→g1
        board = _board("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")
        move = board.parse_san("O-O")
        assert move_to_index(move, board) == 307

    def test_white_queenside_castling(self):
        # O-O-O White: e1→c1
        board = _board("d4", "d5", "Nc3", "Nc6", "Bf4", "Bf5", "Qd2", "Qd7")
        move = board.parse_san("O-O-O")
        assert move_to_index(move, board) == 335

    def test_queen_promotion(self):
        # e7-e8=Q  (White pawn on e7)
        board = chess.Board("8/4P3/8/2k5/8/2K5/8/8 w - - 0 1")
        move = chess.Move.from_uci("e7e8q")
        assert move_to_index(move, board) == 3796

    def test_knight_underpromotion(self):
        # e7-e8=N
        board = chess.Board("8/4P3/8/2k5/8/2K5/8/8 w - - 0 1")
        move = chess.Move.from_uci("e7e8n")
        assert move_to_index(move, board) == 3860

    def test_bishop_underpromotion_capture(self):
        # d7×e8=B  (White pawn on d7, Black piece on e8)
        board = chess.Board("8/4P3/8/2k5/8/2K5/8/8 w - - 0 1")
        move = chess.Move.from_uci("d7e8b")
        assert move_to_index(move, board) == 3792


# ---------------------------------------------------------------------------
# Canonical orientation
# ---------------------------------------------------------------------------

class TestCanonicalOrientation:
    def test_e4_and_e5_same_index(self):
        # 1.e4 and 1...e5 are geometrically the same 2-square forward pawn push
        white_board = chess.Board()
        white_move = white_board.parse_san("e4")

        black_board = _board("e4")
        black_move = black_board.parse_san("e5")

        assert move_to_index(white_move, white_board) == move_to_index(black_move, black_board)

    def test_kingside_castle_same_from_both_sides(self):
        # White O-O (e1→g1) and Black O-O (e8→g8) are geometrically identical
        # from each player's perspective. Use FEN positions so we control whose turn it is.
        white_board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        # white_move = chess.Move(chess.E1, chess.G1)
        white_move = chess.Move.from_uci("e1g1")

        black_board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
        black_move = chess.Move.from_uci("e8g8")
        # black_move = chess.Move(chess.E8, chess.G8)

        assert move_to_index(white_move, white_board) == move_to_index(black_move, black_board)


# ---------------------------------------------------------------------------
# Round-trip: move_to_index → index_to_move
# ---------------------------------------------------------------------------

ROUND_TRIP_POSITIONS = [
    chess.Board(),                                                     # start
    _board("e4", "e5"),                                                # 1.e4 e5
    _board("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"),                   # Italian Game
    chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1"),         # ep + castling
    chess.Board("8/3P4/8/8/8/8/8/4K2k w - - 0 1"),                    # promotion
]


@pytest.mark.parametrize("board", ROUND_TRIP_POSITIONS)
def test_round_trip_all_legal_moves(board):
    """Every legal move from position encodes and decodes back to itself."""
    for move in list(board.legal_moves):
        idx = move_to_index(move, board)
        recovered = index_to_move(idx, board)
        assert move == recovered, (
            f"Round-trip failed for {move.uci()} in {board.fen()}: "
            f"index={idx}, recovered={recovered.uci()}"
        )


# ---------------------------------------------------------------------------
# legal_action_mask
# ---------------------------------------------------------------------------

class TestLegalActionMask:
    def test_starting_position_has_20_moves(self):
        board = chess.Board()
        mask = legal_action_mask(board)
        assert mask.dtype == bool
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.sum() == 20

    def test_after_e4_black_has_20_moves(self):
        board = _board("e4")
        assert legal_action_mask(board).sum() == 20

    def test_mask_agrees_with_legal_moves_count(self):
        # mid-game position with castling and en passant available
        board = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1")
        mask = legal_action_mask(board)
        assert mask.sum() == len(list(board.legal_moves))

    def test_only_legal_indices_are_set(self):
        board = chess.Board()
        mask = legal_action_mask(board)
        expected = {move_to_index(m, board) for m in board.legal_moves}
        assert set(np.where(mask)[0].tolist()) == expected


# ---------------------------------------------------------------------------
# Board plane layout
# ---------------------------------------------------------------------------

class TestEncodePlanes:
    def test_white_pawns_on_starting_rank(self):
        # White to move at start — our pawns (plane 0) on rank 1
        planes = encode(chess.Board())
        pawn_plane = planes[0]
        assert pawn_plane[1].sum() == 8    # rank 1 fully occupied
        assert pawn_plane[0].sum() == 0
        assert pawn_plane[2:].sum() == 0

    def test_black_pawns_on_canonical_rank_1_when_black_to_move(self):
        # After 1.e4, Black to move — their pawns (plane 0 = ours) should be on canonical rank 1
        board = _board("e4")
        planes = encode(board)
        pawn_plane = planes[0]  # current player's pawns
        assert pawn_plane[1].sum() == 8

    def test_opponent_pawns_on_canonical_rank_6(self):
        board = chess.Board()  # White to move — opponent (Black) pawns on canonical rank 6
        planes = encode(board)
        assert planes[6, 6].sum() == 8   # plane 6 = opponent pawns, rank 6

    def test_castling_all_rights_at_start(self):
        planes = encode(chess.Board())
        # All four castling planes should be 1 everywhere
        for p in range(12, 16):
            assert planes[p].min() == 1.0

    def test_castling_plane_swaps_for_black(self):
        # After 1.e4, Black to move: plane 12 = Black's kingside right
        board = _board("e4")
        planes = encode(board)
        assert planes[12].min() == 1.0   # Black kingside still intact

    def test_no_ep_plane_at_start(self):
        assert encode(chess.Board())[16].sum() == 0.0

    def test_ep_plane_set_when_capture_is_legal(self):
        # After 1.e4 d5 2.e5 f5, White's e5 pawn is adjacent to Black's f5 pawn
        # on the same rank — White can legally capture exf6.
        # It's White to move: ep target f6 is real rank 5, file 5 (no flip).
        board = _board("e4", "d5", "e5", "f5")
        assert board.ep_square == chess.F6
        ep_plane = encode(board)[16]
        assert ep_plane.sum() == 1.0
        assert ep_plane[5, 5] == 1.0   # f6: rank 5, file 5, no flip (White to move)

    def test_fifty_move_counter_normalized(self):
        board = chess.Board()
        board.halfmove_clock = 50
        planes = encode(board)
        assert planes[17, 0, 0] == pytest.approx(0.5)

    def test_output_shape_and_dtype(self):
        planes = encode(chess.Board())
        assert planes.shape == (18, 8, 8)
        assert planes.dtype == np.float32
