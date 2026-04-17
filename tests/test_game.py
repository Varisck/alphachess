"""
Tests for alphachess/game/chess_env.py.
"""

import random

import chess
import numpy as np
import pytest

from alphachess.game.chess_env import ChessEnv


# ---------------------------------------------------------------------------
# initial_state
# ---------------------------------------------------------------------------

def test_initial_state_is_standard_start():
    board = ChessEnv.initial_state()
    assert board == chess.Board()
    assert board.turn == chess.WHITE


# ---------------------------------------------------------------------------
# apply (mutates in place, supports push/pop)
# ---------------------------------------------------------------------------

def test_apply_mutates_board():
    board = ChessEnv.initial_state()
    mask = ChessEnv.legal_action_mask(board)
    idx = int(np.where(mask)[0][0])
    ChessEnv.apply(board, idx)
    assert board != chess.Board()

def test_apply_pop_restores_board():
    board = ChessEnv.initial_state()
    original_fen = board.fen()
    mask = ChessEnv.legal_action_mask(board)
    idx = int(np.where(mask)[0][0])
    ChessEnv.apply(board, idx)
    board.pop()
    assert board.fen() == original_fen

def test_apply_20_starting_moves_are_distinct():
    board = ChessEnv.initial_state()
    mask = ChessEnv.legal_action_mask(board)
    indices = np.where(mask)[0].tolist()
    assert len(indices) == 20

    fens = set()
    for idx in indices:
        ChessEnv.apply(board, int(idx))
        fens.add(board.fen())
        board.pop()

    assert len(fens) == 20


# ---------------------------------------------------------------------------
# is_terminal — checkmate
# ---------------------------------------------------------------------------

def test_checkmate_is_terminal_with_minus_one():
    # Fool's mate: 1.f3 e5 2.g4 Qh4# — White is mated (White to move after Qh4#)
    board = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4"]:
        board.push_san(san)
    done, value = ChessEnv.is_terminal(board)
    assert done is True
    assert value == pytest.approx(-1.0)

def test_back_rank_mate_is_terminal():
    board = chess.Board()
    for san in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7"]:
        board.push_san(san)
    done, value = ChessEnv.is_terminal(board)
    assert done is True
    assert value == pytest.approx(-1.0)

# ---------------------------------------------------------------------------
# is_terminal — draws
# ---------------------------------------------------------------------------

def test_stalemate_is_terminal_with_zero():
    # Classic stalemate: Black king on a8, White king on a6, White rook on b1
    # Black king can't move
    board = chess.Board("k7/8/K7/8/8/8/8/1R6 b - - 0 1")
    done, value = ChessEnv.is_terminal(board)
    assert done is True
    assert value == pytest.approx(0.0)

def test_in_progress_game_is_not_terminal():
    board = chess.Board()
    done, _ = ChessEnv.is_terminal(board)
    assert done is False


# ---------------------------------------------------------------------------
# encode (delegates to encoding.py — just sanity-check shape and dtype)
# ---------------------------------------------------------------------------

def test_encode_shape():
    board = chess.Board()
    tensor = ChessEnv.encode(board)
    assert tensor.shape == (18, 8, 8)
    assert tensor.dtype == np.float32


# ---------------------------------------------------------------------------
# Canonical orientation via ChessEnv
# ---------------------------------------------------------------------------

def test_own_pawns_always_on_canonical_rank_1():
    # Planes 0–5 are always the side-to-move's pieces.
    # At the start (White to move), White pawns are on rank 1.
    white_planes = ChessEnv.encode(chess.Board())
    assert white_planes[0, 1].sum() == 8   # White pawns, rank 1

    # After 1.e4 (Black to move), Black pawns should appear on canonical rank 1.
    black_board = chess.Board()
    black_board.push_san("e4")
    black_planes = ChessEnv.encode(black_board)
    assert black_planes[0, 1].sum() == 8   # Black pawns, canonical rank 1
