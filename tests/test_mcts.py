"""
Tests for alphachess/mcts/tree.py and alphachess/mcts/search.py.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from alphachess.config import MCTSConfig
from alphachess.game.encoding import (
    NUM_ACTIONS,
    legal_action_mask,
    move_to_index,
)
from alphachess.mcts.search import MCTS
from alphachess.mcts.tree import Tree
from alphachess.nn.inference import InferenceModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubModel(InferenceModel):
    """Stand-in for InferenceModel: uniform priors, fixed value."""

    def __init__(self, value: float = 0.0) -> None:
        self._value = value

    def predict_batch(
        self, encoded_boards: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        B = encoded_boards.shape[0]
        priors = np.full((B, NUM_ACTIONS), 1.0 / NUM_ACTIONS, dtype=np.float32)
        values = np.full(B, self._value, dtype=np.float32)
        return priors, values


def _board(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


# ---------------------------------------------------------------------------
# Tree.expand
# ---------------------------------------------------------------------------


class TestTreeExpand:
    def test_zeros_illegal_moves_and_normalizes(self):
        tree = Tree(max_nodes=10, config=MCTSConfig())
        board = chess.Board()
        priors = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)

        tree.expand(0, board, priors)

        legal = legal_action_mask(board)
        assert np.all(tree.P[0][~legal] == 0.0)
        assert tree.P[0].sum() == pytest.approx(1.0)
        assert tree.is_expanded[0]
        assert tree.boards[0] is board


# ---------------------------------------------------------------------------
# Tree.backup — sign convention is the bug-prone part
# ---------------------------------------------------------------------------


class TestTreeBackup:
    def test_flips_signs_along_path(self):
        # Manually build path: root(0) -> n1 -> n2.
        tree = Tree(max_nodes=10, config=MCTSConfig())
        n1 = tree._allocate_node(parent_id=0, action=100)
        n2 = tree._allocate_node(parent_id=n1, action=200)

        tree.backup(n2, value=0.6)

        # Edge n1 -> n2: stored from n1's perspective = -0.6
        assert tree.W[n1, 200] == pytest.approx(-0.6)
        assert tree.N[n1, 200] == 1
        # Edge root -> n1: flipped again, root's perspective = +0.6
        assert tree.W[0, 100] == pytest.approx(0.6)
        assert tree.N[0, 100] == 1

    def test_root_does_not_overflow_last_row(self):
        # Calling backup on the root must not touch N[-1] / W[-1].
        tree = Tree(max_nodes=5, config=MCTSConfig())
        before_N = tree.N[-1].copy()
        before_W = tree.W[-1].copy()

        tree.backup(node_id=0, value=0.3)

        assert np.array_equal(tree.N[-1], before_N)
        assert np.array_equal(tree.W[-1], before_W)


# ---------------------------------------------------------------------------
# Tree._puct_scores
# ---------------------------------------------------------------------------


class TestPuctScores:
    def test_illegal_moves_minus_inf_legal_finite(self):
        tree = Tree(max_nodes=10, config=MCTSConfig())
        board = chess.Board()
        priors = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
        tree.expand(0, board, priors)

        scores = tree._puct_scores(0)

        legal = legal_action_mask(board)
        assert np.all(scores[~legal] == -np.inf)
        assert np.all(np.isfinite(scores[legal]))


# ---------------------------------------------------------------------------
# Tree.root_visit_distribution
# ---------------------------------------------------------------------------


class TestRootVisitDistribution:
    def test_normalized_to_one(self):
        tree = Tree(max_nodes=10, config=MCTSConfig())
        tree.N[0, 100] = 5
        tree.N[0, 200] = 3
        tree.N[0, 300] = 2

        dist = tree.root_visit_distribution()

        assert dist.dtype == np.float32
        assert dist.sum() == pytest.approx(1.0)
        assert dist[100] == pytest.approx(0.5)
        assert dist[200] == pytest.approx(0.3)
        assert dist[300] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# MCTS.run smoke tests
# ---------------------------------------------------------------------------


class TestMCTSRun:
    def test_distribution_shape_and_sum(self):
        config = MCTSConfig(num_simulations=16)
        mcts = MCTS(_StubModel(value=0.0), config)

        pi = mcts.run(chess.Board(), add_root_noise=False)

        assert pi.shape == (NUM_ACTIONS,)
        assert pi.sum() == pytest.approx(1.0)

    def test_illegal_moves_zero(self):
        config = MCTSConfig(num_simulations=16)
        mcts = MCTS(_StubModel(value=0.0), config)

        pi = mcts.run(chess.Board(), add_root_noise=False)

        legal = legal_action_mask(chess.Board())
        assert np.all(pi[~legal] == 0.0)

    def test_no_noise_is_deterministic(self):
        config = MCTSConfig(num_simulations=16)
        mcts = MCTS(_StubModel(value=0.0), config)

        pi1 = mcts.run(chess.Board(), add_root_noise=False)
        pi2 = mcts.run(chess.Board(), add_root_noise=False)

        np.testing.assert_array_equal(pi1, pi2)


# ---------------------------------------------------------------------------
# Sign-convention tests — the day-one tests from the plan
# ---------------------------------------------------------------------------


class TestTerminalValue:
    def test_minus_one_when_side_to_move_is_mated(self):
        # Fool's mate: 1.f3 e5 2.g4?? Qh4#
        board = _board("f3", "e5", "g4", "Qh4#")
        assert board.is_checkmate()
        mcts = MCTS(_StubModel(), MCTSConfig())
        assert mcts._terminal_value(board) == -1.0

    def test_zero_on_stalemate(self):
        # Black king a8, White queen b6, White king h1; Black to move, no legal moves.
        board = chess.Board("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
        assert board.is_stalemate()
        mcts = MCTS(_StubModel(), MCTSConfig())
        assert mcts._terminal_value(board) == 0.0

    def test_none_on_non_terminal(self):
        mcts = MCTS(_StubModel(), MCTSConfig())
        assert mcts._terminal_value(chess.Board()) is None


class TestMateInOne:
    def test_mcts_finds_mating_move(self):
        # Scholar's mate setup. After 3...Nf6, White can play Qxf7#.
        board = _board("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6")
        qxf7 = board.parse_san("Qxf7#")
        mating_idx = move_to_index(qxf7, board)

        # Many simulations to guarantee MCTS converges on the mating edge:
        # Q at the mating edge is exactly +1 (highest possible) while every
        # other legal edge has Q = 0 with the stub's value=0.
        config = MCTSConfig(num_simulations=200)
        mcts = MCTS(_StubModel(value=0.0), config)

        pi = mcts.run(board, add_root_noise=False)

        assert int(np.argmax(pi)) == mating_idx
