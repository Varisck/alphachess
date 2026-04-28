"""Tests for alphachess.pretrain.db_ingest."""

from __future__ import annotations

import io
import json
import uuid

import chess
import numpy as np
import pytest

from alphachess.config import Config
from alphachess.pretrain.db_ingest import (
    MANIFEST_NAME,
    ingest,
    parse_result,
    parse_san_moves,
    select_move_indices,
)
from alphachess.storage import Storage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def storage():
    return Storage(f"memory://pretrain-ingest-{uuid.uuid4().hex}")


@pytest.fixture
def config():
    cfg = Config()
    cfg.pretrain.shard_size = 10
    cfg.pretrain.min_game_plies = 2
    cfg.pretrain.max_positions = None
    cfg.pretrain.moves_per_game = None
    return cfg


def _load_shard(storage: Storage, subdir: str, shard_id: int) -> dict:
    raw = storage.read_bytes(f"{subdir}/{shard_id:06d}.npz")
    npz = np.load(io.BytesIO(raw))
    return {
        "states": npz["states"],
        "policy_targets": npz["policy_targets"],
        "value_targets": npz["value_targets"],
    }


def _read_manifest(storage: Storage, subdir: str) -> dict:
    raw = storage.read_bytes(f"{subdir}/{MANIFEST_NAME}")
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# select_move_indices
# ---------------------------------------------------------------------------

class TestSelectMoveIndices:
    def test_none_returns_all(self):
        assert select_move_indices(10, None) == set(range(10))

    def test_zero_returns_empty(self):
        assert select_move_indices(10, 0) == set()

    def test_more_than_available_returns_all(self):
        assert select_move_indices(5, 100) == set(range(5))

    def test_count_matches(self):
        rng = np.random.default_rng(0)
        for num_moves in [10, 20, 40, 100]:
            for k in [2, 3, 5, 10]:
                if k <= num_moves:
                    idx = select_move_indices(num_moves, k, rng)
                    assert len(idx) == k, (num_moves, k, idx)

    def test_no_duplicates(self):
        rng = np.random.default_rng(0)
        idx = select_move_indices(20, 10, rng)
        assert len(idx) == len(set(idx))


# ---------------------------------------------------------------------------
# parse_result
# ---------------------------------------------------------------------------

class TestParseResult:
    def test_white_win(self):
        assert parse_result("1-0") == 1

    def test_black_win(self):
        assert parse_result("0-1") == -1

    def test_draw(self):
        assert parse_result("1/2-1/2") == 0

    def test_unfinished(self):
        assert parse_result("*") is None

    def test_none(self):
        assert parse_result(None) is None


# ---------------------------------------------------------------------------
# parse_san_moves — robustness against PGN noise
# ---------------------------------------------------------------------------

class TestParseSanMoves:
    def test_basic(self):
        moves = parse_san_moves("1. e4 e5 2. Nf3 Nc6", "*")
        assert [m.uci() for m in moves] == ["e2e4", "e7e5", "g1f3", "b8c6"]

    def test_strips_comments(self):
        moves = parse_san_moves("1. e4 {good} e5 2. Nf3 {best} Nc6", "*")
        assert [m.uci() for m in moves] == ["e2e4", "e7e5", "g1f3", "b8c6"]

    def test_strips_variations(self):
        moves = parse_san_moves("1. e4 (1. d4 d5) e5 2. Nf3 Nc6", "*")
        assert [m.uci() for m in moves] == ["e2e4", "e7e5", "g1f3", "b8c6"]

    def test_strips_nags(self):
        moves = parse_san_moves("1. e4 $1 e5 $2 2. Nf3 Nc6", "*")
        assert [m.uci() for m in moves] == ["e2e4", "e7e5", "g1f3", "b8c6"]


# ---------------------------------------------------------------------------
# ingest — end-to-end with hand-crafted games
# ---------------------------------------------------------------------------

class TestIngest:
    def test_writes_shard_and_manifest(self, storage, config):
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
        ]
        n = ingest(config, storage=storage, games=iter(games))
        assert n == 4
        assert storage.exists(f"{config.pretrain.records_subdir}/{MANIFEST_NAME}")
        manifest = _read_manifest(storage, config.pretrain.records_subdir)
        assert manifest["total_positions"] == 4
        assert manifest["num_shards"] == 1

    def test_record_shapes(self, storage, config):
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
        ]
        ingest(config, storage=storage, games=iter(games))
        shard = _load_shard(storage, config.pretrain.records_subdir, 0)
        assert shard["states"].dtype == np.float32
        assert shard["states"].shape == (6, 18, 8, 8)
        assert shard["policy_targets"].dtype == np.int32
        assert shard["policy_targets"].shape == (6,)
        assert shard["value_targets"].dtype == np.float32
        assert shard["value_targets"].shape == (6,)

    def test_keeps_repeated_position_across_games(self, storage, config):
        # Two identical games — frequency must be preserved (no dedup).
        # Value labels also have opposite signs, which is the signal we want
        # to keep so the value head averages toward the empirical win-rate.
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
            {"moves": "1. e4 e5 2. Nf3 Nc6", "result": "0-1",
             "white_elo": 2500, "black_elo": 2500},
        ]
        n = ingest(config, storage=storage, games=iter(games))
        assert n == 8
        shard = _load_shard(storage, config.pretrain.records_subdir, 0)
        # Game 1 (1-0): values [+1, -1, +1, -1]; game 2 (0-1): [-1, +1, -1, +1].
        assert list(shard["value_targets"]) == [
            1.0, -1.0, 1.0, -1.0,
            -1.0, 1.0, -1.0, 1.0,
        ]

    def test_keeps_shared_prefix_across_branching_games(self, storage, config):
        # Same first three plies, then divergence. Both games' positions are
        # written in full — no dedup of the shared prefix.
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
            {"moves": "1. e4 e5 2. Nf3 Nf6", "result": "0-1",
             "white_elo": 2500, "black_elo": 2500},
        ]
        n = ingest(config, storage=storage, games=iter(games))
        assert n == 8

    def test_value_target_perspective_white_wins(self, storage, config):
        # White wins. From White's move (ply 0, 2, ...), value = +1.
        # From Black's move (ply 1, 3, ...), value = -1.
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
        ]
        ingest(config, storage=storage, games=iter(games))
        shard = _load_shard(storage, config.pretrain.records_subdir, 0)
        # ply 0 White -> +1, ply 1 Black -> -1, ply 2 White -> +1, ply 3 Black -> -1
        assert list(shard["value_targets"]) == [1.0, -1.0, 1.0, -1.0]

    def test_value_target_perspective_black_wins(self, storage, config):
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6", "result": "0-1",
             "white_elo": 2500, "black_elo": 2500},
        ]
        ingest(config, storage=storage, games=iter(games))
        shard = _load_shard(storage, config.pretrain.records_subdir, 0)
        assert list(shard["value_targets"]) == [-1.0, 1.0, -1.0, 1.0]

    def test_value_target_draw(self, storage, config):
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6", "result": "1/2-1/2",
             "white_elo": 2500, "black_elo": 2500},
        ]
        ingest(config, storage=storage, games=iter(games))
        shard = _load_shard(storage, config.pretrain.records_subdir, 0)
        assert list(shard["value_targets"]) == [0.0, 0.0, 0.0, 0.0]

    def test_skips_short_games(self, storage, config):
        config.pretrain.min_game_plies = 10
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
        ]
        n = ingest(config, storage=storage, games=iter(games))
        assert n == 0

    def test_skips_unfinished_result(self, storage, config):
        games = [
            {"moves": "1. e4 e5", "result": "*",
             "white_elo": 2500, "black_elo": 2500},
        ]
        n = ingest(config, storage=storage, games=iter(games))
        assert n == 0

    def test_max_positions_caps_output(self, storage, config):
        config.pretrain.max_positions = 3
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
        ]
        n = ingest(config, storage=storage, games=iter(games))
        assert n == 3

    def test_sharding_splits_at_shard_size(self, storage, config):
        config.pretrain.shard_size = 3
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
        ]
        n = ingest(config, storage=storage, games=iter(games))
        assert n == 8
        # 8 positions, shard_size=3 -> shards of 3, 3, 2
        manifest = _read_manifest(storage, config.pretrain.records_subdir)
        assert manifest["num_shards"] == 3
        s0 = _load_shard(storage, config.pretrain.records_subdir, 0)
        s1 = _load_shard(storage, config.pretrain.records_subdir, 1)
        s2 = _load_shard(storage, config.pretrain.records_subdir, 2)
        assert s0["states"].shape[0] == 3
        assert s1["states"].shape[0] == 3
        assert s2["states"].shape[0] == 2

    def test_moves_per_game_caps_per_game(self, storage, config):
        # Game has 10 plies; sampling 4 should yield 4 records per game.
        config.pretrain.moves_per_game = 4
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7",
             "result": "1-0", "white_elo": 2500, "black_elo": 2500},
        ]
        n = ingest(config, storage=storage, games=iter(games))
        assert n == 4

    def test_moves_per_game_none_keeps_all(self, storage, config):
        config.pretrain.moves_per_game = None
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
        ]
        n = ingest(config, storage=storage, games=iter(games))
        assert n == 4

    def test_moves_per_game_random_sampling(self, storage, config):
        # 8-ply game with moves_per_game=2 must yield exactly 2 records,
        # and value targets must be valid (+1 or -1 since white wins).
        config.pretrain.moves_per_game = 2
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6",
             "result": "1-0", "white_elo": 2500, "black_elo": 2500},
        ]
        n = ingest(config, storage=storage, games=iter(games), seed=0)
        assert n == 2
        shard = _load_shard(storage, config.pretrain.records_subdir, 0)
        assert all(v in (1.0, -1.0) for v in shard["value_targets"])

    def test_idempotent_when_manifest_exists(self, storage, config):
        games = [
            {"moves": "1. e4 e5 2. Nf3 Nc6", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
        ]
        ingest(config, storage=storage, games=iter(games))
        # Second call: manifest is present, ingest should be a no-op.
        n = ingest(config, storage=storage, games=iter(games))
        assert n == 0

    def test_policy_target_matches_encoder(self, storage, config):
        from alphachess.game.encoding import move_to_index

        games = [
            {"moves": "1. e4 e5", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
        ]
        ingest(config, storage=storage, games=iter(games))
        shard = _load_shard(storage, config.pretrain.records_subdir, 0)

        # Reconstruct expected indices.
        board = chess.Board()
        e4 = chess.Move.from_uci("e2e4")
        expected_0 = move_to_index(e4, board)
        board.push(e4)
        e5 = chess.Move.from_uci("e7e5")
        expected_1 = move_to_index(e5, board)

        assert int(shard["policy_targets"][0]) == expected_0
        assert int(shard["policy_targets"][1]) == expected_1

    def test_null_move_discards_whole_game(self, storage, config):
        # A game with a null move embedded should produce no records.
        # A second clean game should still produce records normally.
        games = [
            {"moves": "1. e4 e5 2. -- Nc6", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
            {"moves": "1. e4 e5 2. Nf3 Nc6", "result": "1-0",
             "white_elo": 2500, "black_elo": 2500},
        ]
        n = ingest(config, storage=storage, games=iter(games))
        # First game discarded entirely; second contributes 4 records.
        assert n == 4