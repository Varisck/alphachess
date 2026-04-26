"""Pretraining ingest: from games in MongoDB to record shards.

Reads games from a MongoDB, and writes 
(state, policy_target, value_target) records to ``.npz`` file using the Storage.

``config.pretrain.moves_per_game`` defines how many moves in each game are sampled,
sampling is uniform between white and black. If None all positions are kept
"""

from __future__ import annotations

import io
import json
import logging
from typing import Iterable

import chess
import chess.pgn
import numpy as np
from pymongo import MongoClient

from alphachess.config import Config
from alphachess.game.encoding import encode, move_to_index
from alphachess.storage import Storage


log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MANIFEST_NAME = "_manifest.json"


def parse_result(result: str | None) -> int | None:
    """Return outcome from White's perspective: +1, 0, -1, or None."""
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    if result == "1/2-1/2":
        return 0
    return None


def parse_san_moves(moves_text: str, result: str) -> list[chess.Move]:
    """Parse a string of moves into a list of chess.Move
    """
    pgn_text = f'[Result "{result}"]\n\n{moves_text}'
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return []
    return list(game.mainline_moves())


def _shard_filename(shard_id: int) -> str:
    return f"{shard_id:06d}.npz"


def select_move_indices(num_moves: int, moves_per_game: int | None) -> set[int]:
    """Indices of sampled moves from a game, sampling is uniform.
        
    With ``moves_per_game = None`` (or ``>= num_moves``) returns all indices.
    Otherwise returns ``moves_per_game`` indices spread evenly over
    ``[0, num_moves - 1]``.
    """
    if num_moves <= 0:
        return set()
    if moves_per_game is None or moves_per_game >= num_moves:
        return set(range(num_moves))
    if moves_per_game <= 0:
        return set()
    raw = np.linspace(0, num_moves - 1, moves_per_game)
    return {int(round(x)) for x in raw}


def write_shard(
    storage: Storage,
    subdir: str,
    shard_id: int,
    buffer: list[tuple[np.ndarray, int, float]],
) -> None:
    """Compress and write one shard of records to ``{subdir}/{shard_id:06d}.npz``."""
    states = np.stack([rec[0] for rec in buffer]).astype(np.float32, copy=False)
    policies = np.fromiter((rec[1] for rec in buffer), dtype=np.int32, count=len(buffer))
    values = np.fromiter((rec[2] for rec in buffer), dtype=np.float32, count=len(buffer))
    bio = io.BytesIO()
    np.savez_compressed(
        bio,
        states=states,
        policy_targets=policies,
        value_targets=values,
    )
    storage.write_bytes(f"{subdir}/{_shard_filename(shard_id)}", bio.getvalue())


def write_manifest(
    storage: Storage,
    subdir: str,
    total_positions: int,
    num_shards: int,
    config: Config,
) -> None:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "total_positions": total_positions,
        "num_shards": num_shards,
        "shard_size": config.pretrain.shard_size,
        "min_elo": config.pretrain.min_elo,
        "min_game_plies": config.pretrain.min_game_plies,
        "max_positions": config.pretrain.max_positions,
        "moves_per_game": config.pretrain.moves_per_game,
        "input_planes": config.game.input_planes,
        "action_space": config.game.action_space,
        "mongo_db": config.pretrain.mongo_db,
        "mongo_collection": config.pretrain.mongo_collection,
    }
    storage.write_bytes(
        f"{subdir}/{MANIFEST_NAME}",
        json.dumps(manifest, indent=2).encode("utf-8"),
    )


def _iter_games_from_mongo(config: Config) -> Iterable[dict]:
    """Yield game documents from MongoDB filtered by minimum Elo on both sides."""
    client = MongoClient(config.pretrain.mongo_uri)
    coll = client[config.pretrain.mongo_db][config.pretrain.mongo_collection]
    cursor = coll.find(
        {
            "white_elo": {"$gte": config.pretrain.min_elo},
            "black_elo": {"$gte": config.pretrain.min_elo},
        },
        projection={"moves": 1, "result": 1, "white_elo": 1, "black_elo": 1},
        no_cursor_timeout=True,
    ).batch_size(500)
    try:
        yield from cursor
    finally:
        cursor.close()
        client.close()


def ingest(
    config: Config,
    storage: Storage | None = None,
    games: Iterable[dict] | None = None,
) -> int:
    """Read games, write record shards. Returns positions written.

    if a manifest already exists under ``{subdir}/{MANIFEST_NAME}``, returns 0.

    (The ``games`` parameter is for tests; production calls leave it as None
    and the function reads from MongoDB.)
    """
    if storage is None:
        storage = Storage(config.storage.root_uri)
    subdir = config.pretrain.records_subdir

    if storage.exists(f"{subdir}/{MANIFEST_NAME}"):
        log.info("manifest exists at %s/%s, skipping ingest", subdir, MANIFEST_NAME)
        return 0

    if games is None:
        games = _iter_games_from_mongo(config)

    buffer: list[tuple[np.ndarray, int, float]] = []
    shard_id = 0
    total_written = 0
    max_positions = config.pretrain.max_positions

    for game in games:
        if max_positions is not None and total_written >= max_positions:
            break

        outcome_white = parse_result(game.get("result"))
        if outcome_white is None:
            continue

        try:
            moves = parse_san_moves(game.get("moves", ""), game["result"])
        except Exception:
            log.warning("failed to parse SAN for game, skipping", exc_info=True)
            continue

        if len(moves) < config.pretrain.min_game_plies:
            continue

        selected = select_move_indices(len(moves), config.pretrain.moves_per_game)

        board = chess.Board()
        stop = False
        for i, move in enumerate(moves):
            if i in selected:
                value = outcome_white if board.turn == chess.WHITE else -outcome_white
                buffer.append(
                    (encode(board), move_to_index(move, board), float(value))
                )
                total_written += 1

                if len(buffer) >= config.pretrain.shard_size:
                    write_shard(storage, subdir, shard_id, buffer)
                    log.info("wrote shard %d (%d positions)", shard_id, len(buffer))
                    buffer.clear()
                    shard_id += 1

                if max_positions is not None and total_written >= max_positions:
                    stop = True
                    break
            board.push(move)

        if stop:
            break

    if buffer:
        write_shard(storage, subdir, shard_id, buffer)
        log.info("wrote final shard %d (%d positions)", shard_id, len(buffer))
        num_shards = shard_id + 1
    else:
        num_shards = shard_id

    write_manifest(storage, subdir, total_written, num_shards, config)
    log.info("ingest complete: %d positions across %d shards", total_written, num_shards)
    return total_written
