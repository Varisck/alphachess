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
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Iterator

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


def select_move_indices(
    num_moves: int,
    moves_per_game: int | None,
    rng: np.random.Generator | None = None,
) -> set[int]:
    """Randomly sample move indices from a game without replacement.

    With ``moves_per_game = None`` (or ``>= num_moves``) returns all indices.
    Otherwise returns ``moves_per_game`` indices drawn uniformly at random
    from ``[0, num_moves - 1]``.
    """
    if num_moves <= 0:
        return set()
    if moves_per_game is None or moves_per_game >= num_moves:
        return set(range(num_moves))
    if moves_per_game <= 0:
        return set()
    if rng is None:
        rng = np.random.default_rng()
    return set(rng.choice(num_moves, size=moves_per_game, replace=False).tolist())


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


def _process_game(
    game_doc: dict,
    moves_per_game: int | None,
    min_game_plies: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[np.ndarray, int, float]], str | None]:
    """Process one game atomically.

    Returns ``(records, drop_reason)``. ``drop_reason`` is one of
    ``"no_outcome"``, ``"san_error"``, ``"short"``, ``"null_move"`` or
    ``None``. On any drop, ``records`` is empty.
    """
    outcome = parse_result(game_doc.get("result"))
    if outcome is None:
        return [], "no_outcome"

    try:
        moves = parse_san_moves(game_doc.get("moves", ""), game_doc["result"])
    except Exception:
        return [], "san_error"

    if len(moves) < min_game_plies:
        return [], "short"

    selected = select_move_indices(len(moves), moves_per_game, rng)

    board = chess.Board()
    records: list[tuple[np.ndarray, int, float]] = []
    for i, move in enumerate(moves):
        if move == chess.Move.null():
            return [], "null_move"
        if i in selected:
            value = outcome if board.turn == chess.WHITE else -outcome
            records.append(
                (encode(board), move_to_index(move, board), float(value))
            )
        board.push(move)
    return records, None


def _process_batch(
    batch: list[dict],
    moves_per_game: int | None,
    min_game_plies: int,
    seed: int | None,
) -> tuple[list[tuple[np.ndarray, int, float]], dict[str, int]]:
    """Worker entry point: process a batch of games, return all records."""
    rng = np.random.default_rng(seed)
    all_records: list[tuple[np.ndarray, int, float]] = []
    counts = {"null_move": 0, "san_error": 0, "no_outcome": 0, "short": 0}
    for game_doc in batch:
        records, status = _process_game(
            game_doc, moves_per_game, min_game_plies, rng
        )
        if status is not None:
            counts[status] += 1
        all_records.extend(records)
    return all_records, counts


def _iter_records_serial(
    games: Iterable[dict],
    config: Config,
    seed: int | None,
) -> Iterator[tuple[np.ndarray, int, float]]:
    rng = np.random.default_rng(seed)
    for game in games:
        records, status = _process_game(
            game,
            config.pretrain.moves_per_game,
            config.pretrain.min_game_plies,
            rng,
        )
        if status == "null_move":
            log.warning("null move; discarding game (result=%s)", game.get("result"))
        elif status == "san_error":
            log.warning("SAN parse error; skipping game")
        yield from records


def _iter_records_parallel(
    games: Iterable[dict],
    config: Config,
    seed: int | None,
    num_workers: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, int, float]]:
    """Yield records from worker processes, preserving submission order."""

    def batches() -> Iterator[list[dict]]:
        batch: list[dict] = []
        for game in games:
            batch.append(game)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    iter_batches = batches()
    pool = ProcessPoolExecutor(max_workers=num_workers)
    try:
        pending: list = []
        batch_idx = 0

        for _ in range(num_workers * 2):
            try:
                batch = next(iter_batches)
            except StopIteration:
                break
            worker_seed = (seed + batch_idx) if seed is not None else None
            pending.append(pool.submit(
                _process_batch, batch,
                config.pretrain.moves_per_game,
                config.pretrain.min_game_plies,
                worker_seed,
            ))
            batch_idx += 1

        while pending:
            future = pending.pop(0)
            records, counts = future.result()
            if counts["null_move"]:
                log.warning("discarded %d games with null moves", counts["null_move"])
            if counts["san_error"]:
                log.warning("skipped %d games with SAN errors", counts["san_error"])
            yield from records

            try:
                batch = next(iter_batches)
            except StopIteration:
                continue
            worker_seed = (seed + batch_idx) if seed is not None else None
            pending.append(pool.submit(
                _process_batch, batch,
                config.pretrain.moves_per_game,
                config.pretrain.min_game_plies,
                worker_seed,
            ))
            batch_idx += 1
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def ingest(
    config: Config,
    storage: Storage | None = None,
    games: Iterable[dict] | None = None,
    seed: int | None = None,
    num_workers: int = 1,
    batch_size: int = 200,
) -> int:
    """Read games, write record shards. Returns positions written.

    With ``num_workers > 1`` games are processed in worker processes in
    chunks of ``batch_size``; the main process reads the cursor and writes
    shards in submission order. With ``num_workers = 1`` (default) the
    serial path is used.

    Idempotent: if a manifest already exists under
    ``{subdir}/{MANIFEST_NAME}``, returns 0.
    """
    if storage is None:
        storage = Storage(config.storage.root_uri)
    subdir = config.pretrain.records_subdir

    if storage.exists(f"{subdir}/{MANIFEST_NAME}"):
        log.info("manifest exists at %s/%s, skipping ingest", subdir, MANIFEST_NAME)
        return 0

    if games is None:
        games = _iter_games_from_mongo(config)

    if num_workers > 1:
        records_iter = _iter_records_parallel(
            games, config, seed, num_workers, batch_size,
        )
    else:
        records_iter = _iter_records_serial(games, config, seed)

    buffer: list[tuple[np.ndarray, int, float]] = []
    shard_id = 0
    total_written = 0
    max_positions = config.pretrain.max_positions

    for record in records_iter:
        if max_positions is not None and total_written >= max_positions:
            break
        buffer.append(record)
        total_written += 1

        if len(buffer) >= config.pretrain.shard_size:
            write_shard(storage, subdir, shard_id, buffer)
            log.info("wrote shard %d (%d positions)", shard_id, len(buffer))
            buffer.clear()
            shard_id += 1

    if buffer:
        write_shard(storage, subdir, shard_id, buffer)
        log.info("wrote final shard %d (%d positions)", shard_id, len(buffer))
        num_shards = shard_id + 1
    else:
        num_shards = shard_id

    write_manifest(storage, subdir, total_written, num_shards, config)
    log.info("ingest complete: %d positions across %d shards", total_written, num_shards)
    return total_written
