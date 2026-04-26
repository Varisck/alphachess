"""IterableDataset over pretraining record shards.

The dataset reads ``.npz`` shards (written by :mod:`alphachess.pretrain.db_ingest`)
one at a time from the configured ``Storage`` and yields
``(state, policy_target, value_target)`` triples for the trainer.

Train/val split is **by shard index** — the last ``ceil(val_split * num_shards)``
shards are validation. This keeps the split deterministic and reproducible
across machines without needing a row-level shuffle of the entire dataset.
"""

from __future__ import annotations

import io
import math
import random
from typing import Literal

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from alphachess.storage import Storage


def _split_shards(
    shards: list[str], split: Literal["train", "val"], val_split: float
) -> list[str]:
    if not shards:
        return []
    n_val = max(1, math.ceil(len(shards) * val_split))
    n_val = min(n_val, len(shards))
    if split == "val":
        return shards[-n_val:]
    return shards[:-n_val] if n_val < len(shards) else []


def _partition_for_worker(shards: list[str]) -> list[str]:
    info = get_worker_info()
    if info is None:
        return list(shards)
    return [s for i, s in enumerate(shards) if i % info.num_workers == info.id]


class PretrainDataset(IterableDataset):
    """Iterates records across shards. One shard in memory at a time."""

    def __init__(
        self,
        storage: Storage,
        records_subdir: str = "pretrain_records",
        split: Literal["train", "val"] = "train",
        val_split: float = 0.05,
        shuffle: bool | None = None,
    ):
        super().__init__()
        self.storage = storage
        self.records_subdir = records_subdir
        self.split = split
        self.val_split = val_split
        # Train shuffles, val keeps deterministic order, unless overridden.
        self.shuffle = shuffle if shuffle is not None else (split == "train")

        all_shards = sorted(
            storage.list(records_subdir, suffix=".npz")
        )
        self.shards = _split_shards(all_shards, split, val_split)

    def __iter__(self):
        my_shards = _partition_for_worker(self.shards)
        if self.shuffle:
            my_shards = list(my_shards)
            random.shuffle(my_shards)

        for name in my_shards:
            raw = self.storage.read_bytes(f"{self.records_subdir}/{name}")
            with np.load(io.BytesIO(raw)) as data:
                states = data["states"]
                policies = data["policy_targets"]
                values = data["value_targets"]

            n = len(states)
            order = np.random.permutation(n) if self.shuffle else np.arange(n)
            for i in order:
                yield (
                    torch.from_numpy(states[i]),
                    int(policies[i]),
                    torch.tensor(values[i], dtype=torch.float32),
                )
