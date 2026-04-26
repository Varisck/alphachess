"""Tests for alphachess.pretrain.dataset."""

from __future__ import annotations

import io
import uuid

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from alphachess.pretrain.dataset import PretrainDataset
from alphachess.storage import Storage


SUBDIR = "pretrain_records"


@pytest.fixture
def storage():
    return Storage(f"memory://pretrain-dataset-{uuid.uuid4().hex}")


def _write_fake_shard(
    storage: Storage,
    shard_id: int,
    n: int,
    seed_offset: int = 0,
    policy_offset: int = 0,
) -> None:
    rng = np.random.default_rng(shard_id + seed_offset)
    states = rng.standard_normal((n, 18, 8, 8)).astype(np.float32)
    policies = (
        np.arange(n, dtype=np.int32) + policy_offset + shard_id * 10000
    )
    values = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    bio = io.BytesIO()
    np.savez_compressed(
        bio, states=states, policy_targets=policies, value_targets=values
    )
    storage.write_bytes(f"{SUBDIR}/{shard_id:06d}.npz", bio.getvalue())


# ---------------------------------------------------------------------------
# split logic
# ---------------------------------------------------------------------------

class TestSplit:
    def test_last_shards_are_val(self, storage):
        for sid in range(10):
            _write_fake_shard(storage, sid, n=4)
        train = PretrainDataset(storage, SUBDIR, split="train", val_split=0.2)
        val = PretrainDataset(storage, SUBDIR, split="val", val_split=0.2)
        assert train.shards == [f"{i:06d}.npz" for i in range(8)]
        assert val.shards == [f"{i:06d}.npz" for i in range(8, 10)]

    def test_val_is_at_least_one_shard(self, storage):
        for sid in range(3):
            _write_fake_shard(storage, sid, n=4)
        # 0.05 * 3 = 0.15 -> ceil -> 1
        val = PretrainDataset(storage, SUBDIR, split="val", val_split=0.05)
        assert len(val.shards) == 1

    def test_split_disjoint(self, storage):
        for sid in range(5):
            _write_fake_shard(storage, sid, n=4)
        train = PretrainDataset(storage, SUBDIR, split="train", val_split=0.2)
        val = PretrainDataset(storage, SUBDIR, split="val", val_split=0.2)
        assert set(train.shards).isdisjoint(set(val.shards))


# ---------------------------------------------------------------------------
# iteration
# ---------------------------------------------------------------------------

class TestIteration:
    def test_yields_correct_count(self, storage):
        for sid in range(4):
            _write_fake_shard(storage, sid, n=5)
        # 4 shards, val_split=0.25 -> last 1 is val (5 records)
        # train: 3 shards * 5 = 15
        train = PretrainDataset(storage, SUBDIR, split="train", val_split=0.25)
        items = list(train)
        assert len(items) == 15

        val = PretrainDataset(storage, SUBDIR, split="val", val_split=0.25)
        assert len(list(val)) == 5

    def test_yields_correct_types_and_shapes(self, storage):
        _write_fake_shard(storage, 0, n=3)
        _write_fake_shard(storage, 1, n=3)
        ds = PretrainDataset(storage, SUBDIR, split="val", val_split=0.5,
                             shuffle=False)
        items = list(ds)
        assert len(items) == 3
        state, policy, value = items[0]
        assert isinstance(state, torch.Tensor)
        assert state.dtype == torch.float32
        assert state.shape == (18, 8, 8)
        assert isinstance(policy, int)
        assert isinstance(value, torch.Tensor)
        assert value.shape == ()
        assert value.dtype == torch.float32

    def test_val_no_shuffle_preserves_order(self, storage):
        _write_fake_shard(storage, 0, n=4, policy_offset=0)
        _write_fake_shard(storage, 1, n=4, policy_offset=0)
        ds = PretrainDataset(storage, SUBDIR, split="val", val_split=0.5,
                             shuffle=False)
        policies = [int(p) for _, p, _ in ds]
        # val gets shard 1 only (last 50%); shard 1 policies are 10000..10003
        assert policies == [10000, 10001, 10002, 10003]

    def test_dataloader_batches_correctly(self, storage):
        for sid in range(3):
            _write_fake_shard(storage, sid, n=8)
        ds = PretrainDataset(storage, SUBDIR, split="val", val_split=1.0,
                             shuffle=False)
        loader = DataLoader(ds, batch_size=4)
        batches = list(loader)
        # 24 records / batch_size 4 -> 6 batches
        assert len(batches) == 6
        states, policies, values = batches[0]
        assert states.shape == (4, 18, 8, 8)
        assert policies.shape == (4,)
        assert values.shape == (4,)
