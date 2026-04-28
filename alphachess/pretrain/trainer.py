"""Supervised pretraining loop.

Trains network on the shards written by db_ingest.

Produces ``models/000000.pt``

Per-step and per-epoch logs include policy_loss, value_loss, and total_loss
as separate fields.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from alphachess.config import Config
from alphachess.nn.model import AlphaChessNet
from alphachess.pretrain.dataset import PretrainDataset
from alphachess.storage import Storage


log = logging.getLogger(__name__)

PRETRAIN_BEST = "models/_pretrain_best.pt"
PRETRAIN_FINAL_GENERATION = 0
DEFAULT_LOG_PATH = "data/logs/pretrain.jsonl"
DEFAULT_TRAIN_LOG_INTERVAL = 50


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(config: Config) -> AlphaChessNet:
    return AlphaChessNet(
        num_blocks=config.nn.num_blocks,
        channels=config.nn.channels,
        input_planes=config.game.input_planes,
        action_space=config.game.action_space,
    )


def _save_full_checkpoint(
    storage: Storage,
    model: AlphaChessNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    buf = io.BytesIO()
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": {
                "num_blocks": len(model.tower),
                "channels": model.start_bn.num_features,
                "input_planes": model.start_conv.in_channels,
                "action_space": model.action_space,
            },
        },
        buf,
    )
    storage.atomic_put(PRETRAIN_BEST, buf.getvalue())


class JsonlLogger:
    """Append-only JSON-Lines metrics logger."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _compute_top_k(logits: torch.Tensor, targets: torch.Tensor, k: int):
    topk = logits.topk(k, dim=-1).indices
    correct = (topk == targets.unsqueeze(-1)).any(dim=-1).sum().item()
    return correct


def _evaluate(
    model: AlphaChessNet,
    loader: DataLoader,
    device: torch.device,
    value_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    policy_loss_fn = nn.CrossEntropyLoss(reduction="sum")
    value_loss_fn = nn.MSELoss(reduction="sum")

    total_policy = 0.0
    total_value = 0.0
    correct_top1 = 0
    correct_top5 = 0
    n = 0

    with torch.no_grad():
        for states, policies, values in loader:
            states = states.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True).long()
            values = values.to(device, non_blocking=True).float().unsqueeze(-1)

            logits, value_pred = model(states)
            total_policy += policy_loss_fn(logits, policies).item()
            total_value += value_loss_fn(value_pred, values).item()
            correct_top1 += _compute_top_k(logits, policies, 1)
            correct_top5 += _compute_top_k(logits, policies, 5)
            n += states.size(0)

    if n == 0:
        return {
            "policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0,
            "top1": 0.0, "top5": 0.0,
        }

    policy_loss = total_policy / n
    value_loss = total_value / n
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "total_loss": policy_loss + value_loss_weight * value_loss,
        "top1": correct_top1 / n,
        "top5": correct_top5 / n,
    }


def train(
    config: Config,
    storage: Storage | None = None,
    log_path: str | os.PathLike[str] = DEFAULT_LOG_PATH,
    train_log_interval: int = DEFAULT_TRAIN_LOG_INTERVAL,
    num_workers: int = 0,
) -> str:
    """Run the supervised pretraining loop.

    Returns the relative storage path of the final checkpoint
    (``models/000000.pt``).
    """
    if storage is None:
        storage = Storage(config.storage.root_uri)

    device = _select_device()
    log.info("training on device: %s", device)

    train_ds = PretrainDataset(
        storage,
        records_subdir=config.pretrain.records_subdir,
        split="train",
        val_split=config.pretrain.val_split,
        cache_shards=True,
    )
    val_ds = PretrainDataset(
        storage,
        records_subdir=config.pretrain.records_subdir,
        split="val",
        val_split=config.pretrain.val_split,
        cache_shards=True,
    )
    loader_kwargs = {
        "batch_size": config.pretrain.batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)

    model = _build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.pretrain.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    vlw = config.pretrain.value_loss_weight

    metrics_log = JsonlLogger(log_path)

    final_path = f"models/{PRETRAIN_FINAL_GENERATION:06d}.pt"
    early_stopped = False

    best_val_loss = float("inf")

    for epoch in range(1, config.pretrain.epochs + 1):
        model.train()
        step = 0
        epoch_start = time.monotonic()

        for states, policies, values in train_loader:
            states = states.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True).long()
            values = values.to(device, non_blocking=True).float().unsqueeze(-1)

            logits, value_pred = model(states)
            policy_loss = policy_loss_fn(logits, policies)
            value_loss = value_loss_fn(value_pred, values)
            total_loss = policy_loss + vlw * value_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            step += 1
            if step % train_log_interval == 0:
                metrics_log.log({
                    "epoch": epoch,
                    "step": step,
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "total_loss": float(total_loss.item()),
                    "split": "train",
                })

        elapsed = time.monotonic() - epoch_start

        val = _evaluate(model, val_loader, device, vlw)
        metrics_log.log({
            "epoch": epoch,
            "policy_loss": val["policy_loss"],
            "value_loss": val["value_loss"],
            "total_loss": val["total_loss"],
            "top1": val["top1"],
            "top5": val["top5"],
            "split": "val",
            "elapsed_sec": elapsed,
        })
        log.info(
            "epoch %d: val policy=%.4f value=%.4f total=%.4f top1=%.4f top5=%.4f (%.1fs)",
            epoch, val["policy_loss"], val["value_loss"], val["total_loss"],
            val["top1"], val["top5"], elapsed,
        )

        if val["total_loss"] < best_val_loss:
            best_val_loss = val["total_loss"]
            _save_full_checkpoint(storage, model, optimizer, epoch)
            log.info("epoch %d: new best val loss %.4f — checkpoint saved", epoch, best_val_loss)
        else:
            log.info("epoch %d: val loss %.4f did not improve over %.4f — skipping checkpoint",
                     epoch, val["total_loss"], best_val_loss)

        if val["top1"] >= config.pretrain.early_stop_top1:
            log.info(
                "early stop: val top1 %.4f >= %.4f",
                val["top1"], config.pretrain.early_stop_top1,
            )
            early_stopped = True
            break

    best_ckpt = torch.load(io.BytesIO(storage.read_bytes(PRETRAIN_BEST)), map_location="cpu")
    model.cpu().load_state_dict(best_ckpt["model_state"])
    model.save_to(storage, generation=PRETRAIN_FINAL_GENERATION)
    log.info(
        "wrote final checkpoint %s (best val loss=%.4f, early_stopped=%s)",
        final_path, best_val_loss, early_stopped,
    )

    log_rel = "logs/pretrain.jsonl"
    storage.write_bytes(log_rel, Path(log_path).read_bytes())
    log.info("uploaded training log to %s", log_rel)

    return final_path
