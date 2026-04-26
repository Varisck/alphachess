"""Pretraining entry point.

Three steps in order, each conditional:

  1. If ``models/000000.pt`` exists already, exit.
  2. If the records manifest is missing, run the MongoDB ingest.
  3. Run the supervised trainer, which writes ``models/000000.pt``.

Re-runs are safe: ingest is skipped if the manifest is present, training
is skipped if the final checkpoint is present.

Usage:
    python scripts/pretrain.py --config configs/laptop.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

from alphachess.config import Config
from alphachess.pretrain.db_ingest import MANIFEST_NAME, ingest
from alphachess.pretrain.trainer import PRETRAIN_FINAL_GENERATION, train
from alphachess.storage import Storage


log = logging.getLogger(__name__)


def main(config_path: str) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config = Config.from_yaml(config_path)
    storage = Storage(config.storage.root_uri)

    final_rel = f"models/{PRETRAIN_FINAL_GENERATION:06d}.pt"
    if storage.exists(final_rel):
        log.info("%s already exists, nothing to do", final_rel)
        return 0

    manifest_rel = f"{config.pretrain.records_subdir}/{MANIFEST_NAME}"
    if not storage.exists(manifest_rel):
        log.info("no records manifest at %s, ingesting from MongoDB", manifest_rel)
        ingest(config, storage=storage)

    log.info("starting supervised trainer")
    train(config, storage=storage)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain AlphaChessNet from master games")
    parser.add_argument(
        "--config",
        required=True,
        help="path to YAML config file",
    )
    args = parser.parse_args()
    sys.exit(main(args.config))
