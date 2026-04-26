"""
Project-wide configuration using Pydantic.

Every component loads the Config object.
To override fileds with env variables use prefix ALPHACHESS__

Log config.hash() on startup
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

class StorageConfig(BaseModel):
    root_uri: str = "file:///data/alphachess/run_local"


class GameConfig(BaseModel):
    action_space: int = 4672
    input_planes: int = 18


# 
# Some template for next modules to implement
#    - Check values when modules are implemented
# 

class NNConfig(BaseModel):
    num_blocks: int = 10
    channels: int = 128
    policy_head_channels: int = 73
    value_head_hidden: int = 256


class MCTSConfig(BaseModel):
    num_simulations: int = 400
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    virtual_loss: float = 1.0


class SelfplayConfig(BaseModel):
    games_per_worker_batch: int = 32
    temperature_moves: int = 30
    resign_threshold: float = -0.9
    resign_disable_pct: float = 0.1
    max_game_plies: int = 512


class TrainConfig(BaseModel):
    batch_size: int = 1024
    learning_rate: float = 0.005
    lr_milestones: list[int] = Field(default_factory=lambda: [100_000, 300_000])
    window_size_games: int = 100_000
    min_games_before_training: int = 1000
    steps_per_checkpoint: int = 1000
    weight_decay: float = 1e-4


class PretrainConfig(BaseModel):
    pgn_glob: str = "data/pgn/*.pgn"
    min_elo: int = 2000
    min_game_plies: int = 20
    max_positions: int | None = 2_000_000
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 1024
    val_split: float = 0.05
    early_stop_top1: float = 0.50
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "twic"
    mongo_collection: str = "games"
    shard_size: int = 50_000
    records_subdir: str = "pretrain_records"
    value_loss_weight: float = 1.0
    moves_per_game: int | None = 5


class OrchestratorConfig(BaseModel):
    role: Literal["selfplay", "train", "eval", "pretrain", "local"] = "local"
    selfplay_workers: int = 2


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ALPHACHESS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    storage: StorageConfig = Field(default_factory=StorageConfig)
    game: GameConfig = Field(default_factory=GameConfig)
    nn: NNConfig = Field(default_factory=NNConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    selfplay: SelfplayConfig = Field(default_factory=SelfplayConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    pretrain: PretrainConfig = Field(default_factory=PretrainConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    @classmethod
    def from_env(cls) -> "Config":
        return cls()

    def hash(self) -> str:
        serialized = self.model_dump_json(indent=None)
        return hashlib.md5(serialized.encode()).hexdigest()