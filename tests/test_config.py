"""
Tests for alphachess.config.

"""

import os
from pathlib import Path

import pytest

from alphachess.config import Config


LAPTOP_YAML = Path(__file__).parent.parent / "configs" / "laptop.yaml"


class TestFromYaml:
    def test_loads_laptop_yaml(self):
        cfg = Config.from_yaml(LAPTOP_YAML)
        assert cfg.nn.num_blocks == 5
        assert cfg.nn.channels == 64

    def test_mcts_override(self):
        cfg = Config.from_yaml(LAPTOP_YAML)
        assert cfg.mcts.num_simulations == 200

    def test_defaults_fill_unspecified_sections(self):
        cfg = Config.from_yaml(LAPTOP_YAML)
        assert cfg.game.action_space == 4672
        assert cfg.game.input_planes == 18

    def test_orchestrator_role(self):
        cfg = Config.from_yaml(LAPTOP_YAML)
        assert cfg.orchestrator.role == "local"
        assert cfg.orchestrator.selfplay_workers == 2


class TestFromEnv:
    def test_env_overrides_nn_blocks(self, monkeypatch):
        monkeypatch.setenv("ALPHACHESS_NN__NUM_BLOCKS", "7")
        cfg = Config.from_env()
        assert cfg.nn.num_blocks == 7

    def test_env_overrides_storage_uri(self, monkeypatch):
        monkeypatch.setenv("ALPHACHESS_STORAGE__ROOT_URI", "memory://test")
        cfg = Config.from_env()
        assert cfg.storage.root_uri == "memory://test"

    def test_env_overrides_orchestrator_role(self, monkeypatch):
        monkeypatch.setenv("ALPHACHESS_ORCHESTRATOR__ROLE", "train")
        cfg = Config.from_env()
        assert cfg.orchestrator.role == "train"


class TestHash:
    def test_same_config_same_hash(self):
        cfg1 = Config.from_yaml(LAPTOP_YAML)
        cfg2 = Config.from_yaml(LAPTOP_YAML)
        assert cfg1.hash() == cfg2.hash()

    def test_different_config_different_hash(self, monkeypatch):
        monkeypatch.setenv("ALPHACHESS_NN__NUM_BLOCKS", "99")
        cfg_env = Config.from_env()
        cfg_yaml = Config.from_yaml(LAPTOP_YAML)
        assert cfg_env.hash() != cfg_yaml.hash()

    def test_hash_is_hex_string(self):
        cfg = Config.from_yaml(LAPTOP_YAML)
        h = cfg.hash()
        assert isinstance(h, str)
        assert len(h) == 32
        int(h, 16)  # raises if not valid hex