"""Configuration loading.

The config file is looked up in this order:

1. the ``BASES_CONFIG`` environment variable,
2. ``bases.yml`` in the current working directory,
3. ``bases.yml`` next to the package directory (the repository layout).

Relative paths inside the file are resolved against the file's directory, so
the package works from any working directory.  Missing keys fall back to
:data:`DEFAULTS`.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILENAME = "bases.yml"

DEFAULTS: dict[str, Any] = {
    "paths": {
        "policy": "data/policy/policy_{size}x{size}.json.gz",
        "training_log": "data/logs/training_{size}x{size}.csv",
        "lines_log": "data/logs/lines.csv",
        "game_log": "data/logs/game.log",
        "tuning_results": "data/hp/hyperparameter_results_{size}x{size}.csv",
    },
    "rules": {
        "extra_turn_on_box": False,
    },
    # GitHub repository whose releases carry the trained policies
    "release_repo": "degen00/bases",
    "training": {
        "default": {
            "learning_rate": 0.3,
            "discount_factor": 0.98,
            "exploration_rate": 0.3,
            "exploration_min": 0.02,
            "opponent_mix": {"self": 0.8, "greedy": 0.1, "random": 0.1},
        },
    },
}

HYPERPARAMETER_KEYS = ("learning_rate", "discount_factor",
                       "exploration_rate", "exploration_min", "opponent_mix")


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def find_config_file() -> Path | None:
    env = os.environ.get("BASES_CONFIG")
    candidates = [Path(env)] if env else []
    candidates.append(Path.cwd() / CONFIG_FILENAME)
    candidates.append(Path(__file__).resolve().parent.parent / CONFIG_FILENAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


class Config:
    """Merged configuration with helpers for paths and hyperparameters."""

    def __init__(self, data: dict[str, Any], base_dir: Path, source: Path | None):
        self.data = data
        self.base_dir = base_dir
        self.source = source

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def extra_turn_on_box(self) -> bool:
        return bool(self.data["rules"].get("extra_turn_on_box", False))

    def path(self, key: str, size: int | None = None) -> Path:
        template = self.data["paths"][key]
        if size is not None:
            template = template.format(size=size)
        p = Path(template)
        return p if p.is_absolute() else self.base_dir / p

    def hyperparameters(self, size: int) -> dict[str, Any]:
        training = self.data.get("training", {})
        params = copy.deepcopy(DEFAULTS["training"]["default"])
        params.update(training.get("default", {}))
        specific = training.get(size, training.get(str(size), {}))
        params.update(specific or {})
        return {k: params[k] for k in HYPERPARAMETER_KEYS if k in params}


def load_config(path: str | os.PathLike | None = None) -> Config:
    file = Path(path) if path else find_config_file()
    data: dict[str, Any] = {}
    if file is not None and file.is_file():
        with open(file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        base_dir = file.resolve().parent
    else:
        file = None
        base_dir = Path(__file__).resolve().parent.parent
    return Config(_merge(DEFAULTS, data), base_dir, file)
