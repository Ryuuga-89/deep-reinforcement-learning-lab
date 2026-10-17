from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from deep_reinforcement_learning_lab.config import (
    ExperimentConfig,
    load_experiment_config_from_raw,
)

from model_config import ModelConfig, parse_model_config


@dataclass(frozen=True)
class LunarExperiment:
    experiment: ExperimentConfig
    model: ModelConfig
    seeds: tuple[int, ...] | None = None


def _parse_optional_seeds(raw: dict[str, Any], path: Path) -> tuple[int, ...] | None:
    if "seeds" not in raw:
        return None
    seeds_raw = raw.pop("seeds")
    if seeds_raw is None:
        return None
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError(f"seeds must be a non-empty list of integers: {path}")
    seeds: list[int] = []
    for item in seeds_raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"seeds entries must be integers: {path}")
        seeds.append(item)
    return tuple(seeds)


def load_lunar_experiment(path: Path) -> LunarExperiment:
    with path.open(encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")

    seeds = _parse_optional_seeds(raw, path)
    model_block = raw.pop("model", None)
    model = parse_model_config(model_block, path)
    experiment = load_experiment_config_from_raw(raw, path)
    return LunarExperiment(experiment=experiment, model=model, seeds=seeds)


def load_model_config_from_path(config_path: Path) -> ModelConfig:
    return load_lunar_experiment(config_path).model
