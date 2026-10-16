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


def load_lunar_experiment(path: Path) -> LunarExperiment:
    with path.open(encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")

    model_block = raw.pop("model", None)
    model = parse_model_config(model_block, path)
    experiment = load_experiment_config_from_raw(raw, path)
    return LunarExperiment(experiment=experiment, model=model)


def load_model_config_from_path(config_path: Path) -> ModelConfig:
    return load_lunar_experiment(config_path).model
