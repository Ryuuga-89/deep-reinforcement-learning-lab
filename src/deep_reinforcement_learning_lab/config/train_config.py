from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

LossType = Literal["mse", "huber"]
RewardMode = Literal["default", "sutton_barto"]

_CONFIG_OPTIONAL_KEYS = frozenset(
    {
        "loss_type",
        "hidden_sizes",
        "reward_mode",
        "save_checkpoint",
        "video_interval_steps",
    }
)


@dataclass(frozen=True)
class TrainConfig:
    """DQN 学習用の設定（YAML から読み込む）。"""

    run_name: str
    env_id: str
    seed: int
    max_env_steps: int
    gamma: float
    learning_rate: float
    buffer_size: int
    batch_size: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    target_update_interval: int
    eval_episodes: int
    log_interval_steps: int
    loss_type: LossType
    hidden_sizes: tuple[int, ...]
    reward_mode: RewardMode
    save_checkpoint: bool
    video_interval_steps: int

    @property
    def epsilon_rate(self) -> float:
        """環境ステップあたりの ε 減衰量（``epsilon_start / epsilon_decay_steps``）。"""
        return self.epsilon_start / self.epsilon_decay_steps


def load_train_config(path: Path) -> TrainConfig:
    with path.open(encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")

    allowed = {field.name for field in fields(TrainConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")

    missing = sorted(name for name in allowed if name not in raw and name not in _CONFIG_OPTIONAL_KEYS)
    if missing:
        raise ValueError(f"missing required config keys in {path}: {missing}")

    loss_type = raw.get("loss_type", "huber")
    if loss_type not in get_args(LossType):
        raise ValueError(f"unsupported loss_type: {loss_type}")

    hidden_sizes = raw.get("hidden_sizes", [128, 128])
    if not isinstance(hidden_sizes, list) or len(hidden_sizes) == 0:
        raise ValueError("hidden_sizes must be a non-empty list of positive integers")
    parsed_sizes = tuple(int(size) for size in hidden_sizes)
    if any(size <= 0 for size in parsed_sizes):
        raise ValueError("hidden_sizes must contain positive integers")

    reward_mode = raw.get("reward_mode", "default")
    if reward_mode not in get_args(RewardMode):
        raise ValueError(f"unsupported reward_mode: {reward_mode}")

    save_checkpoint = raw.get("save_checkpoint", True)
    if not isinstance(save_checkpoint, bool):
        raise ValueError("save_checkpoint must be a boolean")

    max_env_steps = int(raw["max_env_steps"])
    if max_env_steps <= 0:
        raise ValueError("max_env_steps must be a positive integer")

    log_interval_steps = int(raw["log_interval_steps"])
    if log_interval_steps <= 0:
        raise ValueError("log_interval_steps must be a positive integer")

    video_interval_steps = int(raw.get("video_interval_steps", 0))
    if video_interval_steps < 0:
        raise ValueError("video_interval_steps must be >= 0 (0 disables recording)")
    if video_interval_steps > 0 and video_interval_steps % log_interval_steps != 0:
        raise ValueError(
            f"video_interval_steps ({video_interval_steps}) must be a multiple of "
            f"log_interval_steps ({log_interval_steps})"
        )

    data = {
        **raw,
        "loss_type": loss_type,
        "hidden_sizes": parsed_sizes,
        "reward_mode": reward_mode,
        "save_checkpoint": save_checkpoint,
        "video_interval_steps": video_interval_steps,
    }
    return TrainConfig(**{key: data[key] for key in allowed})
