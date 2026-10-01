from __future__ import annotations

from dataclasses import dataclass, fields
from math import isqrt
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

from deep_reinforcement_learning_lab.algorithms.names import validate_algorithm

LossType = Literal["mse", "huber"]

_DEFAULT_EVAL_SEED_OFFSET = 10_000
_DEFAULT_VIDEO_SEED_OFFSET = 20_000

_RUN_OPTIONAL_KEYS = frozenset(
    {
        "save_checkpoint",
        "video_interval_steps",
        "video_enabled",
        "video_episodes",
        "eval_seed_offset",
        "video_seed_offset",
    }
)

_DQN_HPARAM_OPTIONAL_KEYS = frozenset({"loss_type", "hidden_sizes"})


@dataclass(frozen=True)
class RunConfig:
    """アルゴリズムに依存しない実験設定。"""

    name: str
    seed: int
    max_env_steps: int
    eval_episodes: int
    log_interval_steps: int
    save_checkpoint: bool
    video_enabled: bool
    video_interval_steps: int
    video_episodes: int
    eval_seed_offset: int
    video_seed_offset: int

    @property
    def video_grid_side(self) -> int:
        return isqrt(self.video_episodes)

    def eval_rollout_seed(self, episode: int) -> int:
        return self.seed + self.eval_seed_offset + episode

    def video_rollout_seed(self, episode: int) -> int:
        return self.seed + self.video_seed_offset + episode


@dataclass(frozen=True)
class DQNHyperparameters:
    """DQN 固有のハイパーパラメータ。"""

    gamma: float
    learning_rate: float
    buffer_size: int
    batch_size: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    target_update_interval: int
    loss_type: LossType
    hidden_sizes: tuple[int, ...]

    @property
    def epsilon_rate(self) -> float:
        return self.epsilon_start / self.epsilon_decay_steps


@dataclass(frozen=True)
class ExperimentConfig:
    subtask: str
    algorithm: str
    run: RunConfig
    hyperparameters: DQNHyperparameters


def resolve_run_dir(task_dir: Path, experiment: ExperimentConfig) -> Path:
    """学習成果物の出力先: ``runs/<subtask>/<algorithm>/<run.name>/``。"""
    return (
        task_dir
        / "runs"
        / experiment.subtask
        / experiment.algorithm
        / experiment.run.name
    )


def ensure_run_dir_is_new(run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError(
            f"run_dir already exists: {run_dir}. "
            "Use a different run.name or remove the existing directory."
        )


def load_experiment_config(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")

    allowed_top = {"subtask", "algorithm", "run", "hyperparameters"}
    unknown_top = set(raw) - allowed_top
    if unknown_top:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown_top)}")

    subtask = raw.get("subtask")
    algorithm = raw.get("algorithm")
    if not isinstance(subtask, str) or not subtask.strip():
        raise ValueError(f"subtask must be a non-empty string: {path}")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValueError(f"algorithm must be a non-empty string: {path}")

    validate_algorithm(algorithm)
    run_block = raw.get("run")
    hyper_block = raw.get("hyperparameters")
    if not isinstance(run_block, dict):
        raise ValueError(f"run must be a mapping: {path}")
    if not isinstance(hyper_block, dict):
        raise ValueError(f"hyperparameters must be a mapping: {path}")

    run = _parse_run_config(run_block, path)
    if algorithm == "dqn":
        hyperparameters = _parse_dqn_hyperparameters(hyper_block, path)
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")

    return ExperimentConfig(
        subtask=subtask.strip(),
        algorithm=algorithm.strip(),
        run=run,
        hyperparameters=hyperparameters,
    )


def _parse_run_config(raw: dict[str, Any], path: Path) -> RunConfig:
    allowed = {field.name for field in fields(RunConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown run keys in {path}: {sorted(unknown)}")

    missing = sorted(
        name for name in allowed if name not in raw and name not in _RUN_OPTIONAL_KEYS
    )
    if missing:
        raise ValueError(f"missing required run keys in {path}: {missing}")

    save_checkpoint = raw.get("save_checkpoint", True)
    if not isinstance(save_checkpoint, bool):
        raise ValueError("save_checkpoint must be a boolean")

    max_env_steps = int(raw["max_env_steps"])
    if max_env_steps <= 0:
        raise ValueError("max_env_steps must be a positive integer")

    log_interval_steps = int(raw["log_interval_steps"])
    if log_interval_steps <= 0:
        raise ValueError("log_interval_steps must be a positive integer")

    video_enabled = raw.get("video_enabled", False)
    if not isinstance(video_enabled, bool):
        raise ValueError("video_enabled must be a boolean")

    video_interval_steps = int(raw.get("video_interval_steps", 0))
    if video_interval_steps < 0:
        raise ValueError("video_interval_steps must be >= 0")
    if video_enabled and video_interval_steps <= 0:
        raise ValueError("video_interval_steps must be positive when video_enabled is true")
    if video_interval_steps > 0 and video_interval_steps % log_interval_steps != 0:
        raise ValueError(
            f"video_interval_steps ({video_interval_steps}) must be a multiple of "
            f"log_interval_steps ({log_interval_steps})"
        )

    video_episodes = int(raw.get("video_episodes", 16))
    if video_episodes <= 0:
        raise ValueError("video_episodes must be a positive integer")
    grid_side = isqrt(video_episodes)
    if grid_side * grid_side != video_episodes:
        raise ValueError(
            f"video_episodes ({video_episodes}) must be a perfect square (e.g. 16 for 4x4)"
        )

    eval_episodes = int(raw["eval_episodes"])
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be a positive integer")

    eval_seed_offset = int(raw.get("eval_seed_offset", _DEFAULT_EVAL_SEED_OFFSET))
    video_seed_offset = int(raw.get("video_seed_offset", _DEFAULT_VIDEO_SEED_OFFSET))
    if eval_seed_offset == video_seed_offset:
        raise ValueError("eval_seed_offset and video_seed_offset must differ")

    seed = int(raw["seed"])
    eval_seeds = {seed + eval_seed_offset + episode for episode in range(eval_episodes)}
    video_seeds = {seed + video_seed_offset + episode for episode in range(video_episodes)}
    if eval_seeds & video_seeds:
        raise ValueError(
            "eval and video rollout seeds must not overlap; "
            "adjust offsets or episode counts"
        )

    return RunConfig(
        name=str(raw["name"]),
        seed=seed,
        max_env_steps=max_env_steps,
        eval_episodes=eval_episodes,
        log_interval_steps=log_interval_steps,
        save_checkpoint=save_checkpoint,
        video_enabled=video_enabled,
        video_interval_steps=video_interval_steps,
        video_episodes=video_episodes,
        eval_seed_offset=eval_seed_offset,
        video_seed_offset=video_seed_offset,
    )


def _parse_dqn_hyperparameters(raw: dict[str, Any], path: Path) -> DQNHyperparameters:
    allowed = {field.name for field in fields(DQNHyperparameters)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown hyperparameters keys in {path}: {sorted(unknown)}")

    missing = sorted(
        name for name in allowed if name not in raw and name not in _DQN_HPARAM_OPTIONAL_KEYS
    )
    if missing:
        raise ValueError(f"missing required hyperparameters keys in {path}: {missing}")

    loss_type = raw.get("loss_type", "huber")
    if loss_type not in get_args(LossType):
        raise ValueError(f"unsupported loss_type: {loss_type}")

    hidden_sizes = raw.get("hidden_sizes", [128, 128])
    if not isinstance(hidden_sizes, list) or len(hidden_sizes) == 0:
        raise ValueError("hidden_sizes must be a non-empty list of positive integers")
    parsed_sizes = tuple(int(size) for size in hidden_sizes)
    if any(size <= 0 for size in parsed_sizes):
        raise ValueError("hidden_sizes must contain positive integers")

    epsilon_decay_steps = int(raw["epsilon_decay_steps"])
    if epsilon_decay_steps <= 0:
        raise ValueError("epsilon_decay_steps must be a positive integer")

    return DQNHyperparameters(
        gamma=float(raw["gamma"]),
        learning_rate=float(raw["learning_rate"]),
        buffer_size=int(raw["buffer_size"]),
        batch_size=int(raw["batch_size"]),
        epsilon_start=float(raw["epsilon_start"]),
        epsilon_end=float(raw["epsilon_end"]),
        epsilon_decay_steps=epsilon_decay_steps,
        target_update_interval=int(raw["target_update_interval"]),
        loss_type=loss_type,
        hidden_sizes=parsed_sizes,
    )
