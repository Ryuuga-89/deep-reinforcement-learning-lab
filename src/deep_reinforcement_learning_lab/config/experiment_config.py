from __future__ import annotations

from dataclasses import dataclass, fields
from math import isqrt
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

from deep_reinforcement_learning_lab.algorithms.names import validate_algorithm

LossType = Literal["mse", "huber"]
QNetType = Literal["mlp", "dueling"]
AcType = Literal["shared", "separate"]

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

_DQN_HPARAM_OPTIONAL_KEYS = frozenset(
    {"loss_type", "hidden_sizes", "tau", "target_update_interval", "qnet_type"}
)

_PER_HPARAM_OPTIONAL_KEYS = frozenset({"priority_epsilon"})

_PPO_HPARAM_OPTIONAL_KEYS = frozenset(
    {"hidden_sizes", "ac_type", "clip_value_coef", "max_grad_norm", "n_envs"}
)


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
    loss_type: LossType
    hidden_sizes: tuple[int, ...]
    target_update_interval: int | None = None
    tau: float | None = None
    qnet_type: QNetType = "mlp"

    @property
    def epsilon_rate(self) -> float:
        return self.epsilon_start / self.epsilon_decay_steps


@dataclass(frozen=True)
class PERHyperparameters(DQNHyperparameters):
    """DQN hyperparameters plus prioritized replay settings."""

    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    priority_epsilon: float = 1e-6


@dataclass(frozen=True)
class PPOHyperparameters:
    """PPO 固有のハイパーパラメータ。"""

    gamma: float
    learning_rate: float
    gae_lambda: float
    clip_coef: float
    entropy_coef: float
    value_coef: float
    rollout_steps: int
    n_epochs: int
    minibatch_size: int
    hidden_sizes: tuple[int, ...]
    ac_type: AcType = "shared"
    clip_value_coef: float | None = None
    max_grad_norm: float = 0.5
    n_envs: int = 1

    @property
    def rollout_transitions(self) -> int:
        return self.n_envs * self.rollout_steps


@dataclass(frozen=True)
class SubtaskOptions:
    """サブタスク固有の環境オプション（タスク側の ``make_subtask_env`` が解釈する）。"""

    center_bonus_coef: float | None = None
    move_alpha: float | None = None
    move_beta: int | None = None
    noise_rho: float | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    subtask: str
    algorithm: str
    run: RunConfig
    hyperparameters: DQNHyperparameters | PERHyperparameters | PPOHyperparameters
    subtask_options: SubtaskOptions = SubtaskOptions()


ExperimentFlatFields = dict[str, str | int | float | bool]


def experiment_flat_fields(experiment: ExperimentConfig) -> ExperimentFlatFields:
    """実験設定の平坦表現（TensorBoard TEXT / HPARAMS 用。値はスカラーのみ）。"""
    run = experiment.run
    hparams = experiment.hyperparameters
    base: ExperimentFlatFields = {
        "subtask": experiment.subtask,
        "algorithm": experiment.algorithm,
        "run/name": run.name,
        "run/seed": run.seed,
        "run/max_env_steps": run.max_env_steps,
        "run/eval_episodes": run.eval_episodes,
        "run/log_interval_steps": run.log_interval_steps,
        "run/save_checkpoint": run.save_checkpoint,
        "run/video_enabled": run.video_enabled,
        "run/video_interval_steps": run.video_interval_steps,
        "run/video_episodes": run.video_episodes,
        "run/eval_seed_offset": run.eval_seed_offset,
        "run/video_seed_offset": run.video_seed_offset,
        **(
            {
                "subtask_options/center_bonus_coef": experiment.subtask_options.center_bonus_coef,
            }
            if experiment.subtask_options.center_bonus_coef is not None
            else {}
        ),
        **(
            {
                "subtask_options/move_alpha": experiment.subtask_options.move_alpha,
                "subtask_options/move_beta": experiment.subtask_options.move_beta,
            }
            if experiment.subtask_options.move_alpha is not None
            and experiment.subtask_options.move_beta is not None
            else {}
        ),
        **(
            {
                "subtask_options/noise_rho": experiment.subtask_options.noise_rho,
            }
            if experiment.subtask_options.noise_rho is not None
            else {}
        ),
    }
    if isinstance(hparams, PPOHyperparameters):
        return {
            **base,
            "hyperparameters/gamma": hparams.gamma,
            "hyperparameters/learning_rate": hparams.learning_rate,
            "hyperparameters/gae_lambda": hparams.gae_lambda,
            "hyperparameters/clip_coef": hparams.clip_coef,
            "hyperparameters/entropy_coef": hparams.entropy_coef,
            "hyperparameters/value_coef": hparams.value_coef,
            "hyperparameters/rollout_steps": hparams.rollout_steps,
            "hyperparameters/n_envs": hparams.n_envs,
            "hyperparameters/n_epochs": hparams.n_epochs,
            "hyperparameters/minibatch_size": hparams.minibatch_size,
            "hyperparameters/max_grad_norm": hparams.max_grad_norm,
            "hyperparameters/hidden_sizes": ",".join(
                str(size) for size in hparams.hidden_sizes
            ),
            "hyperparameters/ac_type": hparams.ac_type,
            **(
                {"hyperparameters/clip_value_coef": hparams.clip_value_coef}
                if hparams.clip_value_coef is not None
                else {}
            ),
        }

    return {
        **base,
        "hyperparameters/gamma": hparams.gamma,
        "hyperparameters/learning_rate": hparams.learning_rate,
        "hyperparameters/buffer_size": hparams.buffer_size,
        "hyperparameters/batch_size": hparams.batch_size,
        "hyperparameters/epsilon_start": hparams.epsilon_start,
        "hyperparameters/epsilon_end": hparams.epsilon_end,
        "hyperparameters/epsilon_decay_steps": hparams.epsilon_decay_steps,
        **(
            {"hyperparameters/target_update_interval": hparams.target_update_interval}
            if hparams.target_update_interval is not None
            else {}
        ),
        **(
            {"hyperparameters/tau": hparams.tau}
            if hparams.tau is not None
            else {}
        ),
        "hyperparameters/loss_type": hparams.loss_type,
        "hyperparameters/hidden_sizes": ",".join(
            str(size) for size in hparams.hidden_sizes
        ),
        "hyperparameters/qnet_type": hparams.qnet_type,
        **(
            {
                "hyperparameters/per_alpha": hparams.per_alpha,
                "hyperparameters/per_beta_start": hparams.per_beta_start,
                "hyperparameters/per_beta_end": hparams.per_beta_end,
                "hyperparameters/priority_epsilon": hparams.priority_epsilon,
            }
            if isinstance(hparams, PERHyperparameters)
            else {}
        ),
    }


def format_experiment_flat_fields(fields: ExperimentFlatFields) -> str:
    return "\n".join(f"{key}: {value}" for key, value in fields.items())


def archive_experiment_yaml(run_dir: Path, config_path: Path) -> Path:
    """使用した YAML を ``run_dir/experiment.yaml`` に保存する。"""
    destination = run_dir / "experiment.yaml"
    destination.write_text(config_path.resolve().read_text(encoding="utf-8"), encoding="utf-8")
    return destination


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

    allowed_top = {"subtask", "algorithm", "run", "hyperparameters", "subtask_options"}
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
    subtask_options = _parse_subtask_options(
        subtask.strip(),
        raw.get("subtask_options"),
        path,
    )
    if algorithm in ("dqn", "ddqn"):
        hyperparameters = _parse_dqn_hyperparameters(hyper_block, path)
    elif algorithm in ("per_dqn", "per_ddqn"):
        hyperparameters = _parse_per_hyperparameters(hyper_block, path)
    elif algorithm == "ppo":
        hyperparameters = _parse_ppo_hyperparameters(hyper_block, path)
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")

    return ExperimentConfig(
        subtask=subtask.strip(),
        algorithm=algorithm.strip(),
        run=run,
        hyperparameters=hyperparameters,
        subtask_options=subtask_options,
    )


def _parse_subtask_options(
    subtask: str,
    block: Any,
    path: Path,
) -> SubtaskOptions:
    if block is None:
        block = {}
    if not isinstance(block, dict):
        raise ValueError(f"subtask_options must be a mapping: {path}")

    if subtask == "center":
        unknown = set(block) - {"center_bonus_coef"}
        if unknown:
            raise ValueError(f"unknown subtask_options keys in {path}: {sorted(unknown)}")
        center_bonus_coef = float(block.get("center_bonus_coef", 1.0))
        if center_bonus_coef < 0.0:
            raise ValueError(f"center_bonus_coef must be non-negative: {path}")
        return SubtaskOptions(center_bonus_coef=center_bonus_coef)

    if subtask == "move":
        unknown = set(block) - {"move_alpha", "move_beta"}
        if unknown:
            raise ValueError(f"unknown subtask_options keys in {path}: {sorted(unknown)}")
        move_alpha = float(block.get("move_alpha", 1.0))
        if move_alpha < 0.0:
            raise ValueError(f"move_alpha must be non-negative: {path}")
        raw_beta = block.get("move_beta", 100)
        if isinstance(raw_beta, bool) or not isinstance(raw_beta, int) or not 1 <= raw_beta <= 500:
            raise ValueError(f"move_beta must be an integer in 1..500: {path}")
        return SubtaskOptions(move_alpha=move_alpha, move_beta=raw_beta)

    if subtask == "noise":
        unknown = set(block) - {"noise_rho"}
        if unknown:
            raise ValueError(f"unknown subtask_options keys in {path}: {sorted(unknown)}")
        noise_rho = float(block.get("noise_rho", 0.1))
        if noise_rho < 0.0:
            raise ValueError(f"noise_rho must be non-negative: {path}")
        return SubtaskOptions(noise_rho=noise_rho)

    if block:
        raise ValueError(
            f"subtask_options is only supported for subtasks 'center', 'move', and 'noise': {path}"
        )
    return SubtaskOptions()


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

    has_tau = "tau" in raw
    has_interval = "target_update_interval" in raw
    if has_tau and has_interval:
        raise ValueError(
            f"in {path}: use either tau (soft target update) or "
            "target_update_interval (hard), not both"
        )
    if not has_tau and not has_interval:
        raise ValueError(
            f"in {path}: tau or target_update_interval is required"
        )

    tau: float | None = None
    target_update_interval: int | None = None
    if has_tau:
        tau = float(raw["tau"])
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1] when set")
    else:
        target_update_interval = int(raw["target_update_interval"])
        if target_update_interval <= 0:
            raise ValueError("target_update_interval must be a positive integer")

    qnet_type = raw.get("qnet_type", "mlp")
    if qnet_type not in get_args(QNetType):
        raise ValueError(f"unsupported qnet_type: {qnet_type}")

    return DQNHyperparameters(
        gamma=float(raw["gamma"]),
        learning_rate=float(raw["learning_rate"]),
        buffer_size=int(raw["buffer_size"]),
        batch_size=int(raw["batch_size"]),
        epsilon_start=float(raw["epsilon_start"]),
        epsilon_end=float(raw["epsilon_end"]),
        epsilon_decay_steps=epsilon_decay_steps,
        loss_type=loss_type,
        hidden_sizes=parsed_sizes,
        target_update_interval=target_update_interval,
        tau=tau,
        qnet_type=qnet_type,
    )


def _parse_per_hyperparameters(raw: dict[str, Any], path: Path) -> PERHyperparameters:
    allowed = {field.name for field in fields(PERHyperparameters)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown hyperparameters keys in {path}: {sorted(unknown)}")

    optional = _DQN_HPARAM_OPTIONAL_KEYS | _PER_HPARAM_OPTIONAL_KEYS
    missing = sorted(
        name for name in allowed if name not in raw and name not in optional
    )
    if missing:
        raise ValueError(f"missing required hyperparameters keys in {path}: {missing}")

    dqn_field_names = {field.name for field in fields(DQNHyperparameters)}
    dqn_raw = {key: value for key, value in raw.items() if key in dqn_field_names}
    dqn = _parse_dqn_hyperparameters(dqn_raw, path)

    per_alpha = float(raw["per_alpha"])
    if not 0.0 <= per_alpha <= 1.0:
        raise ValueError("per_alpha must be in [0, 1]")

    per_beta_start = float(raw["per_beta_start"])
    per_beta_end = float(raw["per_beta_end"])
    if not 0.0 <= per_beta_start <= 1.0:
        raise ValueError("per_beta_start must be in [0, 1]")
    if not 0.0 <= per_beta_end <= 1.0:
        raise ValueError("per_beta_end must be in [0, 1]")

    priority_epsilon = float(raw.get("priority_epsilon", 1e-6))
    if priority_epsilon < 0.0:
        raise ValueError("priority_epsilon must be non-negative")

    return PERHyperparameters(
        gamma=dqn.gamma,
        learning_rate=dqn.learning_rate,
        buffer_size=dqn.buffer_size,
        batch_size=dqn.batch_size,
        epsilon_start=dqn.epsilon_start,
        epsilon_end=dqn.epsilon_end,
        epsilon_decay_steps=dqn.epsilon_decay_steps,
        loss_type=dqn.loss_type,
        hidden_sizes=dqn.hidden_sizes,
        target_update_interval=dqn.target_update_interval,
        tau=dqn.tau,
        qnet_type=dqn.qnet_type,
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        per_beta_end=per_beta_end,
        priority_epsilon=priority_epsilon,
    )


def _parse_hidden_sizes(raw: dict[str, Any], path: Path) -> tuple[int, ...]:
    hidden_sizes = raw.get("hidden_sizes", [128, 128])
    if not isinstance(hidden_sizes, list) or len(hidden_sizes) == 0:
        raise ValueError("hidden_sizes must be a non-empty list of positive integers")
    parsed_sizes = tuple(int(size) for size in hidden_sizes)
    if any(size <= 0 for size in parsed_sizes):
        raise ValueError("hidden_sizes must contain positive integers")
    return parsed_sizes


def _parse_ppo_hyperparameters(raw: dict[str, Any], path: Path) -> PPOHyperparameters:
    allowed = {field.name for field in fields(PPOHyperparameters)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown hyperparameters keys in {path}: {sorted(unknown)}")

    missing = sorted(
        name for name in allowed if name not in raw and name not in _PPO_HPARAM_OPTIONAL_KEYS
    )
    if missing:
        raise ValueError(f"missing required hyperparameters keys in {path}: {missing}")

    ac_type = raw.get("ac_type", "shared")
    if ac_type not in get_args(AcType):
        raise ValueError(f"unsupported ac_type: {ac_type}")

    clip_value_coef: float | None = None
    if "clip_value_coef" in raw:
        clip_value_coef = float(raw["clip_value_coef"])
        if clip_value_coef <= 0.0:
            raise ValueError("clip_value_coef must be positive when set")

    rollout_steps = int(raw["rollout_steps"])
    n_epochs = int(raw["n_epochs"])
    minibatch_size = int(raw["minibatch_size"])
    if rollout_steps <= 0 or n_epochs <= 0 or minibatch_size <= 0:
        raise ValueError("rollout_steps, n_epochs, and minibatch_size must be positive")

    max_grad_norm = float(raw.get("max_grad_norm", 0.5))
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")

    gae_lambda = float(raw["gae_lambda"])
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")

    clip_coef = float(raw["clip_coef"])
    if clip_coef <= 0.0:
        raise ValueError("clip_coef must be positive")

    n_envs = int(raw.get("n_envs", 1))
    if n_envs <= 0:
        raise ValueError("n_envs must be a positive integer")

    rollout_transitions = n_envs * rollout_steps
    if rollout_transitions < minibatch_size:
        raise ValueError(
            f"n_envs * rollout_steps ({rollout_transitions}) must be >= "
            f"minibatch_size ({minibatch_size})"
        )
    if minibatch_size < 2:
        raise ValueError("minibatch_size must be >= 2 for PPO advantage normalization")

    return PPOHyperparameters(
        gamma=float(raw["gamma"]),
        learning_rate=float(raw["learning_rate"]),
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        entropy_coef=float(raw["entropy_coef"]),
        value_coef=float(raw["value_coef"]),
        rollout_steps=rollout_steps,
        n_epochs=n_epochs,
        minibatch_size=minibatch_size,
        hidden_sizes=_parse_hidden_sizes(raw, path),
        ac_type=ac_type,
        clip_value_coef=clip_value_coef,
        max_grad_norm=max_grad_norm,
        n_envs=n_envs,
    )
