from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.config import ExperimentConfig, PPOHyperparameters, RunConfig
from ac_model_registry import (
    PpoKind,
    build_registered_model,
    resolve_legacy_model_id,
)
from model_config import ModelConfig
import models as _models  # noqa: F401 — register default builders

from policy import EvalMetrics

CHECKPOINT_FILENAME = "ac.pt"
BEST_CHECKPOINT_FILENAME = "ac_best.pt"


def checkpoint_path(log_dir: Path) -> Path:
    return log_dir / CHECKPOINT_FILENAME


def best_checkpoint_path(log_dir: Path) -> Path:
    return log_dir / BEST_CHECKPOINT_FILENAME


def save_ac_checkpoint(
    path: Path,
    ac_model: nn.Module,
    experiment: ExperimentConfig,
    state_dim: int,
    rollout_action_dim: int,
    ppo_kind: PpoKind,
    model: ModelConfig,
    *,
    action_low: torch.Tensor | None = None,
    action_high: torch.Tensor | None = None,
    env_step: int | None = None,
    eval_mean_episode_return: float | None = None,
    eval_success_rate: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    hparams = experiment.hyperparameters
    if not isinstance(hparams, PPOHyperparameters):
        raise TypeError("save_ac_checkpoint requires PPO hyperparameters")

    payload: dict[str, Any] = {
        "state_dict": ac_model.state_dict(),
        "state_dim": state_dim,
        "action_dim": rollout_action_dim,
        "hidden_sizes": list(hparams.hidden_sizes),
        "ac_type": hparams.ac_type,
        "policy_kind": ppo_kind,
        "model_id": model.id,
        "model_options": dict(model.options),
        "subtask": experiment.subtask,
        "algorithm": experiment.algorithm,
        "run_name": experiment.run.name,
    }
    if ppo_kind == "continuous":
        if action_low is None or action_high is None:
            raise ValueError("continuous checkpoint requires action bounds")
        payload["action_low"] = action_low.tolist()
        payload["action_high"] = action_high.tolist()
    if env_step is not None:
        payload["env_step"] = env_step
    if eval_mean_episode_return is not None:
        payload["eval_mean_episode_return"] = eval_mean_episode_return
    if eval_success_rate is not None:
        payload["eval_success_rate"] = eval_success_rate
    torch.save(payload, path)
    return path


def _experiment_from_checkpoint_payload(payload: dict[str, Any]) -> ExperimentConfig:
    subtask = str(payload.get("subtask", "default"))
    run = RunConfig(
        name=str(payload.get("run_name", "checkpoint")),
        seed=0,
        max_env_steps=1,
        eval_episodes=1,
        eval_seed_offset=0,
        video_seed_offset=0,
        log_interval_steps=1,
        video_enabled=False,
        video_interval_steps=1,
        video_episodes=1,
        save_checkpoint=False,
    )
    hparams = PPOHyperparameters(
        gamma=0.99,
        learning_rate=0.0003,
        gae_lambda=0.95,
        clip_coef=0.2,
        entropy_coef=0.0,
        value_coef=0.5,
        rollout_steps=64,
        n_epochs=1,
        minibatch_size=64,
        hidden_sizes=tuple(int(s) for s in payload["hidden_sizes"]),
        ac_type=payload.get("ac_type", "shared"),
    )
    return ExperimentConfig(
        subtask=subtask,
        algorithm=str(payload.get("algorithm", "ppo")),
        run=run,
        hyperparameters=hparams,
    )


def load_ac_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> nn.Module:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"invalid checkpoint format: {path}")

    state_dim = int(payload["state_dim"])
    rollout_action_dim = int(payload["action_dim"])
    hidden_sizes = tuple(int(size) for size in payload["hidden_sizes"])
    ac_type = payload.get("ac_type", "shared")
    policy_kind = str(payload.get("policy_kind", "continuous"))
    raw_model_id = str(payload.get("model_id", "lunar_continuous"))
    model_id = resolve_legacy_model_id(raw_model_id, policy_kind)
    options_raw = payload.get("model_options", {})
    if options_raw is None:
        options_raw = {}
    if not isinstance(options_raw, dict):
        raise ValueError(f"invalid model_options in checkpoint: {path}")
    model = ModelConfig(id=model_id, options=dict(options_raw))

    action_low = None
    action_high = None
    if policy_kind == "continuous":
        action_low = torch.tensor(payload["action_low"], dtype=torch.float32)
        action_high = torch.tensor(payload["action_high"], dtype=torch.float32)

    ac_model = build_registered_model(
        model,
        _experiment_from_checkpoint_payload(payload),
        state_dim,
        rollout_action_dim,
        ac_type,
        hidden_sizes,
        action_low=action_low,
        action_high=action_high,
    )
    state_dict = payload["state_dict"]
    if (
        policy_kind == "discrete"
        and model.id == "lunar_table"
        and not any(key.startswith("inner.") for key in state_dict)
    ):
        state_dict = {f"inner.{key}": value for key, value in state_dict.items()}
    ac_model.load_state_dict(state_dict)
    ac_model.eval()
    return ac_model


@dataclass
class BestCheckpointTracker:
    best_success_rate: float = float("-inf")
    best_mean_return: float = float("-inf")
    best_env_step: int | None = None
    path: Path | None = None

    def update_if_better(
        self,
        metrics: EvalMetrics,
        env_step: int,
        *,
        experiment: ExperimentConfig,
        log_dir: Path,
        ac_model: nn.Module,
        state_dim: int,
        rollout_action_dim: int,
        ppo_kind: PpoKind,
        model: ModelConfig,
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
    ) -> bool:
        candidate = metrics.primary_score
        current = (self.best_success_rate, self.best_mean_return)
        if candidate <= current:
            return False
        self.best_success_rate = metrics.success_rate
        self.best_mean_return = metrics.mean_episode_return
        self.best_env_step = env_step
        self.path = save_ac_checkpoint(
            best_checkpoint_path(log_dir),
            ac_model,
            experiment,
            state_dim,
            rollout_action_dim,
            ppo_kind,
            model,
            action_low=action_low,
            action_high=action_high,
            env_step=env_step,
            eval_mean_episode_return=metrics.mean_episode_return,
            eval_success_rate=metrics.success_rate,
        )
        return True
