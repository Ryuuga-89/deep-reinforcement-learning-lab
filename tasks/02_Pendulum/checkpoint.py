from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.config import ExperimentConfig, PPOHyperparameters
from models import build_ac_model

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
    action_dim: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    *,
    env_step: int | None = None,
    eval_mean_episode_return: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    hparams = experiment.hyperparameters
    if not isinstance(hparams, PPOHyperparameters):
        raise TypeError("save_ac_checkpoint requires PPO hyperparameters")

    payload: dict[str, Any] = {
        "state_dict": ac_model.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_sizes": list(hparams.hidden_sizes),
        "ac_type": hparams.ac_type,
        "action_low": action_low.tolist(),
        "action_high": action_high.tolist(),
        "subtask": experiment.subtask,
        "algorithm": experiment.algorithm,
        "run_name": experiment.run.name,
    }
    if env_step is not None:
        payload["env_step"] = env_step
    if eval_mean_episode_return is not None:
        payload["eval_mean_episode_return"] = eval_mean_episode_return
    torch.save(payload, path)
    return path


def load_ac_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> nn.Module:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"invalid checkpoint format: {path}")

    state_dim = int(payload["state_dim"])
    action_dim = int(payload["action_dim"])
    hidden_sizes = tuple(int(size) for size in payload["hidden_sizes"])
    ac_type = payload.get("ac_type", "shared")
    action_low = torch.tensor(payload["action_low"], dtype=torch.float32)
    action_high = torch.tensor(payload["action_high"], dtype=torch.float32)
    ac_model = build_ac_model(
        ac_type,
        state_dim,
        action_dim,
        hidden_sizes,
        action_low,
        action_high,
    )
    ac_model.load_state_dict(payload["state_dict"])
    ac_model.eval()
    return ac_model


@dataclass
class BestCheckpointTracker:
    best_eval: float = float("-inf")
    best_env_step: int | None = None
    path: Path | None = None

    def update_if_better(
        self,
        mean_eval: float,
        env_step: int,
        *,
        experiment: ExperimentConfig,
        log_dir: Path,
        ac_model: nn.Module,
        state_dim: int,
        action_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> bool:
        if mean_eval <= self.best_eval:
            return False
        self.best_eval = mean_eval
        self.best_env_step = env_step
        if not experiment.run.save_checkpoint:
            return False
        self.path = save_ac_checkpoint(
            best_checkpoint_path(log_dir),
            ac_model,
            experiment,
            state_dim,
            action_dim,
            action_low,
            action_high,
            env_step=env_step,
            eval_mean_episode_return=mean_eval,
        )
        return True
