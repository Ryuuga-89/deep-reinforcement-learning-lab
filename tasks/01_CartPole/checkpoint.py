from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.config import TrainConfig
from models import QNet, build_qnet

CHECKPOINT_FILENAME = "qnet.pt"
BEST_CHECKPOINT_FILENAME = "qnet_best.pt"


def checkpoint_path(log_dir: Path) -> Path:
    return log_dir / CHECKPOINT_FILENAME


def best_checkpoint_path(log_dir: Path) -> Path:
    return log_dir / BEST_CHECKPOINT_FILENAME


def save_qnet_checkpoint(
    path: Path,
    q_model: nn.Module,
    config: TrainConfig,
    state_dim: int,
    action_dim: int,
    *,
    env_step: int | None = None,
    eval_score: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": q_model.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_sizes": list(config.hidden_sizes),
        "env_id": config.env_id,
        "reward_mode": config.reward_mode,
        "run_name": config.run_name,
    }
    if env_step is not None:
        payload["env_step"] = env_step
    if eval_score is not None:
        payload["eval_score"] = eval_score
    torch.save(payload, path)
    return path


def load_qnet_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> QNet:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"invalid checkpoint format: {path}")

    state_dim = int(payload["state_dim"])
    action_dim = int(payload["action_dim"])
    hidden_sizes = tuple(int(size) for size in payload["hidden_sizes"])
    q_model = build_qnet(state_dim, action_dim, hidden_sizes)
    q_model.load_state_dict(payload["state_dict"])
    q_model.eval()
    return q_model


@dataclass
class BestCheckpointTracker:
    """Eval 平均報酬が更新されたときだけ best ckpt を上書きする。"""

    best_eval: float = float("-inf")
    best_env_step: int | None = None
    path: Path | None = None

    def update_if_better(
        self,
        mean_eval: float,
        env_step: int,
        *,
        config: TrainConfig,
        log_dir: Path,
        q_model: nn.Module,
        state_dim: int,
        action_dim: int,
    ) -> bool:
        if mean_eval <= self.best_eval:
            return False
        self.best_eval = mean_eval
        self.best_env_step = env_step
        if not config.save_checkpoint:
            return False
        self.path = save_qnet_checkpoint(
            best_checkpoint_path(log_dir),
            q_model,
            config,
            state_dim,
            action_dim,
            env_step=env_step,
            eval_score=mean_eval,
        )
        return True
