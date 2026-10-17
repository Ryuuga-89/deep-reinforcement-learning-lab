from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from ac_model_registry import PpoKind
from checkpoint import BestCheckpointTracker, best_checkpoint_path, load_ac_checkpoint
from deep_reinforcement_learning_lab.config import ExperimentConfig
from policy import EvalMetrics, evaluate_policy

EvalSource = Literal["best_checkpoint", "last_weights"]

EVAL_FINAL_FILENAME = "eval_final.json"


def eval_final_path(run_dir: Path) -> Path:
    return run_dir / EVAL_FINAL_FILENAME


@dataclass(frozen=True)
class EvalExportResult:
    metrics: EvalMetrics
    env_step: int
    eval_source: EvalSource


def build_eval_export(
    run_dir: Path,
    experiment: ExperimentConfig,
    tracker: BestCheckpointTracker,
    ac_model: torch.nn.Module,
    training_end_env_step: int,
) -> EvalExportResult:
    best_path = best_checkpoint_path(run_dir)
    if best_path.is_file():
        best_model = load_ac_checkpoint(best_path)
        metrics = evaluate_policy(best_model, experiment)
        env_step = tracker.best_env_step
        if env_step is None:
            payload = torch.load(best_path, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and payload.get("env_step") is not None:
                env_step = int(payload["env_step"])
            else:
                env_step = training_end_env_step
        return EvalExportResult(metrics, env_step, "best_checkpoint")

    metrics = evaluate_policy(ac_model, experiment)
    return EvalExportResult(metrics, training_end_env_step, "last_weights")


def save_eval_final(
    run_dir: Path,
    metrics: EvalMetrics,
    *,
    experiment: ExperimentConfig,
    config_path: Path,
    env_step: int,
    model_id: str | None = None,
    ppo_kind: PpoKind | None = None,
    model_options: dict[str, object] | None = None,
    eval_source: EvalSource = "best_checkpoint",
    training_end_env_step: int | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "seed": experiment.run.seed,
        "env_step": env_step,
        "eval_source": eval_source,
        "config_path": str(config_path.resolve()),
        "subtask": experiment.subtask,
        "algorithm": experiment.algorithm,
        "run_name": experiment.run.name,
        "metrics": asdict(metrics),
    }
    if model_id is not None:
        payload["model_id"] = model_id
    if ppo_kind is not None:
        payload["policy_kind"] = ppo_kind
    if model_options:
        payload["model_options"] = model_options
    if training_end_env_step is not None:
        payload["training_end_env_step"] = training_end_env_step
    path = eval_final_path(run_dir)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_eval_final(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "metrics" not in raw:
        raise ValueError(f"invalid eval_final.json: {path}")
    return raw
