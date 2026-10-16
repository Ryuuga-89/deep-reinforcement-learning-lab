from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ac_model_registry import PpoKind
from deep_reinforcement_learning_lab.config import ExperimentConfig
from policy import EvalMetrics

EVAL_FINAL_FILENAME = "eval_final.json"


def eval_final_path(run_dir: Path) -> Path:
    return run_dir / EVAL_FINAL_FILENAME


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
) -> Path:
    payload: dict[str, Any] = {
        "seed": experiment.run.seed,
        "env_step": env_step,
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
    path = eval_final_path(run_dir)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_eval_final(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "metrics" not in raw:
        raise ValueError(f"invalid eval_final.json: {path}")
    return raw
