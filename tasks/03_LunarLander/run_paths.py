from __future__ import annotations

from pathlib import Path

from deep_reinforcement_learning_lab.config import ExperimentConfig


def resolve_run_dir(task_dir: Path, experiment: ExperimentConfig) -> Path:
    """学習成果物: ``runs/<subtask>/<algorithm>/<run.name>/seed_XXXX/``."""
    return (
        task_dir
        / "runs"
        / experiment.subtask
        / experiment.algorithm
        / experiment.run.name
        / f"seed_{experiment.run.seed:04d}"
    )


def resolve_run_group_dir(task_dir: Path, experiment: ExperimentConfig) -> Path:
    """同一設定（run.name）の seed 群の親ディレクトリ。"""
    return (
        task_dir
        / "runs"
        / experiment.subtask
        / experiment.algorithm
        / experiment.run.name
    )
