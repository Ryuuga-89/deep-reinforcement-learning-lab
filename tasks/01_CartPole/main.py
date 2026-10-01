from __future__ import annotations

import sys
from pathlib import Path

from deep_reinforcement_learning_lab.algorithms.registry import (
    TrainingContext,
    register_algorithm,
    run_training,
)
from deep_reinforcement_learning_lab.config import load_experiment_config
from env import make_subtask_env
from train_dqn import train_dqn

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = TASK_DIR / "configs" / "baseline.yaml"

register_algorithm("dqn", train_dqn)


def resolve_config_path(arg: str | None) -> Path:
    if arg is None:
        return DEFAULT_CONFIG_PATH
    path = Path(arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def main() -> None:
    config_path = resolve_config_path(sys.argv[1] if len(sys.argv) > 1 else None)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    experiment = load_experiment_config(config_path)
    subtask = experiment.subtask

    def make_env(*, render_mode: str | None = None):
        return make_subtask_env(subtask, render_mode=render_mode)

    context = TrainingContext(
        experiment=experiment,
        config_path=config_path,
        task_dir=TASK_DIR,
        make_env=make_env,
    )
    try:
        run_training(context)
    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
