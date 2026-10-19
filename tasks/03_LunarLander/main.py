from __future__ import annotations

import sys
from pathlib import Path

from deep_reinforcement_learning_lab.algorithms.registry import (
    TrainingContext,
    register_algorithm,
    run_training,
)
from config_loader import load_lunar_experiment
from env import make_env_for_experiment
from train_ppo import train_ppo

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = (
    TASK_DIR / "configs" / "default" / "baseline_continuous_preview.yaml"
)

register_algorithm("ppo", train_ppo)


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
    lunar = load_lunar_experiment(config_path)
    experiment = lunar.experiment

    def make_env(*, render_mode: str | None = None):
        return make_env_for_experiment(experiment, render_mode=render_mode)

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
