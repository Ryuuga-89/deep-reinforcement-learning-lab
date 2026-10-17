from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from deep_reinforcement_learning_lab.algorithms.registry import (
    TrainingContext,
    register_algorithm,
    run_training,
)
from deep_reinforcement_learning_lab.config import ExperimentConfig
from config_loader import load_lunar_experiment
from env import make_env_for_experiment
from run_paths import resolve_run_dir
from train_ppo import train_ppo

TASK_DIR = Path(__file__).resolve().parent

register_algorithm("ppo", train_ppo)


def resolve_config_path(arg: str) -> Path:
    path = Path(arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_multiseed_training_config(path: Path) -> tuple[Path, list[int]]:
    lunar = load_lunar_experiment(path)
    if lunar.seeds is None:
        raise ValueError(
            f"config has no top-level seeds: {path}. "
            "Add seeds: [42, 43, ...] for run_seeds.py, or use main.py for a single run."
        )
    return path, list(lunar.seeds)


def experiment_for_seed(
    experiment: ExperimentConfig,
    seed: int,
) -> ExperimentConfig:
    return replace(experiment, run=replace(experiment.run, seed=seed))


def run_seed_sweep(
    config_path: Path,
    *,
    skip_existing: bool,
) -> None:
    base_config_path, seeds = load_multiseed_training_config(config_path)
    lunar_base = load_lunar_experiment(base_config_path)
    base_experiment = lunar_base.experiment
    for seed in seeds:
        experiment = experiment_for_seed(base_experiment, seed)
        run_dir = resolve_run_dir(TASK_DIR, experiment)
        if run_dir.exists():
            if skip_existing:
                print(f"skip existing seed {seed}: {run_dir}", file=sys.stderr)
                continue
            raise FileExistsError(
                f"run_dir already exists: {run_dir}. "
                "Use --skip-existing or remove the directory."
            )

        def make_env(*, render_mode: str | None = None):
            return make_env_for_experiment(experiment, render_mode=render_mode)

        context = TrainingContext(
            experiment=experiment,
            config_path=base_config_path,
            task_dir=TASK_DIR,
            make_env=make_env,
        )
        print(f"=== seed {seed} ===", file=sys.stderr)
        run_training(context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LunarLander training for multiple seeds.")
    parser.add_argument(
        "config",
        help="Path to train YAML with top-level seeds: [ ... ].",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip seeds whose run_dir already exists.",
    )
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    try:
        run_seed_sweep(config_path, skip_existing=args.skip_existing)
    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
