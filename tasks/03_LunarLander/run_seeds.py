from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import yaml

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
CONFIGS_DIR = TASK_DIR / "configs"

register_algorithm("ppo", train_ppo)


def resolve_sweep_path(arg: str) -> Path:
    path = Path(arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_seed_sweep_config(path: Path) -> tuple[Path, list[int]]:
    with path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"seed sweep config must be a mapping: {path}")

    allowed = {"base_config", "seeds"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown keys in seed sweep config {path}: {sorted(unknown)}")

    base_rel = raw.get("base_config")
    seeds_raw = raw.get("seeds")
    if not isinstance(base_rel, str) or not base_rel.strip():
        raise ValueError(f"base_config must be a non-empty string: {path}")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError(f"seeds must be a non-empty list: {path}")

    seeds: list[int] = []
    for item in seeds_raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"seeds entries must be integers: {path}")
        seeds.append(item)

    base_path = CONFIGS_DIR / base_rel.strip()
    if not base_path.is_file():
        raise FileNotFoundError(f"base_config not found: {base_path}")
    return base_path, seeds


def experiment_for_seed(
    experiment: ExperimentConfig,
    seed: int,
) -> ExperimentConfig:
    return replace(experiment, run=replace(experiment.run, seed=seed))


def run_seed_sweep(
    sweep_path: Path,
    *,
    skip_existing: bool,
) -> None:
    base_config_path, seeds = load_seed_sweep_config(sweep_path)
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
        "sweep_config",
        help="Path to multiseed YAML (base_config + seeds).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip seeds whose run_dir already exists.",
    )
    args = parser.parse_args()
    sweep_path = resolve_sweep_path(args.sweep_config)
    if not sweep_path.is_file():
        raise FileNotFoundError(f"sweep config not found: {sweep_path}")
    try:
        run_seed_sweep(sweep_path, skip_existing=args.skip_existing)
    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
