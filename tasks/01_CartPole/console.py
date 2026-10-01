from __future__ import annotations

from pathlib import Path

from deep_reinforcement_learning_lab.config import ExperimentConfig

from checkpoint import BestCheckpointTracker


def print_run_header(
    experiment: ExperimentConfig,
    config_path: Path,
    log_dir: Path,
    task_dir: Path,
) -> None:
    run = experiment.run
    hparams = experiment.hyperparameters
    print()
    print(
        f"CartPole {experiment.algorithm.upper()}  |  "
        f"run: {run.name}  |  subtask: {experiment.subtask}"
    )
    print(f"config: {config_path}")
    print("-" * 60)
    print(f"  seed                 {run.seed}")
    print(f"  max_env_steps        {run.max_env_steps}")
    print(f"  learning_rate        {hparams.learning_rate}")
    print(f"  batch_size           {hparams.batch_size}")
    print(f"  hidden_sizes         {list(hparams.hidden_sizes)}")
    print(f"  epsilon_decay_steps  {hparams.epsilon_decay_steps}")
    print(f"  log_interval_steps   {run.log_interval_steps}")
    print(f"  video_enabled          {run.video_enabled}")
    if run.video_enabled:
        print(f"  video_interval_steps   {run.video_interval_steps}")
        side = run.video_grid_side
        print(f"  video_episodes         {run.video_episodes} ({side}x{side} grid)")
    print(f"  log_dir              {log_dir}")
    print(f"  tensorboard          uv run tensorboard --logdir {task_dir / 'runs'}")
    print("-" * 60)
    print(f"{'Step':>7}  {'Train':>7}  {'Eval':>7}  Note")
    print("-" * 60)


def print_progress_line(
    env_step: int,
    train_reward: float,
    eval_avg: float,
    *,
    note: str | None = None,
) -> None:
    print(f"{env_step:7d}  {train_reward:7.1f}  {eval_avg:7.1f}  {note or ''}")


def print_run_footer(
    experiment: ExperimentConfig,
    env_step: int,
    mean_eval: float,
    checkpoint_path: Path | None,
    best: BestCheckpointTracker,
) -> None:
    run = experiment.run
    print("-" * 60)
    print(f"Finished at env step {env_step} / {run.max_env_steps}")
    print(f"Final eval ({run.eval_episodes} ep avg): {mean_eval:.1f}")
    if checkpoint_path is not None:
        print(f"Checkpoint: {checkpoint_path}")
    if best.path is not None and best.best_env_step is not None:
        print(
            f"Best eval ({run.eval_episodes} ep avg): {best.best_eval:.1f} "
            f"at env step {best.best_env_step}"
        )
        print(f"Best checkpoint: {best.path}")
    print()
