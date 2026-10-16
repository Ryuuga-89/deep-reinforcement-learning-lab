from __future__ import annotations

from pathlib import Path

from deep_reinforcement_learning_lab.config import ExperimentConfig, PPOHyperparameters

from checkpoint import BestCheckpointTracker
from ac_model_registry import ppo_kind_for_model
from model_config import ModelConfig
from policy import EvalMetrics


def print_run_header(
    experiment: ExperimentConfig,
    config_path: Path,
    log_dir: Path,
    task_dir: Path,
    model: ModelConfig,
) -> None:
    run = experiment.run
    hparams = experiment.hyperparameters
    if not isinstance(hparams, PPOHyperparameters):
        raise TypeError("LunarLander task expects PPO hyperparameters")

    print()
    print(
        f"LunarLander {experiment.algorithm.upper()}  |  "
        f"run: {run.name}  |  subtask: {experiment.subtask}"
    )
    print(f"config: {config_path}")
    print("-" * 60)
    print(f"  seed                 {run.seed}")
    print(f"  max_env_steps        {run.max_env_steps}")
    print(f"  learning_rate        {hparams.learning_rate}")
    print(f"  n_envs               {hparams.n_envs}")
    print(f"  rollout_steps        {hparams.rollout_steps} (per env)")
    print(f"  rollout_transitions  {hparams.rollout_transitions}")
    print(f"  n_epochs             {hparams.n_epochs}")
    print(f"  minibatch_size       {hparams.minibatch_size}")
    print(f"  clip_coef            {hparams.clip_coef}")
    print(f"  hidden_sizes         {list(hparams.hidden_sizes)}")
    print(f"  ac_type              {hparams.ac_type}")
    print(f"  model.id             {model.id}")
    print(f"  model.ppo_kind       {ppo_kind_for_model(model)}")
    print(f"  log_interval_steps   {run.log_interval_steps}")
    print(f"  video_enabled          {run.video_enabled}")
    if run.video_enabled:
        print(f"  video_interval_steps   {run.video_interval_steps}")
        side = run.video_grid_side
        print(f"  video_episodes         {run.video_episodes} ({side}x{side} grid)")
    print(f"  log_dir              {log_dir}")
    print(f"  tensorboard          uv run tensorboard --logdir {task_dir / 'runs'}")
    print("-" * 60)
    print(f"{'Step':>7}  {'Train':>7}  {'Succ%':>6}  {'Ret':>7}  Note")
    print("-" * 60)


def print_progress_line(
    env_step: int,
    train_reward: float,
    metrics: EvalMetrics,
    *,
    note: str | None = None,
) -> None:
    success_pct = 100.0 * metrics.success_rate
    print(
        f"{env_step:7d}  {train_reward:7.1f}  {success_pct:5.1f}%  "
        f"{metrics.mean_episode_return:7.1f}  {note or ''}"
    )


def print_run_footer(
    experiment: ExperimentConfig,
    env_step: int,
    metrics: EvalMetrics,
    checkpoint_path: Path | None,
    best: BestCheckpointTracker,
) -> None:
    run = experiment.run
    print("-" * 60)
    print(f"Finished at env step {env_step} / {run.max_env_steps}")
    print(
        f"Final eval ({run.eval_episodes} ep): "
        f"success {100.0 * metrics.success_rate:.1f}%  "
        f"mean return {metrics.mean_episode_return:.1f}"
    )
    if checkpoint_path is not None:
        print(f"Checkpoint: {checkpoint_path}")
    if best.path is not None and best.best_env_step is not None:
        print(
            f"Best eval at env step {best.best_env_step}: "
            f"success {100.0 * best.best_success_rate:.1f}%  "
            f"mean return {best.best_mean_return:.1f}"
        )
        print(f"Best checkpoint: {best.path}")
    print()
