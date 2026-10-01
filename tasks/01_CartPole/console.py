from __future__ import annotations

from pathlib import Path

from deep_reinforcement_learning_lab.config import TrainConfig

from checkpoint import BestCheckpointTracker


def print_run_header(config: TrainConfig, config_path: Path, log_dir: Path) -> None:
    print()
    print(f"CartPole DQN  |  run: {config.run_name}")
    print(f"config: {config_path}")
    print("-" * 60)
    print(f"  env_id               {config.env_id}")
    print(f"  seed                 {config.seed}")
    print(f"  max_env_steps        {config.max_env_steps}")
    print(f"  learning_rate        {config.learning_rate}")
    print(f"  batch_size           {config.batch_size}")
    print(f"  hidden_sizes         {list(config.hidden_sizes)}")
    print(f"  reward_mode          {config.reward_mode}")
    print(f"  epsilon_decay_steps  {config.epsilon_decay_steps}")
    print(f"  log_interval_steps   {config.log_interval_steps}")
    if config.video_interval_steps > 0:
        print(f"  video_interval_steps {config.video_interval_steps}")
    print(f"  log_dir              {log_dir}")
    print(f"  tensorboard          uv run tensorboard --logdir {log_dir.parent}")
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
    config: TrainConfig,
    env_step: int,
    mean_eval: float,
    checkpoint_path: Path | None,
    best: BestCheckpointTracker,
) -> None:
    print("-" * 60)
    print(f"Finished at env step {env_step} / {config.max_env_steps}")
    print(f"Final eval ({config.eval_episodes} ep avg): {mean_eval:.1f}")
    if checkpoint_path is not None:
        print(f"Checkpoint: {checkpoint_path}")
    if best.path is not None and best.best_env_step is not None:
        print(
            f"Best eval ({config.eval_episodes} ep avg): {best.best_eval:.1f} "
            f"at env step {best.best_env_step}"
        )
        print(f"Best checkpoint: {best.path}")
    print()
