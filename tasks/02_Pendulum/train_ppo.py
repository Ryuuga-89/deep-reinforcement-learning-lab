from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from deep_reinforcement_learning_lab.algorithms.PPO import (
    ContinuousPPO,
    VectorContinuousFixedRolloutBuffer,
)
from deep_reinforcement_learning_lab.algorithms.registry import TrainingContext
from deep_reinforcement_learning_lab.config import (
    ExperimentConfig,
    PPOHyperparameters,
    RunConfig,
    ensure_run_dir_is_new,
    resolve_run_dir,
)
from checkpoint import BestCheckpointTracker, checkpoint_path, save_ac_checkpoint
from console import print_progress_line, print_run_footer, print_run_header
from env import make_sync_vector_env
from logger import TrainingLogger, attach_ppo_loss_logging
from models import build_ac_model
from policy import evaluate_mean_episode_return
from record import record_deterministic_grid_video, should_record_video


def should_log_metrics(env_step: int, run: RunConfig, n_envs: int) -> bool:
    return env_step == n_envs or env_step % run.log_interval_steps == 0


def log_eval_checkpoint_progress(
    env_step: int,
    train_reward: float,
    *,
    experiment: ExperimentConfig,
    ac_model: torch.nn.Module,
    logger: TrainingLogger,
    tracker: BestCheckpointTracker,
    run_dir: Path,
    state_dim: int,
    action_dim: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    record_video: bool,
    report_progress: bool = True,
) -> float:
    mean_eval = evaluate_mean_episode_return(ac_model, experiment)
    logger.log_eval_mean_episode_return(env_step, mean_eval)
    saved_best = tracker.update_if_better(
        mean_eval,
        env_step,
        experiment=experiment,
        log_dir=run_dir,
        ac_model=ac_model,
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
    )

    note_parts: list[str] = []
    if saved_best:
        note_parts.append("best ckpt")
    if record_video:
        video_path = record_deterministic_grid_video(
            ac_model, experiment, run_dir, env_step
        )
        note_parts.append(f"video -> {video_path.name}")

    if report_progress:
        print_progress_line(
            env_step,
            train_reward,
            mean_eval,
            note="  ".join(note_parts) or None,
        )
    return mean_eval


def train_ppo(context: TrainingContext) -> float:
    experiment = context.experiment
    if experiment.algorithm != "ppo":
        raise ValueError(f"expected algorithm ppo, got {experiment.algorithm}")

    run = experiment.run
    hparams = experiment.hyperparameters
    if not isinstance(hparams, PPOHyperparameters):
        raise TypeError("train_ppo requires PPOHyperparameters")

    n_envs = hparams.n_envs
    torch.manual_seed(run.seed)
    env = make_sync_vector_env(experiment, n_envs, seed=run.seed)

    state_dim = int(env.single_observation_space.shape[0])
    action_dim = int(env.single_action_space.shape[0])
    action_low = torch.tensor(env.single_action_space.low, dtype=torch.float32)
    action_high = torch.tensor(env.single_action_space.high, dtype=torch.float32)

    ac_model = build_ac_model(
        hparams.ac_type,
        state_dim,
        action_dim,
        hparams.hidden_sizes,
        action_low,
        action_high,
    )
    optimizer = torch.optim.Adam(ac_model.parameters(), lr=hparams.learning_rate)
    rollout_buffer = VectorContinuousFixedRolloutBuffer(
        hparams.rollout_steps,
        n_envs,
        state_dim,
        action_dim,
    )
    agent = ContinuousPPO(
        ac_model,
        optimizer,
        rollout_buffer,
        gamma=hparams.gamma,
        gae_lambda=hparams.gae_lambda,
        clip_coef=hparams.clip_coef,
        entropy_coef=hparams.entropy_coef,
        value_coef=hparams.value_coef,
        rollout_steps=hparams.rollout_steps,
        n_epochs=hparams.n_epochs,
        minibatch_size=hparams.minibatch_size,
        clip_value_coef=hparams.clip_value_coef,
        max_grad_norm=hparams.max_grad_norm,
    )

    run_dir = resolve_run_dir(context.task_dir, experiment)
    ensure_run_dir_is_new(run_dir)
    tracker = BestCheckpointTracker()

    print_run_header(experiment, context.config_path, run_dir, context.task_dir)

    observation, _ = env.reset(seed=run.seed)
    episode_rewards = np.zeros(n_envs, dtype=np.float64)
    episode_lengths = np.zeros(n_envs, dtype=np.int64)
    last_completed_train_reward = 0.0
    env_step = 0

    with TrainingLogger(log_dir=run_dir) as logger:
        attach_ppo_loss_logging(agent, logger)
        logger.log_experiment_start(experiment, context.config_path)

        while env_step < run.max_env_steps:
            states = torch.as_tensor(observation, dtype=torch.float32)
            actions, log_probs, values_old = agent.select_actions(states)
            next_observation, rewards, terminated, truncated, _ = env.step(
                actions.numpy().astype(np.float32)
            )
            bootstrap_done = terminated.astype(np.float32)

            agent.step_vector(
                states,
                actions,
                log_probs,
                torch.as_tensor(rewards, dtype=torch.float32),
                torch.as_tensor(bootstrap_done, dtype=torch.float32),
                values_old,
                torch.as_tensor(next_observation, dtype=torch.float32),
            )
            observation = next_observation
            env_step += n_envs

            for env_index in range(n_envs):
                episode_rewards[env_index] += rewards[env_index]
                episode_lengths[env_index] += 1
                if terminated[env_index] or truncated[env_index]:
                    last_completed_train_reward = float(episode_rewards[env_index])
                    logger.log_train_episode(
                        env_step,
                        last_completed_train_reward,
                        int(episode_lengths[env_index]),
                    )
                    episode_rewards[env_index] = 0.0
                    episode_lengths[env_index] = 0

            if should_log_metrics(env_step, run, n_envs):
                log_eval_checkpoint_progress(
                    env_step,
                    last_completed_train_reward,
                    experiment=experiment,
                    ac_model=ac_model,
                    logger=logger,
                    tracker=tracker,
                    run_dir=run_dir,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    action_low=action_low,
                    action_high=action_high,
                    record_video=should_record_video(env_step, run),
                )

        agent.flush_if_pending(torch.as_tensor(observation, dtype=torch.float32))
        env.close()

        mean_eval = log_eval_checkpoint_progress(
            env_step,
            last_completed_train_reward,
            experiment=experiment,
            ac_model=ac_model,
            logger=logger,
            tracker=tracker,
            run_dir=run_dir,
            state_dim=state_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            record_video=False,
            report_progress=False,
        )
        logger.log_experiment_end(experiment, mean_eval)

    ckpt: Path | None = None
    if run.save_checkpoint:
        ckpt = save_ac_checkpoint(
            checkpoint_path(run_dir),
            ac_model,
            experiment,
            state_dim,
            action_dim,
            action_low,
            action_high,
            env_step=env_step,
            eval_mean_episode_return=mean_eval,
        )

    print_run_footer(experiment, env_step, mean_eval, ckpt, tracker)
    return mean_eval
