from __future__ import annotations

from pathlib import Path
from typing import Type

import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.algorithms.DQN import (
    DDQN,
    DQN,
    PER_DDQN,
    PER_DQN,
    PrioritizedReplayBuffer,
    UniformReplayBuffer,
)
from deep_reinforcement_learning_lab.algorithms.registry import TrainingContext
from deep_reinforcement_learning_lab.config import (
    DQNHyperparameters,
    ExperimentConfig,
    PERHyperparameters,
    RunConfig,
    ensure_run_dir_is_new,
    resolve_run_dir,
)
from checkpoint import BestCheckpointTracker, checkpoint_path, save_qnet_checkpoint
from console import print_progress_line, print_run_footer, print_run_header
from logger import TrainingLogger, attach_loss_logging
from models import build_q_model
from policy import evaluate_mean_episode_length, obs_to_tensor
from record import record_greedy_grid_video, should_record_video


def should_log_metrics(env_step: int, run: RunConfig) -> bool:
    return env_step == 1 or env_step % run.log_interval_steps == 0


def log_eval_checkpoint_progress(
    env_step: int,
    episode_reward: float,
    *,
    experiment: ExperimentConfig,
    q_model: nn.Module,
    logger: TrainingLogger,
    tracker: BestCheckpointTracker,
    run_dir: Path,
    state_dim: int,
    action_dim: int,
    record_video: bool,
    report_progress: bool = True,
) -> float:
    mean_eval = evaluate_mean_episode_length(q_model, experiment)
    logger.log_eval_mean_episode_length(env_step, mean_eval)
    saved_best = tracker.update_if_better(
        mean_eval,
        env_step,
        experiment=experiment,
        log_dir=run_dir,
        q_model=q_model,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    note_parts: list[str] = []
    if saved_best:
        note_parts.append("best ckpt")
    if record_video:
        video_path = record_greedy_grid_video(q_model, experiment, run_dir, env_step)
        note_parts.append(f"video -> {video_path.name}")

    if report_progress:
        print_progress_line(
            env_step,
            episode_reward,
            mean_eval,
            note="  ".join(note_parts) or None,
        )
    return mean_eval


def _build_agent(
    agent_cls: Type[DQN],
    *,
    state_dim: int,
    action_dim: int,
    hparams: DQNHyperparameters | PERHyperparameters,
    max_env_steps: int,
) -> tuple[DQN, nn.Module]:
    q_model = build_q_model(
        hparams.qnet_type, state_dim, action_dim, hparams.hidden_sizes
    )
    optimizer = torch.optim.Adam(q_model.parameters(), lr=hparams.learning_rate)

    per_kwargs: dict[str, float | int] = {}
    if isinstance(hparams, PERHyperparameters):
        replay_buffer = PrioritizedReplayBuffer(hparams.buffer_size, hparams.per_alpha)
        per_kwargs = {
            "per_beta_start": hparams.per_beta_start,
            "per_beta_end": hparams.per_beta_end,
            "beta_anneal_steps": max_env_steps,
            "priority_epsilon": hparams.priority_epsilon,
        }
    else:
        replay_buffer = UniformReplayBuffer(hparams.buffer_size)

    if hparams.tau is not None:
        target_sync_kwargs = {"tau": hparams.tau}
    else:
        if hparams.target_update_interval is None:
            raise ValueError("target_update_interval is required when tau is not set")
        target_sync_kwargs = {
            "target_update_interval_step": hparams.target_update_interval,
        }
    agent = agent_cls(
        q_model=q_model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        gamma=hparams.gamma,
        epsilon_start=hparams.epsilon_start,
        epsilon_end=hparams.epsilon_end,
        epsilon_rate=hparams.epsilon_rate,
        batch_size=hparams.batch_size,
        loss_type=hparams.loss_type,
        **target_sync_kwargs,
        **per_kwargs,
    )
    return agent, q_model


def _train_q_learning(
    context: TrainingContext,
    agent_cls: Type[DQN],
    expected_algorithm: str,
) -> float:
    experiment = context.experiment
    if experiment.algorithm != expected_algorithm:
        raise ValueError(
            f"expected algorithm {expected_algorithm}, got {experiment.algorithm}"
        )

    run = experiment.run
    hparams = experiment.hyperparameters

    torch.manual_seed(run.seed)
    env = context.make_env()
    env.action_space.seed(run.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)

    agent, q_model = _build_agent(
        agent_cls,
        state_dim=state_dim,
        action_dim=action_dim,
        hparams=hparams,
        max_env_steps=run.max_env_steps,
    )

    run_dir = resolve_run_dir(context.task_dir, experiment)
    ensure_run_dir_is_new(run_dir)
    tracker = BestCheckpointTracker()

    print_run_header(experiment, context.config_path, run_dir, context.task_dir)

    episode_index = 1
    observation, _ = env.reset(seed=run.seed + episode_index)
    episode_reward = 0.0
    episode_length = 0
    env_step = 0

    with TrainingLogger(log_dir=run_dir) as logger:
        attach_loss_logging(agent, logger)
        logger.log_experiment_start(experiment, context.config_path)

        while env_step < run.max_env_steps:
            state = obs_to_tensor(observation)
            action = agent.select_action(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = obs_to_tensor(next_observation)
            done = terminated or truncated
            bootstrap_done = terminated

            agent.step(state, action, reward, next_state, bootstrap_done)
            observation = next_observation
            env_step += 1
            episode_reward += reward
            episode_length += 1

            if done:
                logger.log_train_episode(env_step, episode_reward, episode_length)
                logger.log_train_epsilon(env_step, agent)
                episode_index += 1
                if env_step < run.max_env_steps:
                    observation, _ = env.reset(seed=run.seed + episode_index)
                    episode_reward = 0.0
                    episode_length = 0

            if should_log_metrics(env_step, run):
                log_eval_checkpoint_progress(
                    env_step,
                    episode_reward,
                    experiment=experiment,
                    q_model=q_model,
                    logger=logger,
                    tracker=tracker,
                    run_dir=run_dir,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    record_video=should_record_video(env_step, run),
                )

        env.close()

        mean_eval = log_eval_checkpoint_progress(
            env_step,
            episode_reward,
            experiment=experiment,
            q_model=q_model,
            logger=logger,
            tracker=tracker,
            run_dir=run_dir,
            state_dim=state_dim,
            action_dim=action_dim,
            record_video=False,
            report_progress=False,
        )
        logger.log_experiment_end(experiment, mean_eval)

    ckpt: Path | None = None
    if run.save_checkpoint:
        ckpt = save_qnet_checkpoint(
            checkpoint_path(run_dir),
            q_model,
            experiment,
            state_dim,
            action_dim,
            env_step=env_step,
            eval_mean_episode_steps=mean_eval,
        )

    print_run_footer(experiment, env_step, mean_eval, ckpt, tracker)
    return mean_eval


def train_dqn(context: TrainingContext) -> float:
    return _train_q_learning(context, DQN, "dqn")


def train_ddqn(context: TrainingContext) -> float:
    return _train_q_learning(context, DDQN, "ddqn")


def train_per_dqn(context: TrainingContext) -> float:
    return _train_q_learning(context, PER_DQN, "per_dqn")


def train_per_ddqn(context: TrainingContext) -> float:
    return _train_q_learning(context, PER_DDQN, "per_ddqn")
