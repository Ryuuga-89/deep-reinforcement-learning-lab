from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.algorithms.DQN import DQN, UniformReplayBuffer
from deep_reinforcement_learning_lab.algorithms.registry import TrainingContext
from deep_reinforcement_learning_lab.config import (
    DQNHyperparameters,
    ExperimentConfig,
    RunConfig,
    ensure_run_dir_is_new,
    resolve_run_dir,
)
from checkpoint import BestCheckpointTracker, checkpoint_path, save_qnet_checkpoint
from console import print_progress_line, print_run_footer, print_run_header
from logger import TrainingLogger, attach_loss_logging
from models import build_qnet
from policy import evaluate_mean_return, obs_to_tensor
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
    mean_eval = evaluate_mean_return(q_model, experiment)
    logger.log_eval(env_step, mean_eval)
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


def train_dqn(context: TrainingContext) -> float:
    experiment = context.experiment
    if experiment.algorithm != "dqn":
        raise ValueError(f"train_dqn expects algorithm dqn, got {experiment.algorithm}")

    run = experiment.run
    hparams: DQNHyperparameters = experiment.hyperparameters

    torch.manual_seed(run.seed)
    env = context.make_env()
    env.action_space.seed(run.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)

    q_model = build_qnet(state_dim, action_dim, hparams.hidden_sizes)
    optimizer = torch.optim.Adam(q_model.parameters(), lr=hparams.learning_rate)
    replay_buffer = UniformReplayBuffer(hparams.buffer_size)

    agent = DQN(
        q_model=q_model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        gamma=hparams.gamma,
        epsilon_start=hparams.epsilon_start,
        epsilon_end=hparams.epsilon_end,
        epsilon_rate=hparams.epsilon_rate,
        batch_size=hparams.batch_size,
        target_update_interval_step=hparams.target_update_interval,
        loss_type=hparams.loss_type,
    )

    run_dir = resolve_run_dir(context.task_dir, experiment)
    ensure_run_dir_is_new(run_dir)
    tracker = BestCheckpointTracker()

    print_run_header(experiment, context.config_path, run_dir, context.task_dir)

    episode_index = 1
    observation, _ = env.reset(seed=run.seed + episode_index)
    episode_reward = 0.0
    env_step = 0

    with TrainingLogger(log_dir=run_dir) as logger:
        attach_loss_logging(agent, logger)

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

            if done:
                logger.log_episode(env_step, episode_reward)
                logger.log_epsilon(env_step, agent)
                episode_index += 1
                if env_step < run.max_env_steps:
                    observation, _ = env.reset(seed=run.seed + episode_index)
                    episode_reward = 0.0

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

    ckpt: Path | None = None
    if run.save_checkpoint:
        ckpt = save_qnet_checkpoint(
            checkpoint_path(run_dir),
            q_model,
            experiment,
            state_dim,
            action_dim,
            env_step=env_step,
            eval_score=mean_eval,
        )

    print_run_footer(experiment, env_step, mean_eval, ckpt, tracker)
    return mean_eval
