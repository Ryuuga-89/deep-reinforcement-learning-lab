from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.algorithms.DQN import DQN, UniformReplayBuffer
from deep_reinforcement_learning_lab.config import TrainConfig, load_train_config
from checkpoint import BestCheckpointTracker, checkpoint_path, save_qnet_checkpoint
from console import print_progress_line, print_run_footer, print_run_header
from env import make_env
from logger import TrainingLogger, attach_loss_logging
from models import build_qnet
from policy import evaluate_mean_return, obs_to_tensor
from record import record_greedy_episode, should_record_video

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = TASK_DIR / "configs" / "baseline.yaml"
EVAL_SEED_OFFSET = 10_000


def resolve_config_path(arg: str | None) -> Path:
    if arg is None:
        return DEFAULT_CONFIG_PATH
    path = Path(arg)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def should_log_metrics(env_step: int, config: TrainConfig) -> bool:
    return env_step == 1 or env_step % config.log_interval_steps == 0


def log_eval_checkpoint_progress(
    env_step: int,
    episode_reward: float,
    *,
    q_model: nn.Module,
    config: TrainConfig,
    eval_seed: int,
    logger: TrainingLogger,
    tracker: BestCheckpointTracker,
    log_dir: Path,
    state_dim: int,
    action_dim: int,
    record_video: bool,
    report_progress: bool = True,
) -> float:
    mean_eval = evaluate_mean_return(q_model, config, eval_seed)
    logger.log_eval(env_step, mean_eval)
    saved_best = tracker.update_if_better(
        mean_eval,
        env_step,
        config=config,
        log_dir=log_dir,
        q_model=q_model,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    note_parts: list[str] = []
    if saved_best:
        note_parts.append("best ckpt")
    if record_video:
        video_path = record_greedy_episode(q_model, config, log_dir, env_step)
        note_parts.append(f"video -> {video_path.name}")

    if report_progress:
        print_progress_line(
            env_step,
            episode_reward,
            mean_eval,
            note="  ".join(note_parts) or None,
        )
    return mean_eval


def train(config: TrainConfig, config_path: Path) -> float:
    torch.manual_seed(config.seed)
    env = make_env(config)
    env.action_space.seed(config.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)

    q_model = build_qnet(state_dim, action_dim, config.hidden_sizes)
    optimizer = torch.optim.Adam(q_model.parameters(), lr=config.learning_rate)
    replay_buffer = UniformReplayBuffer(config.buffer_size)

    agent = DQN(
        q_model=q_model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_rate=config.epsilon_rate,
        batch_size=config.batch_size,
        target_update_interval_step=config.target_update_interval,
        loss_type=config.loss_type,
    )

    log_dir = TASK_DIR / "runs" / config.run_name
    eval_seed = config.seed + EVAL_SEED_OFFSET
    tracker = BestCheckpointTracker()

    print_run_header(config, config_path, log_dir)

    episode_index = 1
    observation, _ = env.reset(seed=config.seed + episode_index)
    episode_reward = 0.0
    env_step = 0

    with TrainingLogger(log_dir=log_dir) as logger:
        attach_loss_logging(agent, logger)

        while env_step < config.max_env_steps:
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
                if env_step < config.max_env_steps:
                    observation, _ = env.reset(seed=config.seed + episode_index)
                    episode_reward = 0.0

            if should_log_metrics(env_step, config):
                log_eval_checkpoint_progress(
                    env_step,
                    episode_reward,
                    q_model=q_model,
                    config=config,
                    eval_seed=eval_seed,
                    logger=logger,
                    tracker=tracker,
                    log_dir=log_dir,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    record_video=should_record_video(env_step, config),
                )

        env.close()

        mean_eval = log_eval_checkpoint_progress(
            env_step,
            episode_reward,
            q_model=q_model,
            config=config,
            eval_seed=eval_seed,
            logger=logger,
            tracker=tracker,
            log_dir=log_dir,
            state_dim=state_dim,
            action_dim=action_dim,
            record_video=False,
            report_progress=False,
        )

    ckpt: Path | None = None
    if config.save_checkpoint:
        ckpt = save_qnet_checkpoint(
            checkpoint_path(log_dir),
            q_model,
            config,
            state_dim,
            action_dim,
            env_step=env_step,
            eval_score=mean_eval,
        )

    print_run_footer(config, env_step, mean_eval, ckpt, tracker)
    return mean_eval


def main() -> None:
    config_path = resolve_config_path(sys.argv[1] if len(sys.argv) > 1 else None)
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")
    config = load_train_config(config_path)
    train(config, config_path)


if __name__ == "__main__":
    main()
