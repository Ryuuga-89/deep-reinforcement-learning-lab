from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from deep_reinforcement_learning_lab.config import TrainConfig
from env import make_env
from policy import greedy_action

PLAYBACK_SPEED = 0.5
VIDEO_SEED_OFFSET = 20_000


def should_record_video(env_step: int, config: TrainConfig) -> bool:
    if config.video_interval_steps <= 0:
        return False
    return env_step % config.video_interval_steps == 0


def video_output_path(log_dir: Path, env_step: int) -> Path:
    return log_dir / "videos" / f"train_step{env_step:07d}.mp4"


@torch.no_grad()
def record_greedy_episode(
    q_model: nn.Module,
    config: TrainConfig,
    log_dir: Path,
    env_step: int,
) -> Path:
    """現在の Q ネットを greedy で 1 エピソード実行し、MP4 を保存する。"""
    was_training = q_model.training
    q_model.eval()

    video_dir = log_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_path = video_output_path(log_dir, env_step)

    env = make_env(config, render_mode="rgb_array")
    sim_fps = int(env.metadata.get("render_fps", 50))
    video_fps = max(1, int(sim_fps * PLAYBACK_SPEED))

    rollout_seed = config.seed + VIDEO_SEED_OFFSET + env_step
    observation, _ = env.reset(seed=rollout_seed)
    frames = [env.render()]
    done = False
    while not done:
        action = greedy_action(q_model, observation)
        observation, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        done = terminated or truncated

    env.close()
    q_model.train(was_training)

    clip = ImageSequenceClip(frames, fps=video_fps)
    clip.write_videofile(
        str(output_path),
        logger=None,
        write_logfile=False,
    )
    clip.close()
    return output_path
