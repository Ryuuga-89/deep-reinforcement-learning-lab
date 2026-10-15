"""Deterministic 方策まわり（評価・録画で共通）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.config import ExperimentConfig
from env import make_env_for_experiment


def obs_to_tensor(observation) -> torch.Tensor:
    return torch.tensor(observation, dtype=torch.float32)


@torch.no_grad()
def deterministic_action(ac_model: nn.Module, observation) -> np.ndarray:
    state = obs_to_tensor(observation).unsqueeze(0)
    action = ac_model.deterministic_action(state)
    return action.squeeze(0).numpy().astype(np.float32)


@dataclass(frozen=True)
class DeterministicEpisodeResult:
    total_reward: float
    episode_steps: int
    frames: tuple[np.ndarray, ...]
    fail_frame_index: int | None
    ended_by_termination: bool

    def display_step_at(self, time_index: int) -> int:
        last_frame_index = len(self.frames) - 1
        if self.fail_frame_index is not None:
            return min(time_index, self.fail_frame_index)
        return min(time_index, last_frame_index)


@torch.no_grad()
def run_deterministic_episode(
    ac_model: nn.Module,
    *,
    experiment: ExperimentConfig,
    seed: int,
    render: bool,
) -> DeterministicEpisodeResult:
    render_mode = "rgb_array" if render else None
    env = make_env_for_experiment(experiment, render_mode=render_mode)

    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    frames: list[np.ndarray] = []
    if render:
        frames.append(env.render())

    fail_frame_index: int | None = None
    ended_by_termination = False
    episode_steps = 0
    while True:
        action = deterministic_action(ac_model, observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_steps += 1
        total_reward += reward
        if render:
            frames.append(env.render())
        if terminated:
            ended_by_termination = True
            if render:
                fail_frame_index = len(frames) - 1
            break
        if truncated:
            break

    env.close()
    return DeterministicEpisodeResult(
        total_reward=total_reward,
        episode_steps=episode_steps,
        frames=tuple(frames if render else ()),
        fail_frame_index=fail_frame_index,
        ended_by_termination=ended_by_termination,
    )


@torch.no_grad()
def evaluate_mean_episode_return(
    ac_model: nn.Module,
    experiment: ExperimentConfig,
) -> float:
    was_training = ac_model.training
    ac_model.eval()
    try:
        run = experiment.run
        total_return = 0.0
        for episode in range(run.eval_episodes):
            result = run_deterministic_episode(
                ac_model,
                experiment=experiment,
                seed=run.eval_rollout_seed(episode),
                render=False,
            )
            total_return += result.total_reward
        return total_return / run.eval_episodes
    finally:
        ac_model.train(was_training)
