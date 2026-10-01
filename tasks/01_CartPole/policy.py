"""Greedy 方策まわり（評価・録画で共通）。"""

from __future__ import annotations

import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.config import TrainConfig
from env import make_env


def obs_to_tensor(observation) -> torch.Tensor:
    return torch.tensor(observation, dtype=torch.float32)


@torch.no_grad()
def greedy_action(q_model: nn.Module, observation) -> int:
    state = obs_to_tensor(observation).unsqueeze(0)
    return int(q_model(state).argmax(dim=1).item())


@torch.no_grad()
def evaluate_mean_return(
    q_model: nn.Module,
    config: TrainConfig,
    eval_seed: int,
) -> float:
    was_training = q_model.training
    q_model.eval()
    try:
        env = make_env(config)
        total_reward = 0.0
        for episode in range(config.eval_episodes):
            observation, _ = env.reset(seed=eval_seed + episode)
            done = False
            while not done:
                action = greedy_action(q_model, observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
        env.close()
        return total_reward / config.eval_episodes
    finally:
        q_model.train(was_training)
