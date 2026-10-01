"""Greedy 方策まわり（評価・録画で共通）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.config import ExperimentConfig
from env import make_subtask_env


def obs_to_tensor(observation) -> torch.Tensor:
    return torch.tensor(observation, dtype=torch.float32)


@torch.no_grad()
def greedy_action(q_model: nn.Module, observation) -> int:
    state = obs_to_tensor(observation).unsqueeze(0)
    return int(q_model(state).argmax(dim=1).item())


@dataclass(frozen=True)
class GreedyEpisodeResult:
    """greedy 1 エピソードの結果（評価・録画共通）。

    ``frames[i]`` は環境ステップ ``i`` の描画（``i == 0`` は ``reset`` 直後）。
    ``fail_frame_index`` は ``terminated`` 時の最終フレーム位置。``truncated`` のみなら ``None``。
    """

    total_reward: float
    frames: tuple[np.ndarray, ...]
    fail_frame_index: int | None
    ended_by_termination: bool

    def display_step_at(self, time_index: int) -> int:
        """録画タイムライン上で表示する環境ステップ（終了・失敗後は固定）。"""
        last_frame_index = len(self.frames) - 1
        if self.fail_frame_index is not None:
            return min(time_index, self.fail_frame_index)
        return min(time_index, last_frame_index)


@torch.no_grad()
def run_greedy_episode(
    q_model: nn.Module,
    *,
    subtask: str,
    seed: int,
    render: bool,
) -> GreedyEpisodeResult:
    """1 エピソード greedy 実行。``render=True`` のときフレーム列も返す。"""
    make_kwargs: dict[str, str] = {}
    if render:
        make_kwargs["render_mode"] = "rgb_array"
    env = make_subtask_env(subtask, **make_kwargs)

    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    frames: list[np.ndarray] = []
    if render:
        frames.append(env.render())

    fail_frame_index: int | None = None
    ended_by_termination = False
    while True:
        action = greedy_action(q_model, observation)
        observation, reward, terminated, truncated, _ = env.step(action)
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
    return GreedyEpisodeResult(
        total_reward=total_reward,
        frames=tuple(frames if render else ()),
        fail_frame_index=fail_frame_index,
        ended_by_termination=ended_by_termination,
    )


@torch.no_grad()
def evaluate_mean_return(q_model: nn.Module, experiment: ExperimentConfig) -> float:
    was_training = q_model.training
    q_model.eval()
    try:
        run = experiment.run
        total_reward = 0.0
        for episode in range(run.eval_episodes):
            result = run_greedy_episode(
                q_model,
                subtask=experiment.subtask,
                seed=run.eval_rollout_seed(episode),
                render=False,
            )
            total_reward += result.total_reward
        return total_reward / run.eval_episodes
    finally:
        q_model.train(was_training)
