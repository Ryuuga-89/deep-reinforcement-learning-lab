"""Deterministic 方策まわり（評価・録画で共通）。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.config import ExperimentConfig
from env import make_env_for_experiment

# Gymnasium LunarLander-v3 EnvSpec.reward_threshold
SOLVED_RETURN_THRESHOLD = 200.0
LAND_TERMINAL_REWARD = 100.0
CRASH_TERMINAL_REWARD = -100.0
MAIN_ENGINE_FUEL_COEF = 0.30
SIDE_ENGINE_FUEL_COEF = 0.03


class EpisodeOutcome(str, Enum):
    LAND_SUCCESS = "land_success"
    CRASH = "crash"
    TIMEOUT = "timeout"


def obs_to_tensor(observation) -> torch.Tensor:
    return torch.tensor(observation, dtype=torch.float32)


def fuel_penalty_for_continuous_action(action: np.ndarray) -> tuple[float, float, float]:
    """環境と同じ係数で 1 ステップの燃料ペナルティ（main, side, total）。"""
    action_arr = np.asarray(action, dtype=np.float64)
    m_power = 0.0
    s_power = 0.0
    if action_arr[0] > 0.0:
        m_power = (np.clip(action_arr[0], 0.0, 1.0) + 1.0) * 0.5
    lateral = float(action_arr[1])
    if abs(lateral) > 0.5:
        s_power = float(np.clip(abs(lateral), 0.5, 1.0))
    main_penalty = m_power * MAIN_ENGINE_FUEL_COEF
    side_penalty = s_power * SIDE_ENGINE_FUEL_COEF
    return main_penalty, side_penalty, main_penalty + side_penalty


def classify_episode(
    *,
    total_reward: float,
    last_step_reward: float,
    truncated: bool,
    terminated: bool,
) -> EpisodeOutcome:
    if truncated:
        return EpisodeOutcome.TIMEOUT
    if terminated:
        if last_step_reward >= LAND_TERMINAL_REWARD - 1.0:
            return EpisodeOutcome.LAND_SUCCESS
        if last_step_reward <= CRASH_TERMINAL_REWARD + 1.0:
            return EpisodeOutcome.CRASH
    if total_reward >= SOLVED_RETURN_THRESHOLD:
        return EpisodeOutcome.LAND_SUCCESS
    return EpisodeOutcome.CRASH


@torch.no_grad()
def deterministic_env_action(ac_model: nn.Module, observation) -> np.ndarray:
    state = obs_to_tensor(observation).unsqueeze(0)
    action = ac_model.deterministic_action(state)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


@dataclass(frozen=True)
class DeterministicEpisodeResult:
    total_reward: float
    episode_steps: int
    outcome: EpisodeOutcome
    solved: bool
    main_fuel_penalty: float
    side_fuel_penalty: float
    landing_abs_tilt: float | None
    landing_abs_angular_speed: float | None
    frames: tuple[np.ndarray, ...]
    fuel_at_frames: tuple[float, ...]
    fail_frame_index: int | None
    ended_by_termination: bool

    def _display_frame_index(self, time_index: int) -> int:
        last_frame_index = len(self.frames) - 1
        if self.fail_frame_index is not None:
            return min(time_index, self.fail_frame_index)
        return min(time_index, last_frame_index)

    def display_step_at(self, time_index: int) -> int:
        return self._display_frame_index(time_index)

    def display_fuel_at(self, time_index: int) -> float:
        if not self.fuel_at_frames:
            return 0.0
        frame_index = min(
            self._display_frame_index(time_index),
            len(self.fuel_at_frames) - 1,
        )
        return self.fuel_at_frames[frame_index]

    @property
    def total_fuel_penalty(self) -> float:
        return self.main_fuel_penalty + self.side_fuel_penalty


@dataclass(frozen=True)
class EvalMetrics:
    mean_episode_return: float
    success_rate: float
    land_success_rate: float
    crash_rate: float
    timeout_rate: float
    mean_main_fuel_on_land: float | None
    mean_side_fuel_on_land: float | None
    mean_abs_tilt_on_land: float | None
    mean_abs_angular_speed_on_land: float | None
    n_eval_episodes: int
    n_solved: int
    n_land_success: int

    @property
    def primary_score(self) -> tuple[float, float]:
        """best チェックポイント用: 成功率優先、同率なら平均リターン。"""
        return (self.success_rate, self.mean_episode_return)


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
    main_fuel = 0.0
    side_fuel = 0.0
    frames: list[np.ndarray] = []
    fuel_at_frames: list[float] = []
    if render:
        frames.append(env.render())
        fuel_at_frames.append(0.0)

    fail_frame_index: int | None = None
    ended_by_termination = False
    truncated = False
    last_step_reward = 0.0
    episode_steps = 0
    landing_abs_tilt: float | None = None
    landing_abs_angular_speed: float | None = None

    while True:
        action = deterministic_env_action(ac_model, observation)
        main_step, side_step, _ = fuel_penalty_for_continuous_action(action)
        main_fuel += main_step
        side_fuel += side_step

        observation, reward, terminated, step_truncated, _ = env.step(action)
        episode_steps += 1
        total_reward += reward
        last_step_reward = float(reward)
        if render:
            frames.append(env.render())
            fuel_at_frames.append(main_fuel + side_fuel)
        if terminated:
            ended_by_termination = True
            obs = np.asarray(observation, dtype=np.float32)
            landing_abs_tilt = float(abs(obs[4]))
            landing_abs_angular_speed = float(abs(obs[5]))
            if render and last_step_reward <= CRASH_TERMINAL_REWARD + 1.0:
                fail_frame_index = len(frames) - 1
            break
        if step_truncated:
            truncated = True
            break

    env.close()
    outcome = classify_episode(
        total_reward=total_reward,
        last_step_reward=last_step_reward,
        truncated=truncated,
        terminated=ended_by_termination,
    )
    solved = total_reward >= SOLVED_RETURN_THRESHOLD
    if outcome != EpisodeOutcome.LAND_SUCCESS:
        landing_abs_tilt = None
        landing_abs_angular_speed = None

    return DeterministicEpisodeResult(
        total_reward=total_reward,
        episode_steps=episode_steps,
        outcome=outcome,
        solved=solved,
        main_fuel_penalty=main_fuel,
        side_fuel_penalty=side_fuel,
        landing_abs_tilt=landing_abs_tilt,
        landing_abs_angular_speed=landing_abs_angular_speed,
        frames=tuple(frames if render else ()),
        fuel_at_frames=tuple(fuel_at_frames if render else ()),
        fail_frame_index=fail_frame_index,
        ended_by_termination=ended_by_termination,
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


@torch.no_grad()
def evaluate_policy(
    ac_model: nn.Module,
    experiment: ExperimentConfig,
) -> EvalMetrics:
    was_training = ac_model.training
    ac_model.eval()
    try:
        run = experiment.run
        n_episodes = run.eval_episodes
        returns: list[float] = []
        n_solved = 0
        n_land = 0
        n_crash = 0
        n_timeout = 0
        main_fuels_land: list[float] = []
        side_fuels_land: list[float] = []
        tilts_land: list[float] = []
        ang_vels_land: list[float] = []

        for episode in range(n_episodes):
            result = run_deterministic_episode(
                ac_model,
                experiment=experiment,
                seed=run.eval_rollout_seed(episode),
                render=False,
            )
            returns.append(result.total_reward)
            if result.solved:
                n_solved += 1
            if result.outcome == EpisodeOutcome.LAND_SUCCESS:
                n_land += 1
                main_fuels_land.append(result.main_fuel_penalty)
                side_fuels_land.append(result.side_fuel_penalty)
                if result.landing_abs_tilt is not None:
                    tilts_land.append(result.landing_abs_tilt)
                if result.landing_abs_angular_speed is not None:
                    ang_vels_land.append(result.landing_abs_angular_speed)
            elif result.outcome == EpisodeOutcome.CRASH:
                n_crash += 1
            else:
                n_timeout += 1

        inv_n = 1.0 / n_episodes
        return EvalMetrics(
            mean_episode_return=sum(returns) * inv_n,
            success_rate=n_solved * inv_n,
            land_success_rate=n_land * inv_n,
            crash_rate=n_crash * inv_n,
            timeout_rate=n_timeout * inv_n,
            mean_main_fuel_on_land=_mean(main_fuels_land),
            mean_side_fuel_on_land=_mean(side_fuels_land),
            mean_abs_tilt_on_land=_mean(tilts_land),
            mean_abs_angular_speed_on_land=_mean(ang_vels_land),
            n_eval_episodes=n_episodes,
            n_solved=n_solved,
            n_land_success=n_land,
        )
    finally:
        ac_model.train(was_training)