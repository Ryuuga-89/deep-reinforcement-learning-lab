from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium.core import Env, ObsType
from gymnasium.spaces import Box

from deep_reinforcement_learning_lab.config import ExperimentConfig

CartPoleSubtaskName = Literal["default", "sutton_barto", "center", "move", "noise"]

ENV_ID = "CartPole-v1"

# CartPole-v1 default, random policy, 200 episodes (reset seed 42+ep): per-dim std of observations.
NOISE_OBS_REFERENCE_STD: tuple[float, float, float, float] = (
    0.11306557501388154,
    0.5649093754725865,
    0.09963853907810975,
    0.8305133733004123,
)


def noise_obs_reference_std_text() -> str:
    return ",".join(str(value) for value in NOISE_OBS_REFERENCE_STD)

SubtaskBuilder = Callable[..., Env]


class CenterRewardWrapper(gym.Wrapper):
    """生存 +1 に、カートが中央に近いほど大きいボーナスを足す。"""

    def __init__(self, env: Env, *, center_bonus_coef: float = 1.0) -> None:
        super().__init__(env)
        if center_bonus_coef < 0.0:
            raise ValueError("center_bonus_coef must be non-negative")
        self.center_bonus_coef = center_bonus_coef

    def step(self, action: Any) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        observation, _reward, terminated, truncated, info = self.env.step(action)
        x = float(observation[0])
        threshold = float(self.env.unwrapped.x_threshold)
        center_bonus = 1.0 - min(abs(x) / threshold, 1.0)
        reward = 1.0 + self.center_bonus_coef * center_bonus
        return observation, reward, terminated, truncated, info


class MoveTargetRewardWrapper(gym.Wrapper):
    """生存 +1 に、移動する目標位置へ近いほど大きいボーナスを足す。

    目標の正規化座標はエピソード内ステップ ``t``（``step`` 呼び出し回数）で切り替わる。
    ``t < 3 * beta`` は 0.25。以降は ``3β, 5β, 7β, ...`` で 0.75 と 0.25 を交互に跳ぶ。
    0.25 / 0.75 は軌道 ``[-x_threshold, x_threshold]`` 上の位置に対応する。
    """

    def __init__(self, env: Env, *, move_alpha: float = 1.0, move_beta: int = 100) -> None:
        super().__init__(env)
        if move_alpha < 0.0:
            raise ValueError("move_alpha must be non-negative")
        if isinstance(move_beta, bool) or not isinstance(move_beta, int):
            raise ValueError("move_beta must be an integer")
        if not 1 <= move_beta <= 500:
            raise ValueError("move_beta must be an integer in 1..500")
        self.move_alpha = move_alpha
        self.move_beta = move_beta
        self._t = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self._t = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        observation, _reward, terminated, truncated, info = self.env.step(action)
        self._t += 1
        x = float(observation[0])
        threshold = float(self.env.unwrapped.x_threshold)
        target_x = (2.0 * self._target_norm() - 1.0) * threshold
        distance = min(abs(x - target_x) / threshold, 1.0)
        reward = 1.0 + self.move_alpha * (1.0 - distance)
        return observation, reward, terminated, truncated, info

    def _target_norm(self) -> float:
        beta = self.move_beta
        if self._t < 3 * beta:
            return 0.25
        switches = 1 + (self._t - 3 * beta) // (2 * beta)
        if switches % 2 == 1:
            return 0.75
        return 0.25


class ObservationNoiseWrapper(gym.ObservationWrapper):
    """エージェント向け観測に成分独立のガウスノイズを加える（物理状態は変更しない）。"""

    def __init__(self, env: Env, *, noise_rho: float = 0.1) -> None:
        super().__init__(env)
        if noise_rho < 0.0:
            raise ValueError("noise_rho must be non-negative")
        self.noise_rho = noise_rho
        self._sigmas = noise_rho * np.asarray(NOISE_OBS_REFERENCE_STD, dtype=np.float32)
        space = self.observation_space
        if not isinstance(space, Box):
            raise TypeError("ObservationNoiseWrapper requires a Box observation_space")
        self._low = np.asarray(space.low, dtype=np.float32)
        self._high = np.asarray(space.high, dtype=np.float32)

    def observation(self, observation: ObsType) -> ObsType:
        obs = np.asarray(observation, dtype=np.float32)
        if self.noise_rho == 0.0:
            return obs
        noise = self.np_random.normal(
            loc=0.0,
            scale=self._sigmas,
            size=obs.shape,
        ).astype(np.float32)
        return np.clip(obs + noise, self._low, self._high)


def _make_default(*, render_mode: str | None = None) -> Env:
    kwargs: dict[str, object] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(ENV_ID, **kwargs)


def _make_sutton_barto(*, render_mode: str | None = None) -> Env:
    kwargs: dict[str, object] = {"sutton_barto_reward": True}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(ENV_ID, **kwargs)


def _make_center(
    *,
    render_mode: str | None = None,
    center_bonus_coef: float = 1.0,
) -> Env:
    kwargs: dict[str, object] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return CenterRewardWrapper(
        gym.make(ENV_ID, **kwargs),
        center_bonus_coef=center_bonus_coef,
    )


def _make_move(
    *,
    render_mode: str | None = None,
    move_alpha: float = 1.0,
    move_beta: int = 100,
) -> Env:
    kwargs: dict[str, object] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return MoveTargetRewardWrapper(
        gym.make(ENV_ID, **kwargs),
        move_alpha=move_alpha,
        move_beta=move_beta,
    )


def _make_noise(
    *,
    render_mode: str | None = None,
    noise_rho: float = 0.1,
) -> Env:
    kwargs: dict[str, object] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return ObservationNoiseWrapper(gym.make(ENV_ID, **kwargs), noise_rho=noise_rho)


SUBTASK_BUILDERS: dict[str, SubtaskBuilder] = {
    "default": _make_default,
    "sutton_barto": _make_sutton_barto,
    "center": _make_center,
    "move": _make_move,
    "noise": _make_noise,
}


def list_subtasks() -> tuple[str, ...]:
    return tuple(sorted(SUBTASK_BUILDERS))


def make_subtask_env(
    name: str,
    *,
    render_mode: str | None = None,
    center_bonus_coef: float = 1.0,
    move_alpha: float = 1.0,
    move_beta: int = 100,
    noise_rho: float = 0.1,
) -> Env:
    """サブタスク名に対応する Gymnasium 環境を生成する。"""
    builder = SUBTASK_BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"unsupported subtask: {name!r}. "
            f"registered: {sorted(SUBTASK_BUILDERS)}"
        )
    if name == "center":
        return builder(
            render_mode=render_mode,
            center_bonus_coef=center_bonus_coef,
        )
    if name == "move":
        return builder(
            render_mode=render_mode,
            move_alpha=move_alpha,
            move_beta=move_beta,
        )
    if name == "noise":
        return builder(render_mode=render_mode, noise_rho=noise_rho)
    return builder(render_mode=render_mode)


def make_env_for_experiment(
    experiment: ExperimentConfig,
    *,
    render_mode: str | None = None,
) -> Env:
    """実験 YAML の ``subtask_options`` を反映して環境を生成する。"""
    kwargs: dict[str, object] = {}
    if experiment.subtask == "center":
        coef = experiment.subtask_options.center_bonus_coef
        if coef is None:
            raise ValueError("center subtask requires center_bonus_coef in subtask_options")
        kwargs["center_bonus_coef"] = coef
    elif experiment.subtask == "move":
        alpha = experiment.subtask_options.move_alpha
        beta = experiment.subtask_options.move_beta
        if alpha is None or beta is None:
            raise ValueError("move subtask requires move_alpha and move_beta in subtask_options")
        kwargs["move_alpha"] = alpha
        kwargs["move_beta"] = beta
    elif experiment.subtask == "noise":
        rho = experiment.subtask_options.noise_rho
        if rho is None:
            raise ValueError("noise subtask requires noise_rho in subtask_options")
        kwargs["noise_rho"] = rho
    return make_subtask_env(experiment.subtask, render_mode=render_mode, **kwargs)


def subtask_render_fps(name: str) -> int:
    env = make_subtask_env(name)
    fps = int(env.metadata.get("render_fps", 50))
    env.close()
    return fps
