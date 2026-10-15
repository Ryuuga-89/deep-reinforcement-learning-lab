from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.vector import SyncVectorEnv

from deep_reinforcement_learning_lab.config import ExperimentConfig

PendulumSubtaskName = Literal["default"]

ENV_ID = "Pendulum-v1"

SubtaskBuilder = Callable[..., Env]


def _make_default(*, render_mode: str | None = None) -> Env:
    kwargs: dict[str, object] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(ENV_ID, **kwargs)


SUBTASK_BUILDERS: dict[str, SubtaskBuilder] = {
    "default": _make_default,
}


def list_subtasks() -> tuple[str, ...]:
    return tuple(sorted(SUBTASK_BUILDERS))


def make_subtask_env(name: str, *, render_mode: str | None = None) -> Env:
    builder = SUBTASK_BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"unsupported subtask: {name!r}. "
            f"registered: {sorted(SUBTASK_BUILDERS)}"
        )
    return builder(render_mode=render_mode)


def make_env_for_experiment(
    experiment: ExperimentConfig,
    *,
    render_mode: str | None = None,
) -> Env:
    return make_subtask_env(experiment.subtask, render_mode=render_mode)


def make_sync_vector_env(
    experiment: ExperimentConfig,
    n_envs: int,
    *,
    seed: int,
) -> SyncVectorEnv:
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")

    env_fns = [
        lambda exp=experiment: make_env_for_experiment(exp) for _ in range(n_envs)
    ]
    vec_env = SyncVectorEnv(env_fns)
    vec_env.reset(seed=seed)
    return vec_env


def subtask_render_fps(name: str) -> int:
    env = make_subtask_env(name)
    fps = int(env.metadata.get("render_fps", 30))
    env.close()
    return fps
