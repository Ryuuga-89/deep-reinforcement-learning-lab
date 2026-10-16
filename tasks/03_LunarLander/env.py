from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.vector import SyncVectorEnv

from deep_reinforcement_learning_lab.config import ExperimentConfig

LunarLanderSubtaskName = "default"

ENV_ID_CONTINUOUS = "LunarLanderContinuous-v3"
LUNAR_LANDER_SUBTASK = "default"
LUNAR_ENV_ACTION_DIM = 2

SubtaskBuilder = Callable[..., Env]


def _make_continuous_base(*, render_mode: str | None = None) -> Env:
    kwargs: dict[str, object] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(ENV_ID_CONTINUOUS, **kwargs)


def _make_default_subtask(*, render_mode: str | None = None) -> Env:
    return _make_continuous_base(render_mode=render_mode)


SUBTASK_BUILDERS: dict[str, SubtaskBuilder] = {
    LUNAR_LANDER_SUBTASK: _make_default_subtask,
}


def list_subtasks() -> tuple[str, ...]:
    return tuple(sorted(SUBTASK_BUILDERS))


def make_subtask_env(name: str, *, render_mode: str | None = None) -> Env:
    if name != LUNAR_LANDER_SUBTASK:
        raise ValueError(
            f"unsupported subtask: {name!r}. registered: {[LUNAR_LANDER_SUBTASK]}"
        )
    return _make_default_subtask(render_mode=render_mode)


def _ensure_lunar_subtask(experiment: ExperimentConfig) -> None:
    if experiment.subtask != LUNAR_LANDER_SUBTASK:
        raise ValueError(
            f"LunarLander expects subtask {LUNAR_LANDER_SUBTASK!r}, "
            f"got {experiment.subtask!r}"
        )


def make_env_for_experiment(
    experiment: ExperimentConfig,
    *,
    render_mode: str | None = None,
) -> Env:
    _ensure_lunar_subtask(experiment)
    return _make_continuous_base(render_mode=render_mode)


def make_sync_vector_env(
    experiment: ExperimentConfig,
    n_envs: int,
    *,
    seed: int,
) -> SyncVectorEnv:
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    _ensure_lunar_subtask(experiment)

    env_fns = [
        lambda exp=experiment: make_env_for_experiment(exp)
        for _ in range(n_envs)
    ]
    vec_env = SyncVectorEnv(env_fns)
    vec_env.reset(seed=seed)
    return vec_env


def lunar_render_fps() -> int:
    env = _make_continuous_base()
    fps = int(env.metadata.get("render_fps", 50))
    env.close()
    return fps
