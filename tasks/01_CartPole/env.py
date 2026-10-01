from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import gymnasium as gym
from gymnasium.core import Env

CartPoleSubtaskName = Literal["default", "sutton_barto"]

ENV_ID = "CartPole-v1"

SubtaskBuilder = Callable[..., Env]


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


SUBTASK_BUILDERS: dict[str, SubtaskBuilder] = {
    "default": _make_default,
    "sutton_barto": _make_sutton_barto,
}


def list_subtasks() -> tuple[str, ...]:
    return tuple(sorted(SUBTASK_BUILDERS))


def make_subtask_env(name: str, *, render_mode: str | None = None) -> Env:
    """サブタスク名に対応する Gymnasium 環境を生成する。"""
    builder = SUBTASK_BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"unsupported subtask: {name!r}. "
            f"registered: {sorted(SUBTASK_BUILDERS)}"
        )
    return builder(render_mode=render_mode)


def subtask_render_fps(name: str) -> int:
    env = make_subtask_env(name)
    fps = int(env.metadata.get("render_fps", 50))
    env.close()
    return fps
