from __future__ import annotations

import gymnasium as gym
from gymnasium.core import Env

from deep_reinforcement_learning_lab.config.train_config import RewardMode, TrainConfig


def make_env(config: TrainConfig, *, render_mode: str | None = None) -> Env:
    """TrainConfig に従って Gymnasium 環境を生成する。"""
    make_kwargs: dict[str, object] = {}
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode

    reward_mode: RewardMode = config.reward_mode
    if reward_mode == "default":
        return gym.make(config.env_id, **make_kwargs)
    if reward_mode == "sutton_barto":
        if not config.env_id.startswith("CartPole"):
            raise ValueError(
                "sutton_barto reward is only supported for CartPole envs; "
                f"got env_id={config.env_id!r}"
            )
        return gym.make(config.env_id, sutton_barto_reward=True, **make_kwargs)
    raise ValueError(f"unsupported reward_mode: {reward_mode}")
