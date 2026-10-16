from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_args

import torch
import torch.nn as nn

from factorized_heads import num_factorized_rollout_dims, parse_factorized_model_options
from lunar_action_tables import num_rollout_actions
from model_config import ModelConfig

PpoKind = Literal["discrete", "continuous", "factorized"]
_PPO_KINDS = frozenset(get_args(PpoKind))

ModelBuilder = Callable[..., nn.Module]

_REGISTRY: dict[str, ModelBuilder] = {}
_PPO_KIND_BY_ID: dict[str, PpoKind] = {}


@dataclass(frozen=True)
class RolloutSpec:
    """PPO ロールアウトバッファ上の行動表現（環境の Box(2) とは別）。"""

    ppo_kind: PpoKind
    stored_action_dim: int


def register_model(model_id: str, builder: ModelBuilder, *, ppo_kind: PpoKind) -> None:
    if model_id in _REGISTRY:
        raise ValueError(f"model id already registered: {model_id}")
    if ppo_kind not in _PPO_KINDS:
        raise ValueError(f"invalid ppo_kind: {ppo_kind!r}")
    _REGISTRY[model_id] = builder
    _PPO_KIND_BY_ID[model_id] = ppo_kind


def ppo_kind_for_model(model: ModelConfig) -> PpoKind:
    return resolve_rollout_spec(model, env_action_dim=2).ppo_kind


def resolve_rollout_spec(model: ModelConfig, *, env_action_dim: int) -> RolloutSpec:
    kind = _PPO_KIND_BY_ID.get(model.id)
    if kind is None:
        registered = sorted(_REGISTRY)
        raise ValueError(
            f"unknown model.id {model.id!r} (no ppo_kind). registered: {registered}"
        )
    if kind == "continuous":
        return RolloutSpec(ppo_kind="continuous", stored_action_dim=env_action_dim)
    if kind == "factorized":
        cfg = parse_factorized_model_options(model.options)
        return RolloutSpec(
            ppo_kind="factorized",
            stored_action_dim=num_factorized_rollout_dims(cfg),
        )
    stored = num_rollout_actions(model.options)
    return RolloutSpec(ppo_kind="discrete", stored_action_dim=stored)


def rollout_action_dim_for_model(model: ModelConfig, *, env_action_dim: int) -> int:
    return resolve_rollout_spec(model, env_action_dim=env_action_dim).stored_action_dim


def build_registered_model(
    model: ModelConfig,
    experiment: Any,
    state_dim: int,
    action_dim: int,
    ac_type: str,
    hidden_sizes: tuple[int, ...],
    *,
    action_low: torch.Tensor | None = None,
    action_high: torch.Tensor | None = None,
) -> nn.Module:
    builder = _REGISTRY.get(model.id)
    if builder is None:
        registered = sorted(_REGISTRY)
        raise ValueError(
            f"unsupported model.id: {model.id!r}. registered: {registered}"
        )
    return builder(
        model,
        experiment,
        state_dim,
        action_dim,
        ac_type,
        hidden_sizes,
        action_low,
        action_high,
    )


def registered_model_ids() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def resolve_legacy_model_id(model_id: str, policy_kind: str) -> str:
    """旧 ckpt: model_id=mlp_actor_critic + policy_kind → 現行 id。"""
    if model_id != "mlp_actor_critic":
        return model_id
    if policy_kind == "discrete":
        return "mlp_actor_critic_discrete"
    if policy_kind == "continuous":
        return "mlp_actor_critic_continuous"
    raise ValueError(f"unsupported legacy policy_kind: {policy_kind!r}")
