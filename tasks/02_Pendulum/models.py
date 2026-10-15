from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch.distributions import Normal

AcType = Literal["shared", "separate"]


def _build_mlp(
    in_dim: int,
    hidden_sizes: tuple[int, ...],
    out_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(dim, hidden))
        layers.append(nn.ReLU())
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class _TanhGaussianPolicyMixin:
    action_scale: torch.Tensor
    log_std: nn.Parameter

    def _distribution(self, states: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean = self._action_mean(states)
        std = self.log_std.exp().expand_as(mean)
        std = torch.clamp(std, min=1e-3)
        return Normal(mean, std), mean

    @staticmethod
    def _squash_action(
        pre_tanh: torch.Tensor,
        action_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tanh(pre_tanh) * action_scale

    def _log_prob_from_pre_tanh(
        self,
        pre_tanh: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        dist = Normal(mean, std)
        log_prob = dist.log_prob(pre_tanh).sum(dim=-1)
        log_prob -= torch.log(1.0 - torch.tanh(pre_tanh).pow(2) + 1e-6).sum(dim=-1)
        return log_prob

    def sample_action(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, mean = self._distribution(states)
        pre_tanh = dist.rsample()
        action = self._squash_action(pre_tanh, self.action_scale)
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, dist.stddev)
        value = self.state_value(states)
        return action, log_prob, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist, mean = self._distribution(states)
        pre_tanh = torch.atanh(torch.clamp(actions / self.action_scale, -0.999, 0.999))
        log_prob = self._log_prob_from_pre_tanh(pre_tanh, mean, dist.stddev)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def deterministic_action(self, states: torch.Tensor) -> torch.Tensor:
        _, mean = self._distribution(states)
        return self._squash_action(mean, self.action_scale)


class SharedActorCritic(nn.Module, _TanhGaussianPolicyMixin):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must not be empty")

        self.action_scale = (action_high - action_low) / 2.0
        self.backbone = _build_mlp(state_dim, hidden_sizes, hidden_sizes[-1])
        self.policy_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _features(self, states: torch.Tensor) -> torch.Tensor:
        return self.backbone(states)

    def _action_mean(self, states: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self._features(states))

    def state_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_head(self._features(states)).squeeze(-1)


class SeparateActorCritic(nn.Module, _TanhGaussianPolicyMixin):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ) -> None:
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must not be empty")

        self.action_scale = (action_high - action_low) / 2.0
        self.policy_net = _build_mlp(state_dim, hidden_sizes, action_dim)
        self.value_net = _build_mlp(state_dim, hidden_sizes, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _action_mean(self, states: torch.Tensor) -> torch.Tensor:
        return self.policy_net(states)

    def state_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_net(states).squeeze(-1)


def build_ac_model(
    ac_type: AcType,
    state_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...],
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> nn.Module:
    if ac_type == "shared":
        return SharedActorCritic(
            state_dim,
            action_dim,
            hidden_sizes,
            action_low,
            action_high,
        )
    if ac_type == "separate":
        return SeparateActorCritic(
            state_dim,
            action_dim,
            hidden_sizes,
            action_low,
            action_high,
        )
    raise ValueError(f"unsupported ac_type: {ac_type}")
