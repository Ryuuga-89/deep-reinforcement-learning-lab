from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from deep_reinforcement_learning_lab.config import ExperimentConfig, PPOHyperparameters
from ac_model_registry import (
    build_registered_model,
    register_model,
    rollout_action_dim_for_model,
)
from env import LUNAR_ENV_ACTION_DIM
from factorized_heads import (
    FactorizedHeadConfig,
    decode_head_actions,
    parse_factorized_model_options,
)
from lunar_action_tables import resolve_action_table
from model_config import ModelConfig

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


class SharedDiscreteActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must not be empty")
        self.backbone = _build_mlp(state_dim, hidden_sizes, hidden_sizes[-1])
        self.policy_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def action_logits(self, states: torch.Tensor) -> torch.Tensor:
        features = self.backbone(states)
        return self.policy_head(features)

    def state_value(self, states: torch.Tensor) -> torch.Tensor:
        features = self.backbone(states)
        return self.value_head(features).squeeze(-1)

    def deterministic_action(self, states: torch.Tensor) -> torch.Tensor:
        logits = self.action_logits(states)
        return logits.argmax(dim=-1)


class SeparateDiscreteActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must not be empty")
        self.policy_net = _build_mlp(state_dim, hidden_sizes, action_dim)
        self.value_net = _build_mlp(state_dim, hidden_sizes, 1)

    def action_logits(self, states: torch.Tensor) -> torch.Tensor:
        return self.policy_net(states)

    def state_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_net(states).squeeze(-1)

    def deterministic_action(self, states: torch.Tensor) -> torch.Tensor:
        logits = self.action_logits(states)
        return logits.argmax(dim=-1)


class SharedFactorizedDiscreteActorCritic(nn.Module):
    """各 env 次元に独立した Categorical ヘッド。"""

    def __init__(
        self,
        state_dim: int,
        head_config: FactorizedHeadConfig,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must not be empty")
        self.head_config = head_config
        self.backbone = _build_mlp(state_dim, hidden_sizes, hidden_sizes[-1])
        self.policy_heads = nn.ModuleList(
            nn.Linear(hidden_sizes[-1], spec.num_choices)
            for spec in head_config.heads
        )
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def _head_logits_list(self, states: torch.Tensor) -> list[torch.Tensor]:
        features = self.backbone(states)
        return [head(features) for head in self.policy_heads]

    def state_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.backbone(states)).squeeze(-1)

    def sample_discrete_heads(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_list = self._head_logits_list(states)
        head_actions: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        for logits in logits_list:
            dist = Categorical(logits=logits)
            action = dist.sample()
            head_actions.append(action)
            log_probs.append(dist.log_prob(action))
        actions = torch.stack(head_actions, dim=-1)
        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
        value = self.state_value(states)
        return actions, log_prob, value

    def evaluate_discrete_heads(
        self,
        states: torch.Tensor,
        head_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits_list = self._head_logits_list(states)
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for h, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            log_probs.append(dist.log_prob(head_actions[:, h]))
            entropies.append(dist.entropy())
        log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
        entropy = torch.stack(entropies, dim=-1).sum(dim=-1)
        return log_prob, entropy

    def rollout_actions_to_env(self, head_actions: torch.Tensor) -> torch.Tensor:
        return decode_head_actions(head_actions, self.head_config)

    def deterministic_action(self, states: torch.Tensor) -> torch.Tensor:
        logits_list = self._head_logits_list(states)
        indices = torch.stack([logits.argmax(dim=-1) for logits in logits_list], dim=-1)
        return self.rollout_actions_to_env(indices)


class SeparateFactorizedDiscreteActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        head_config: FactorizedHeadConfig,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must not be empty")
        self.head_config = head_config
        last = hidden_sizes[-1]
        self.policy_backbone = _build_mlp(state_dim, hidden_sizes, last)
        self.policy_heads = nn.ModuleList(
            nn.Linear(last, spec.num_choices) for spec in head_config.heads
        )
        self.value_net = _build_mlp(state_dim, hidden_sizes, 1)

    def _head_logits_list(self, states: torch.Tensor) -> list[torch.Tensor]:
        features = self.policy_backbone(states)
        return [head(features) for head in self.policy_heads]

    def state_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_net(states).squeeze(-1)

    def sample_discrete_heads(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return SharedFactorizedDiscreteActorCritic.sample_discrete_heads(self, states)

    def evaluate_discrete_heads(
        self,
        states: torch.Tensor,
        head_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return SharedFactorizedDiscreteActorCritic.evaluate_discrete_heads(
            self, states, head_actions
        )

    def rollout_actions_to_env(self, head_actions: torch.Tensor) -> torch.Tensor:
        return decode_head_actions(head_actions, self.head_config)

    def deterministic_action(self, states: torch.Tensor) -> torch.Tensor:
        logits_list = self._head_logits_list(states)
        indices = torch.stack([logits.argmax(dim=-1) for logits in logits_list], dim=-1)
        return self.rollout_actions_to_env(indices)


def build_factorized_discrete_ac_model(
    ac_type: AcType,
    state_dim: int,
    head_config: FactorizedHeadConfig,
    hidden_sizes: tuple[int, ...],
) -> nn.Module:
    if ac_type == "shared":
        return SharedFactorizedDiscreteActorCritic(state_dim, head_config, hidden_sizes)
    if ac_type == "separate":
        return SeparateFactorizedDiscreteActorCritic(state_dim, head_config, hidden_sizes)
    raise ValueError(f"unsupported ac_type: {ac_type}")


class DiscreteLookupEnvPolicy(nn.Module):
    """離散 PPO 用。ロールアウト index → 連続 env 行動 (K, 2) テーブルで写像。"""

    def __init__(self, inner: nn.Module, action_table: np.ndarray) -> None:
        super().__init__()
        self.inner = inner
        if action_table.ndim != 2 or action_table.shape[1] != LUNAR_ENV_ACTION_DIM:
            raise ValueError(
                f"action_table must be (K, {LUNAR_ENV_ACTION_DIM}), got {action_table.shape}"
            )
        self.register_buffer(
            "_discrete_to_continuous",
            torch.tensor(action_table, dtype=torch.float32),
        )

    def action_logits(self, states: torch.Tensor) -> torch.Tensor:
        return self.inner.action_logits(states)

    def state_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.inner.state_value(states)

    def rollout_actions_to_env(self, rollout_actions: torch.Tensor) -> torch.Tensor:
        return self._discrete_to_continuous[rollout_actions.long()]

    def deterministic_action(self, states: torch.Tensor) -> torch.Tensor:
        indices = self.inner.deterministic_action(states)
        return self.rollout_actions_to_env(indices)


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

    def rollout_actions_to_env(self, rollout_actions: torch.Tensor) -> torch.Tensor:
        return rollout_actions


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

    def _action_mean(self, states: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self.backbone(states))

    def state_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.backbone(states)).squeeze(-1)


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


def build_discrete_ac_model(
    ac_type: AcType,
    state_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...],
) -> nn.Module:
    if ac_type == "shared":
        return SharedDiscreteActorCritic(state_dim, action_dim, hidden_sizes)
    if ac_type == "separate":
        return SeparateDiscreteActorCritic(state_dim, action_dim, hidden_sizes)
    raise ValueError(f"unsupported ac_type: {ac_type}")


def build_continuous_ac_model(
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


def _reject_continuous_options(model: ModelConfig) -> None:
    if model.options:
        raise ValueError(
            f"model {model.id!r} does not accept model.options "
            f"(got keys: {sorted(model.options)})"
        )


def _build_mlp_actor_critic_discrete(
    model: ModelConfig,
    _experiment: ExperimentConfig,
    state_dim: int,
    action_dim: int,
    ac_type: AcType,
    hidden_sizes: tuple[int, ...],
    action_low: torch.Tensor | None,
    action_high: torch.Tensor | None,
) -> nn.Module:
    table = resolve_action_table(model.options)
    expected_k = int(table.shape[0])
    if action_dim != expected_k:
        raise ValueError(
            f"discrete model expects rollout action_dim={expected_k} "
            f"(table {model.options.get('table', 'gymnasium_4')!r}), got {action_dim}"
        )
    inner = build_discrete_ac_model(ac_type, state_dim, action_dim, hidden_sizes)
    return DiscreteLookupEnvPolicy(inner, table)


def _build_mlp_actor_critic_continuous(
    model: ModelConfig,
    _experiment: ExperimentConfig,
    state_dim: int,
    action_dim: int,
    ac_type: AcType,
    hidden_sizes: tuple[int, ...],
    action_low: torch.Tensor | None,
    action_high: torch.Tensor | None,
) -> nn.Module:
    _reject_continuous_options(model)
    if action_dim != LUNAR_ENV_ACTION_DIM:
        raise ValueError(
            f"continuous model expects env action_dim={LUNAR_ENV_ACTION_DIM}, "
            f"got {action_dim}"
        )
    if action_low is None or action_high is None:
        raise ValueError("continuous model requires action_low and action_high")
    return build_continuous_ac_model(
        ac_type,
        state_dim,
        action_dim,
        hidden_sizes,
        action_low,
        action_high,
    )


register_model(
    "mlp_actor_critic_discrete",
    _build_mlp_actor_critic_discrete,
    ppo_kind="discrete",
)
register_model(
    "mlp_actor_critic_continuous",
    _build_mlp_actor_critic_continuous,
    ppo_kind="continuous",
)


def _build_mlp_factorized_discrete(
    model: ModelConfig,
    _experiment: ExperimentConfig,
    state_dim: int,
    action_dim: int,
    ac_type: AcType,
    hidden_sizes: tuple[int, ...],
    action_low: torch.Tensor | None,
    action_high: torch.Tensor | None,
) -> nn.Module:
    head_config = parse_factorized_model_options(model.options)
    expected = head_config.n_heads
    if action_dim != expected:
        raise ValueError(
            f"factorized model expects rollout action_dim={expected} (n_heads), "
            f"got {action_dim}"
        )
    return build_factorized_discrete_ac_model(
        ac_type, state_dim, head_config, hidden_sizes
    )


register_model(
    "mlp_factorized_discrete",
    _build_mlp_factorized_discrete,
    ppo_kind="factorized",
)


def build_model_for_experiment(
    experiment: ExperimentConfig,
    model: ModelConfig,
    state_dim: int,
    env_action_dim: int,
    *,
    action_low: torch.Tensor | None = None,
    action_high: torch.Tensor | None = None,
) -> nn.Module:
    hparams = experiment.hyperparameters
    if not isinstance(hparams, PPOHyperparameters):
        raise TypeError("build_model_for_experiment requires PPOHyperparameters")
    rollout_dim = rollout_action_dim_for_model(model, env_action_dim=env_action_dim)
    if env_action_dim != LUNAR_ENV_ACTION_DIM:
        raise ValueError(
            f"expected env action_dim {LUNAR_ENV_ACTION_DIM}, got {env_action_dim}"
        )
    return build_registered_model(
        model,
        experiment,
        state_dim,
        rollout_dim,
        hparams.ac_type,
        hparams.hidden_sizes,
        action_low=action_low,
        action_high=action_high,
    )
