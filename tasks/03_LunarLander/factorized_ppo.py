"""独立離散ヘッド用 PPO（各次元ごとに Categorical、log_prob は和）。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class RolloutLosses:
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total: float


class VectorFactorizedRolloutBuffer:
    def __init__(self, rollout_steps: int, n_envs: int, state_dim: int, n_heads: int) -> None:
        if rollout_steps <= 0 or n_envs <= 0 or n_heads <= 0:
            raise ValueError("rollout_steps, n_envs, n_heads must be positive")
        self._rollout_steps = rollout_steps
        self._n_envs = n_envs
        self._n_heads = n_heads
        self._state_dim = state_dim
        capacity = rollout_steps * n_envs
        self._capacity = capacity
        self._vector_steps = 0
        self._states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self._actions = torch.zeros((capacity, n_heads), dtype=torch.long)
        self._log_probs_old = torch.zeros(capacity, dtype=torch.float32)
        self._rewards = torch.zeros(capacity, dtype=torch.float32)
        self._bootstrap_dones = torch.zeros(capacity, dtype=torch.float32)
        self._values_old = torch.zeros(capacity, dtype=torch.float32)
        self._advantages = torch.zeros(capacity, dtype=torch.float32)
        self._returns = torch.zeros(capacity, dtype=torch.float32)

    def push_vector_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap_dones: torch.Tensor,
        values_old: torch.Tensor,
    ) -> None:
        if self.is_full():
            raise RuntimeError("rollout buffer is full")
        states = states.detach().cpu().float()
        actions = actions.detach().cpu().long()
        log_probs = log_probs.detach().cpu().float().view(-1)
        rewards = rewards.detach().cpu().float().view(-1)
        bootstrap_dones = bootstrap_dones.detach().cpu().float().view(-1)
        values_old = values_old.detach().cpu().float().view(-1)
        if actions.shape != (self._n_envs, self._n_heads):
            raise ValueError(
                f"actions must be ({self._n_envs}, {self._n_heads}), got {tuple(actions.shape)}"
            )
        start = self._vector_steps * self._n_envs
        end = start + self._n_envs
        self._states[start:end] = states
        self._actions[start:end] = actions
        self._log_probs_old[start:end] = log_probs
        self._rewards[start:end] = rewards
        self._bootstrap_dones[start:end] = bootstrap_dones
        self._values_old[start:end] = values_old
        self._vector_steps += 1

    def is_full(self) -> bool:
        return self._vector_steps >= self._rollout_steps

    def clear(self) -> None:
        self._vector_steps = 0

    def set_advantages(self, advantages: torch.Tensor) -> None:
        self._advantages = advantages.detach().cpu().float().reshape(-1)

    def set_returns(self, returns: torch.Tensor) -> None:
        self._returns = returns.detach().cpu().float().reshape(-1)

    def tensor_views_time_env(self) -> dict[str, torch.Tensor]:
        t, e = self._vector_steps, self._n_envs
        return {
            "rewards": self._rewards[: t * e].view(t, e),
            "bootstrap_dones": self._bootstrap_dones[: t * e].view(t, e),
            "values_old": self._values_old[: t * e].view(t, e),
        }

    def iter_minibatches(
        self,
        minibatch_size: int,
        *,
        shuffle: bool,
    ) -> list[tuple[torch.Tensor, ...]]:
        n = self._vector_steps * self._n_envs
        indices = torch.arange(n)
        if shuffle:
            indices = indices[torch.randperm(n)]
        batches: list[tuple[torch.Tensor, ...]] = []
        for start in range(0, n, minibatch_size):
            idx = indices[start : start + minibatch_size]
            batches.append(
                (
                    self._states[idx],
                    self._actions[idx],
                    self._log_probs_old[idx],
                    self._advantages[idx],
                    self._returns[idx],
                    self._values_old[idx],
                )
            )
        return batches


class FactorizedPPO:
    def __init__(
        self,
        ac_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout_buffer: VectorFactorizedRolloutBuffer,
        *,
        gamma: float,
        gae_lambda: float,
        clip_coef: float,
        entropy_coef: float,
        value_coef: float,
        rollout_steps: int,
        n_epochs: int,
        minibatch_size: int,
        clip_value_coef: float | None = None,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.ac_model = ac_model.cpu()
        self.optimizer = optimizer
        self.rollout_buffer = rollout_buffer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.clip_value_coef = clip_coef if clip_value_coef is None else clip_value_coef
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.rollout_steps = rollout_steps
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self._gradient_step = 0

    @property
    def gradient_step(self) -> int:
        return self._gradient_step

    def select_actions(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = states.detach().cpu().float()
        with torch.no_grad():
            head_actions, log_probs, values = self.ac_model.sample_discrete_heads(states)
        return head_actions, log_probs, values

    def step_vector(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap_dones: torch.Tensor,
        values_old: torch.Tensor,
        next_states: torch.Tensor,
    ) -> RolloutLosses | None:
        self.rollout_buffer.push_vector_step(
            states,
            actions,
            log_probs,
            rewards,
            bootstrap_dones,
            values_old,
        )
        if not self.rollout_buffer.is_full():
            return None
        self._finish_rollout(next_states)
        return self._update()

    def flush_if_pending(self, last_states: torch.Tensor) -> RolloutLosses | None:
        if self.rollout_buffer.is_full() or self.rollout_buffer._vector_steps == 0:
            return None
        self._finish_rollout(last_states)
        return self.update()

    def update(self) -> RolloutLosses:
        return self._update()

    def _finish_rollout(self, last_states: torch.Tensor) -> None:
        views = self.rollout_buffer.tensor_views_time_env()
        rewards = views["rewards"]
        bootstrap_dones = views["bootstrap_dones"]
        values_old = views["values_old"]
        last_states = last_states.detach().cpu().float()
        with torch.no_grad():
            last_values = self.ac_model.state_value(last_states)
        t_steps, n_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(n_envs)
        for step in reversed(range(t_steps)):
            mask = 1.0 - bootstrap_dones[step]
            next_v = last_values if step == t_steps - 1 else values_old[step + 1]
            delta = rewards[step] + self.gamma * next_v * mask - values_old[step]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[step] = gae
        adv_flat = advantages.reshape(-1)
        ret_flat = adv_flat + values_old.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        self.rollout_buffer.set_advantages(adv_flat)
        self.rollout_buffer.set_returns(ret_flat)

    def _update(self) -> RolloutLosses:
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        total_losses: list[float] = []
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.iter_minibatches(
                self.minibatch_size, shuffle=True
            ):
                states, actions, log_probs_old, advantages, returns, values_old = batch
                log_probs, entropy = self.ac_model.evaluate_discrete_heads(states, actions)
                values = self.ac_model.state_value(states)
                ratio = torch.exp(log_probs - log_probs_old)
                clipped = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
                clipped_values = values_old + torch.clamp(
                    values - values_old, -self.clip_value_coef, self.clip_value_coef
                )
                value_loss = 0.5 * torch.max(
                    (values - returns).pow(2),
                    (clipped_values - returns).pow(2),
                ).mean()
                entropy_loss = -entropy.mean()
                total = (
                    policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                )
                self.optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(self.ac_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self._gradient_step += 1
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_losses.append(float(entropy_loss.item()))
                total_losses.append(float(total.item()))
        self.rollout_buffer.clear()
        return RolloutLosses(
            policy_loss=sum(policy_losses) / len(policy_losses),
            value_loss=sum(value_losses) / len(value_losses),
            entropy_loss=sum(entropy_losses) / len(entropy_losses),
            total=sum(total_losses) / len(total_losses),
        )
