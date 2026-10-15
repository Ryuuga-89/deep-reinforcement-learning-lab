from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

from deep_reinforcement_learning_lab.algorithms.PPO.rollout_buffer import RolloutBuffer


@dataclass(frozen=True)
class RolloutLosses:
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total: float


class PPO:
    """Proximal Policy Optimization (clipped policy + clipped value loss)."""

    def __init__(
        self,
        ac_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout_buffer: RolloutBuffer,
        gamma: float,
        gae_lambda: float,
        clip_coef: float,
        entropy_coef: float,
        value_coef: float,
        rollout_steps: int,
        n_epochs: int,
        minibatch_size: int,
        *,
        clip_value_coef: float | None = None,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,
    ) -> None:
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if clip_coef <= 0.0:
            raise ValueError("clip_coef must be positive")
        if rollout_steps <= 0 or n_epochs <= 0 or minibatch_size <= 0:
            raise ValueError("rollout_steps, n_epochs, and minibatch_size must be positive")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")

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
        self.normalize_advantage = normalize_advantage

        self._last_action = -1
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._gradient_step = 0
        self._rollout_step = 0

    @property
    def gradient_step(self) -> int:
        return self._gradient_step

    def select_action(self, state: torch.Tensor) -> int:
        batch_state = self._ensure_single_batch(state)
        with torch.no_grad():
            logits = self._action_logits(batch_state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            self._last_action = int(action.item())
            self._last_log_prob = float(dist.log_prob(action).item())
            self._last_value = float(self._state_value(batch_state).item())
        return self._last_action

    def step(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        bootstrap_done: bool,
    ) -> RolloutLosses | None:
        if action != self._last_action:
            raise ValueError("action must match the last select_action() result")

        self.rollout_buffer.push(
            state,
            action,
            self._last_log_prob,
            reward,
            bootstrap_done,
            self._last_value,
        )
        self._rollout_step += 1

        if not self.rollout_buffer.is_full():
            return None

        self.finish_rollout(next_state)
        return self.update()

    def finish_rollout(self, last_state: torch.Tensor) -> None:
        views = self.rollout_buffer.tensor_views()
        rewards = views["rewards"]
        bootstrap_dones = views["bootstrap_dones"]
        values_old = views["values_old"]

        with torch.no_grad():
            last_value = self._state_value(self._ensure_single_batch(last_state))
            next_values = torch.cat([values_old[1:], last_value], dim=0)

        advantages = self._compute_gae(rewards, bootstrap_dones, values_old, next_values)
        returns = advantages + values_old

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.rollout_buffer.set_advantages(advantages)
        self.rollout_buffer.set_returns(returns)

    def update(self) -> RolloutLosses:
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        total_losses: list[float] = []

        for _ in range(self.n_epochs):
            batches = self.rollout_buffer.iter_minibatches(
                self.minibatch_size,
                shuffle=True,
            )
            for batch in batches:
                states, actions, log_probs_old, advantages, returns, values_old = batch
                losses = self._optimization_step(
                    states,
                    actions,
                    log_probs_old,
                    advantages,
                    returns,
                    values_old,
                )
                policy_losses.append(losses.policy_loss)
                value_losses.append(losses.value_loss)
                entropy_losses.append(losses.entropy_loss)
                total_losses.append(losses.total)

        self.rollout_buffer.clear()
        self._rollout_step = 0

        return RolloutLosses(
            policy_loss=self._mean(policy_losses),
            value_loss=self._mean(value_losses),
            entropy_loss=self._mean(entropy_losses),
            total=self._mean(total_losses),
        )

    def _optimization_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values_old: torch.Tensor,
    ) -> RolloutLosses:
        logits = self._action_logits(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        policy_loss = self._policy_loss(log_probs, log_probs_old, advantages)
        values = self._state_value(states)
        value_loss = self._value_loss(values, values_old, returns)
        entropy_loss = -entropy.mean()

        total = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.ac_model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._gradient_step += 1

        return RolloutLosses(
            policy_loss=float(policy_loss.item()),
            value_loss=float(value_loss.item()),
            entropy_loss=float(entropy_loss.item()),
            total=float(total.item()),
        )

    def _policy_loss(
        self,
        log_probs: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        ratio = torch.exp(log_probs - log_probs_old)
        clipped = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
        surrogate = ratio * advantages
        clipped_surrogate = clipped * advantages
        return -torch.min(surrogate, clipped_surrogate).mean()

    def _value_loss(
        self,
        values: torch.Tensor,
        values_old: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        clipped_values = values_old + torch.clamp(
            values - values_old,
            -self.clip_value_coef,
            self.clip_value_coef,
        )
        loss_unclipped = (values - returns).pow(2)
        loss_clipped = (clipped_values - returns).pow(2)
        return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        bootstrap_dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
    ) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(rewards.shape[0])):
            mask = 1.0 - bootstrap_dones[step]
            delta = rewards[step] + self.gamma * next_values[step] * mask - values[step]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[step] = gae
        return advantages

    def _action_logits(self, states: torch.Tensor) -> torch.Tensor:
        return self.ac_model.action_logits(states)

    def _state_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.ac_model.state_value(states).view(-1)

    @staticmethod
    def _ensure_single_batch(state: torch.Tensor) -> torch.Tensor:
        state = state.detach().cpu().float()
        if state.dim() == 1:
            return state.unsqueeze(0)
        if state.dim() >= 2 and state.shape[0] == 1:
            return state
        raise ValueError(
            "select_action expects state shape [*state_shape] or [1, *state_shape]"
        )

    def flush_if_pending(self, last_state: torch.Tensor) -> RolloutLosses | None:
        """GAE + update for a trailing partial rollout (e.g. end of training)."""
        if len(self.rollout_buffer) == 0 or self.rollout_buffer.is_full():
            return None
        self.finish_rollout(last_state)
        return self.update()

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)
