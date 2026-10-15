from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.algorithms.PPO.ppo import RolloutLosses
from deep_reinforcement_learning_lab.algorithms.PPO.rollout_buffer import (
    ContinuousFixedRolloutBuffer,
    VectorContinuousFixedRolloutBuffer,
)


class ContinuousPPO:
    """PPO for continuous actions (model provides sampling and log-prob evaluation)."""

    def __init__(
        self,
        ac_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout_buffer: ContinuousFixedRolloutBuffer | VectorContinuousFixedRolloutBuffer,
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

        action_dim = rollout_buffer.action_dim
        self._last_action = torch.zeros(action_dim)
        self._last_log_prob = 0.0
        self._last_value = 0.0
        self._gradient_step = 0
        self._rollout_step = 0

    @property
    def gradient_step(self) -> int:
        return self._gradient_step

    def select_actions(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = states.detach().cpu().float()
        with torch.no_grad():
            actions, log_probs, values = self.ac_model.sample_action(states)
        return actions, log_probs.view(-1), values.view(-1)

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
        if not isinstance(self.rollout_buffer, VectorContinuousFixedRolloutBuffer):
            raise TypeError("step_vector requires VectorContinuousFixedRolloutBuffer")

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

        self.finish_rollout_vector(next_states)
        return self.update()

    def finish_rollout_vector(self, last_states: torch.Tensor) -> None:
        if not isinstance(self.rollout_buffer, VectorContinuousFixedRolloutBuffer):
            raise TypeError("finish_rollout_vector requires VectorContinuousFixedRolloutBuffer")

        views = self.rollout_buffer.tensor_views_time_env()
        rewards = views["rewards"]
        bootstrap_dones = views["bootstrap_dones"]
        values_old = views["values_old"]

        last_states = last_states.detach().cpu().float()
        if last_states.dim() == 1:
            last_states = last_states.unsqueeze(0)

        with torch.no_grad():
            last_values = self._state_value(last_states)

        advantages = self._compute_gae_multi_env(
            rewards,
            bootstrap_dones,
            values_old,
            last_values,
        )
        values_flat = values_old.reshape(-1)
        returns = advantages + values_flat

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.rollout_buffer.set_advantages(advantages)
        self.rollout_buffer.set_returns(returns)

    def select_action(self, state: torch.Tensor) -> np.ndarray:
        batch_state = self._ensure_single_batch(state)
        with torch.no_grad():
            action, log_prob, value = self.ac_model.sample_action(batch_state)
            self._last_action = action.squeeze(0).detach().cpu()
            self._last_log_prob = float(log_prob.item())
            self._last_value = float(value.item())
        return self._last_action.numpy().astype(np.float32)

    def step(
        self,
        state: torch.Tensor,
        action: np.ndarray | torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        bootstrap_done: bool,
    ) -> RolloutLosses | None:
        if not isinstance(self.rollout_buffer, ContinuousFixedRolloutBuffer):
            raise TypeError("step() requires ContinuousFixedRolloutBuffer")

        action_tensor = torch.as_tensor(action, dtype=torch.float32).view(-1)
        if action_tensor.shape != self._last_action.shape or not torch.allclose(
            action_tensor, self._last_action, atol=1e-5, rtol=1e-5
        ):
            raise ValueError("action must match the last select_action() result")

        self.rollout_buffer.push_continuous(
            state,
            self._last_action,
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
        if not isinstance(self.rollout_buffer, ContinuousFixedRolloutBuffer):
            raise TypeError("finish_rollout() requires ContinuousFixedRolloutBuffer")

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
        log_probs, entropy = self.ac_model.evaluate_actions(states, actions)

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

    def _compute_gae_multi_env(
        self,
        rewards: torch.Tensor,
        bootstrap_dones: torch.Tensor,
        values_old: torch.Tensor,
        last_values: torch.Tensor,
    ) -> torch.Tensor:
        time_steps, n_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        for env_index in range(n_envs):
            rewards_env = rewards[:, env_index]
            dones_env = bootstrap_dones[:, env_index]
            values_env = values_old[:, env_index]
            next_values = torch.cat([values_env[1:], last_values[env_index : env_index + 1]])
            advantages[:, env_index] = self._compute_gae(
                rewards_env,
                dones_env,
                values_env,
                next_values,
            )
        return advantages.reshape(-1)

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

    def flush_if_pending(self, last_states: torch.Tensor) -> RolloutLosses | None:
        if len(self.rollout_buffer) == 0 or self.rollout_buffer.is_full():
            return None
        if isinstance(self.rollout_buffer, VectorContinuousFixedRolloutBuffer):
            self.finish_rollout_vector(last_states)
        else:
            self.finish_rollout(last_states)
        return self.update()

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)
