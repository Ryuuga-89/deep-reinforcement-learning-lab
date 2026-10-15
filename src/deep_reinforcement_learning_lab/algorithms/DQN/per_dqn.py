from __future__ import annotations

import torch
import torch.nn as nn

from deep_reinforcement_learning_lab.algorithms.DQN.dqn import DQN, LossType
from deep_reinforcement_learning_lab.algorithms.DQN.ddqn import DDQN
from deep_reinforcement_learning_lab.algorithms.DQN.prioritized_replay_buffer import (
    PrioritizedReplayBuffer,
)


class PER_DQN(DQN):
    """DQN with prioritized experience replay."""

    def __init__(
        self,
        q_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer: PrioritizedReplayBuffer,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_rate: float,
        batch_size: int,
        *,
        per_beta_start: float,
        per_beta_end: float,
        beta_anneal_steps: int,
        priority_epsilon: float,
        target_update_interval_step: int | None = None,
        tau: float | None = None,
        loss_type: LossType = "mse",
    ) -> None:
        super().__init__(
            q_model=q_model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_rate=epsilon_rate,
            batch_size=batch_size,
            target_update_interval_step=target_update_interval_step,
            tau=tau,
            loss_type=loss_type,
        )
        if beta_anneal_steps <= 0:
            raise ValueError("beta_anneal_steps must be a positive integer")
        if priority_epsilon < 0.0:
            raise ValueError("priority_epsilon must be non-negative")
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.priority_epsilon = priority_epsilon

    @property
    def per_beta(self) -> float:
        return self._current_beta()

    def _current_beta(self) -> float:
        step = min(self._gradient_step, self.beta_anneal_steps)
        fraction = step / self.beta_anneal_steps
        return self.per_beta_start + (self.per_beta_end - self.per_beta_start) * fraction

    def update(self) -> float:
        buffer = self.replay_buffer
        if not isinstance(buffer, PrioritizedReplayBuffer):
            raise TypeError("PER_DQN requires a PrioritizedReplayBuffer")

        beta = self._current_beta()
        states, actions, rewards, next_states, dones, indices, weights = (
            buffer.sample_prioritized(self.batch_size, beta)
        )
        states = states.float()
        next_states = next_states.float()
        rewards = rewards.float()
        dones = dones.float()

        q_values = self.q_model(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        targets = self._compute_bootstrap_targets(rewards, next_states, dones)

        elementwise_loss = self._compute_elementwise_loss(q_sa, targets)
        loss = (elementwise_loss * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        buffer.update_priorities(indices, targets - q_sa, self.priority_epsilon)

        self._gradient_step += 1
        if self.tau is not None:
            self._update_target_network()
        elif self._gradient_step % self.target_update_interval_step == 0:
            self._update_target_network()

        return float(loss.item())


class PER_DDQN(DDQN, PER_DQN):
    """Double DQN with prioritized experience replay."""
