from __future__ import annotations

import copy
import random
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_reinforcement_learning_lab.algorithms.DQN.replay_buffer import ReplayBuffer

LossType = Literal["mse", "huber"]


class DQN:
    """Deep Q-Network trainer (training loop only; environment-agnostic)."""

    def __init__(
        self,
        q_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer: ReplayBuffer,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_rate: float,
        batch_size: int,
        target_update_interval_step: int,
        tau: float | None = None,
        loss_type: LossType = "mse",
    ) -> None:
        self.q_model = q_model.cpu()
        self.target_q_model = copy.deepcopy(self.q_model).cpu()
        self.target_q_model.eval()
        for param in self.target_q_model.parameters():
            param.requires_grad = False

        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_rate = epsilon_rate
        self.batch_size = batch_size
        self.target_update_interval_step = target_update_interval_step
        self.tau = tau
        self.loss_type = loss_type

        self._env_step = 0
        self._gradient_step = 0

    @property
    def env_step(self) -> int:
        return self._env_step

    @property
    def gradient_step(self) -> int:
        return self._gradient_step

    def select_action(self, state: torch.Tensor) -> int:
        batch_state = self._ensure_single_batch(state)
        epsilon = max(
            self.epsilon_end,
            self.epsilon_start - self.epsilon_rate * self._env_step,
        )
        n_actions = self._num_actions(batch_state)
        if random.random() < epsilon:
            return random.randrange(n_actions)

        with torch.no_grad():
            q_values = self.q_model(batch_state)
        return int(q_values.argmax(dim=-1).item())

    def step(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)
        self._env_step += 1
        if len(self.replay_buffer) >= self.batch_size:
            self.update()

    def update(self) -> float:
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.float()
        next_states = next_states.float()
        rewards = rewards.float()
        dones = dones.float()

        q_values = self.q_model(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_q_model(next_states).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q

        loss = self._compute_loss(q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._gradient_step += 1
        if self.tau is not None:
            self._update_target_network()
        elif self._gradient_step % self.target_update_interval_step == 0:
            self._update_target_network()

        return float(loss.item())

    def _update_target_network(self) -> None:
        if self.tau is not None:
            for target_param, param in zip(
                self.target_q_model.parameters(),
                self.q_model.parameters(),
                strict=True,
            ):
                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
        else:
            self.target_q_model.load_state_dict(self.q_model.state_dict())

    def _compute_loss(self, q_sa: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "huber":
            return F.smooth_l1_loss(q_sa, targets)
        if self.loss_type == "mse":
            return F.mse_loss(q_sa, targets)
        raise ValueError(f"unsupported loss_type: {self.loss_type}")

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

    def _num_actions(self, batch_state: torch.Tensor) -> int:
        with torch.no_grad():
            return int(self.q_model(batch_state).shape[-1])
