from __future__ import annotations

import torch

from deep_reinforcement_learning_lab.algorithms.DQN.dqn import DQN


class DDQN(DQN):
    """Double DQN: decouple action selection and evaluation in the bootstrap target."""

    def _compute_bootstrap_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_actions = self.q_model(next_states).argmax(dim=1, keepdim=True)
            next_q = (
                self.target_q_model(next_states)
                .gather(1, next_actions)
                .squeeze(1)
            )
            return rewards + self.gamma * (1.0 - dones) * next_q
