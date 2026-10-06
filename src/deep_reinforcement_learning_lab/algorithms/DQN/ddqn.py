from __future__ import annotations

import torch

from deep_reinforcement_learning_lab.algorithms.DQN.dqn import DQN


class DDQN(DQN):
    """Double DQN: decouple action selection and evaluation in the bootstrap target."""

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
            next_actions = self.q_model(next_states).argmax(dim=1, keepdim=True)
            next_q = (
                self.target_q_model(next_states)
                .gather(1, next_actions)
                .squeeze(1)
            )
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
