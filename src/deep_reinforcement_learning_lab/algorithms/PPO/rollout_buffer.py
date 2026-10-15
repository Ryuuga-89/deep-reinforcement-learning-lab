from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class RolloutBuffer(ABC):
    """Abstract on-policy rollout buffer for one policy snapshot."""

    def __init__(self, rollout_steps: int) -> None:
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        self._rollout_steps = rollout_steps

    @property
    def rollout_steps(self) -> int:
        return self._rollout_steps

    @abstractmethod
    def push(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        bootstrap_done: bool,
        value_old: float,
    ) -> None:
        ...

    @abstractmethod
    def is_full(self) -> bool:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def set_advantages(self, advantages: torch.Tensor) -> None:
        ...

    @abstractmethod
    def set_returns(self, returns: torch.Tensor) -> None:
        ...

    @abstractmethod
    def tensor_views(self) -> dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def iter_minibatches(
        self,
        minibatch_size: int,
        *,
        shuffle: bool,
    ) -> list[tuple[torch.Tensor, ...]]:
        ...


class FixedRolloutBuffer(RolloutBuffer):
    """Stores exactly ``rollout_steps`` transitions, then yields flat minibatches."""

    def __init__(self, rollout_steps: int, state_dim: int) -> None:
        super().__init__(rollout_steps)
        self._state_dim = state_dim
        self._size = 0
        self._states = torch.zeros((rollout_steps, state_dim), dtype=torch.float32)
        self._actions = torch.zeros(rollout_steps, dtype=torch.long)
        self._log_probs_old = torch.zeros(rollout_steps, dtype=torch.float32)
        self._rewards = torch.zeros(rollout_steps, dtype=torch.float32)
        self._bootstrap_dones = torch.zeros(rollout_steps, dtype=torch.float32)
        self._values_old = torch.zeros(rollout_steps, dtype=torch.float32)
        self._advantages = torch.zeros(rollout_steps, dtype=torch.float32)
        self._returns = torch.zeros(rollout_steps, dtype=torch.float32)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        bootstrap_done: bool,
        value_old: float,
    ) -> None:
        if self.is_full():
            raise RuntimeError("rollout buffer is full; call clear() after update()")

        index = self._size
        self._states[index] = self._flatten_state(state)
        self._actions[index] = action
        self._log_probs_old[index] = log_prob
        self._rewards[index] = reward
        self._bootstrap_dones[index] = float(bootstrap_done)
        self._values_old[index] = value_old
        self._size += 1

    def is_full(self) -> bool:
        return self._size >= self._rollout_steps

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._size = 0

    def set_advantages(self, advantages: torch.Tensor) -> None:
        if advantages.shape != (self._size,):
            raise ValueError("advantages must have shape (len(buffer),)")
        self._advantages[: self._size] = advantages

    def set_returns(self, returns: torch.Tensor) -> None:
        if returns.shape != (self._size,):
            raise ValueError("returns must have shape (len(buffer),)")
        self._returns[: self._size] = returns

    def tensor_views(self) -> dict[str, torch.Tensor]:
        """Tensors for the collected prefix (length ``len(buffer)``)."""
        end = self._size
        return {
            "rewards": self._rewards[:end],
            "bootstrap_dones": self._bootstrap_dones[:end],
            "values_old": self._values_old[:end],
        }

    def iter_minibatches(
        self,
        minibatch_size: int,
        *,
        shuffle: bool,
    ) -> list[tuple[torch.Tensor, ...]]:
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if self._size == 0:
            raise RuntimeError("buffer is empty")

        indices = (
            torch.randperm(self._size) if shuffle else torch.arange(self._size)
        )
        batches: list[tuple[torch.Tensor, ...]] = []
        for start in range(0, self._size, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            batches.append(
                (
                    self._states[batch_idx],
                    self._actions[batch_idx],
                    self._log_probs_old[batch_idx],
                    self._advantages[batch_idx],
                    self._returns[batch_idx],
                    self._values_old[batch_idx],
                )
            )
        return batches

    def _flatten_state(self, state: torch.Tensor) -> torch.Tensor:
        state = state.detach().cpu().float().view(-1)
        if state.shape[0] != self._state_dim:
            raise ValueError(
                f"state dim {state.shape[0]} does not match buffer state_dim {self._state_dim}"
            )
        return state


class ContinuousFixedRolloutBuffer(RolloutBuffer):
    """Fixed-size rollout buffer for continuous actions."""

    def __init__(self, rollout_steps: int, state_dim: int, action_dim: int) -> None:
        super().__init__(rollout_steps)
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._size = 0
        self._states = torch.zeros((rollout_steps, state_dim), dtype=torch.float32)
        self._actions = torch.zeros((rollout_steps, action_dim), dtype=torch.float32)
        self._log_probs_old = torch.zeros(rollout_steps, dtype=torch.float32)
        self._rewards = torch.zeros(rollout_steps, dtype=torch.float32)
        self._bootstrap_dones = torch.zeros(rollout_steps, dtype=torch.float32)
        self._values_old = torch.zeros(rollout_steps, dtype=torch.float32)
        self._advantages = torch.zeros(rollout_steps, dtype=torch.float32)
        self._returns = torch.zeros(rollout_steps, dtype=torch.float32)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def push(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        bootstrap_done: bool,
        value_old: float,
    ) -> None:
        raise TypeError(
            "use push_continuous() for ContinuousFixedRolloutBuffer"
        )

    def push_continuous(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: float,
        reward: float,
        bootstrap_done: bool,
        value_old: float,
    ) -> None:
        if self.is_full():
            raise RuntimeError("rollout buffer is full; call clear() after update()")

        index = self._size
        self._states[index] = self._flatten_state(state)
        self._actions[index] = self._flatten_action(action)
        self._log_probs_old[index] = log_prob
        self._rewards[index] = reward
        self._bootstrap_dones[index] = float(bootstrap_done)
        self._values_old[index] = value_old
        self._size += 1

    def is_full(self) -> bool:
        return self._size >= self._rollout_steps

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._size = 0

    def set_advantages(self, advantages: torch.Tensor) -> None:
        if advantages.shape != (self._size,):
            raise ValueError("advantages must have shape (len(buffer),)")
        self._advantages[: self._size] = advantages

    def set_returns(self, returns: torch.Tensor) -> None:
        if returns.shape != (self._size,):
            raise ValueError("returns must have shape (len(buffer),)")
        self._returns[: self._size] = returns

    def tensor_views(self) -> dict[str, torch.Tensor]:
        end = self._size
        return {
            "rewards": self._rewards[:end],
            "bootstrap_dones": self._bootstrap_dones[:end],
            "values_old": self._values_old[:end],
        }

    def iter_minibatches(
        self,
        minibatch_size: int,
        *,
        shuffle: bool,
    ) -> list[tuple[torch.Tensor, ...]]:
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if self._size == 0:
            raise RuntimeError("buffer is empty")

        indices = (
            torch.randperm(self._size) if shuffle else torch.arange(self._size)
        )
        batches: list[tuple[torch.Tensor, ...]] = []
        for start in range(0, self._size, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            batches.append(
                (
                    self._states[batch_idx],
                    self._actions[batch_idx],
                    self._log_probs_old[batch_idx],
                    self._advantages[batch_idx],
                    self._returns[batch_idx],
                    self._values_old[batch_idx],
                )
            )
        return batches

    def _flatten_state(self, state: torch.Tensor) -> torch.Tensor:
        state = state.detach().cpu().float().view(-1)
        if state.shape[0] != self._state_dim:
            raise ValueError(
                f"state dim {state.shape[0]} does not match buffer state_dim {self._state_dim}"
            )
        return state

    def _flatten_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.detach().cpu().float().view(-1)
        if action.shape[0] != self._action_dim:
            raise ValueError(
                f"action dim {action.shape[0]} does not match buffer action_dim {self._action_dim}"
            )
        return action


class VectorContinuousFixedRolloutBuffer(RolloutBuffer):
    """Rollout buffer for ``n_envs`` parallel envs and ``rollout_steps`` vector steps."""

    def __init__(
        self,
        rollout_steps: int,
        n_envs: int,
        state_dim: int,
        action_dim: int,
    ) -> None:
        if n_envs <= 0:
            raise ValueError("n_envs must be positive")
        super().__init__(rollout_steps)
        self._n_envs = n_envs
        self._state_dim = state_dim
        self._action_dim = action_dim
        capacity = rollout_steps * n_envs
        self._capacity = capacity
        self._vector_steps = 0
        self._states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self._actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self._log_probs_old = torch.zeros(capacity, dtype=torch.float32)
        self._rewards = torch.zeros(capacity, dtype=torch.float32)
        self._bootstrap_dones = torch.zeros(capacity, dtype=torch.float32)
        self._values_old = torch.zeros(capacity, dtype=torch.float32)
        self._advantages = torch.zeros(capacity, dtype=torch.float32)
        self._returns = torch.zeros(capacity, dtype=torch.float32)

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(
        self,
        state: torch.Tensor,
        action: int,
        log_prob: float,
        reward: float,
        bootstrap_done: bool,
        value_old: float,
    ) -> None:
        raise TypeError("use push_vector_step() for VectorContinuousFixedRolloutBuffer")

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
            raise RuntimeError("rollout buffer is full; call clear() after update()")

        states = states.detach().cpu().float()
        actions = actions.detach().cpu().float()
        log_probs = log_probs.detach().cpu().float().view(-1)
        rewards = rewards.detach().cpu().float().view(-1)
        bootstrap_dones = bootstrap_dones.detach().cpu().float().view(-1)
        values_old = values_old.detach().cpu().float().view(-1)

        if states.shape != (self._n_envs, self._state_dim):
            raise ValueError(
                f"states must have shape ({self._n_envs}, {self._state_dim}), got {tuple(states.shape)}"
            )
        if actions.shape != (self._n_envs, self._action_dim):
            raise ValueError(
                f"actions must have shape ({self._n_envs}, {self._action_dim}), got {tuple(actions.shape)}"
            )
        for tensor, name in (
            (log_probs, "log_probs"),
            (rewards, "rewards"),
            (bootstrap_dones, "bootstrap_dones"),
            (values_old, "values_old"),
        ):
            if tensor.shape != (self._n_envs,):
                raise ValueError(f"{name} must have shape ({self._n_envs},), got {tuple(tensor.shape)}")

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

    def __len__(self) -> int:
        return self._vector_steps * self._n_envs

    def clear(self) -> None:
        self._vector_steps = 0

    def set_advantages(self, advantages: torch.Tensor) -> None:
        size = len(self)
        if advantages.shape != (size,):
            raise ValueError("advantages must have shape (len(buffer),)")
        self._advantages[:size] = advantages

    def set_returns(self, returns: torch.Tensor) -> None:
        size = len(self)
        if returns.shape != (size,):
            raise ValueError("returns must have shape (len(buffer),)")
        self._returns[:size] = returns

    def tensor_views(self) -> dict[str, torch.Tensor]:
        size = len(self)
        return {
            "rewards": self._rewards[:size],
            "bootstrap_dones": self._bootstrap_dones[:size],
            "values_old": self._values_old[:size],
        }

    def tensor_views_time_env(self) -> dict[str, torch.Tensor]:
        """Views shaped ``(vector_steps, n_envs)`` for per-env GAE."""
        t_steps = self._vector_steps
        size = t_steps * self._n_envs
        return {
            "rewards": self._rewards[:size].view(t_steps, self._n_envs),
            "bootstrap_dones": self._bootstrap_dones[:size].view(t_steps, self._n_envs),
            "values_old": self._values_old[:size].view(t_steps, self._n_envs),
        }

    def iter_minibatches(
        self,
        minibatch_size: int,
        *,
        shuffle: bool,
    ) -> list[tuple[torch.Tensor, ...]]:
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        size = len(self)
        if size == 0:
            raise RuntimeError("buffer is empty")

        indices = torch.randperm(size) if shuffle else torch.arange(size)
        batches: list[tuple[torch.Tensor, ...]] = []
        for start in range(0, size, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            batches.append(
                (
                    self._states[batch_idx],
                    self._actions[batch_idx],
                    self._log_probs_old[batch_idx],
                    self._advantages[batch_idx],
                    self._returns[batch_idx],
                    self._values_old[batch_idx],
                )
            )
        return batches
