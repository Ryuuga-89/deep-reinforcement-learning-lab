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
