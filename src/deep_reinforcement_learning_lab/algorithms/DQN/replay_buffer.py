from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
import random
from typing import Deque

import torch


class ReplayBuffer(ABC):
    """Abstract replay buffer for off-policy RL."""

    def __init__(self, buffer_size: int) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self._buffer_size = buffer_size

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @abstractmethod
    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        ...

    @abstractmethod
    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


class _Transition:
    __slots__ = ("state", "action", "reward", "next_state", "done")

    def __init__(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.state = state.detach().cpu().clone()
        self.action = action
        self.reward = float(reward)
        self.next_state = next_state.detach().cpu().clone()
        self.done = bool(done)


class UniformReplayBuffer(ReplayBuffer):
    """FIFO ring buffer with uniform random sampling."""

    def __init__(self, buffer_size: int) -> None:
        super().__init__(buffer_size)
        self._storage: Deque[_Transition] = deque(maxlen=buffer_size)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self._storage.append(_Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        if batch_size > len(self):
            raise ValueError("batch_size cannot exceed the number of stored transitions")
        batch = random.sample(self._storage, batch_size)
        states = torch.stack([t.state for t in batch])
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.stack([t.next_state for t in batch])
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._storage)
