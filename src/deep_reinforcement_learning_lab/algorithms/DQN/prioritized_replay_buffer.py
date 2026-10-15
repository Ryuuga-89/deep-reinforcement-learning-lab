from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from deep_reinforcement_learning_lab.algorithms.DQN.replay_buffer import (
    ReplayBuffer,
    _Transition,
)


class _SumTree:
    """Binary sum tree for O(log n) proportional sampling."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._data: list[Any | None] = [None] * capacity
        self._write = 0
        self._size = 0

    @property
    def total(self) -> float:
        return float(self._tree[0])

    @property
    def max_priority(self) -> float:
        if self._size == 0:
            return 1.0
        leaf_start = self._capacity - 1
        leaves = self._tree[leaf_start : leaf_start + self._capacity]
        return float(max(leaves.max(), 1e-6))

    def add(self, priority: float, data: _Transition) -> int:
        """Store data at the write cursor; return its data index."""
        data_idx = self._write
        self._data[data_idx] = data
        self._update_tree(data_idx, priority)
        self._write = (self._write + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        return data_idx

    def update(self, data_idx: int, priority: float) -> None:
        self._update_tree(data_idx, priority)

    def _update_tree(self, data_idx: int, priority: float) -> None:
        tree_idx = data_idx + self._capacity - 1
        change = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def _propagate(self, tree_idx: int, change: float) -> None:
        parent = (tree_idx - 1) // 2
        self._tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get(self, value: float) -> int:
        """Return a data index sampled by descending from the root with ``value``."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self._tree):
                break
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = right
        return idx - (self._capacity - 1)

    def leaf_priority(self, data_idx: int) -> float:
        tree_idx = data_idx + self._capacity - 1
        return float(self._tree[tree_idx])

    def __getitem__(self, data_idx: int) -> _Transition:
        item = self._data[data_idx]
        if item is None:
            raise IndexError(data_idx)
        return item


class PrioritizedReplayBuffer(ReplayBuffer):
    """FIFO ring buffer with prioritized sampling (Schaul et al.)."""

    def __init__(self, buffer_size: int, alpha: float) -> None:
        super().__init__(buffer_size)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self._alpha = alpha
        self._tree = _SumTree(buffer_size)

    @property
    def alpha(self) -> float:
        return self._alpha

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        transition = _Transition(state, action, reward, next_state, done)
        priority = self._tree.max_priority
        self._tree.add(priority, transition)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError("use sample_prioritized for prioritized replay")

    def sample_prioritized(
        self,
        batch_size: int,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        if batch_size > len(self):
            raise ValueError("batch_size cannot exceed the number of stored transitions")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")

        n = len(self)
        indices: list[int] = []
        priorities: list[float] = []
        total = self._tree.total

        if total <= 0.0:
            indices = random.sample(range(n), batch_size)
            priorities = [1.0] * batch_size
        else:
            segment = total / batch_size
            for batch_index in range(batch_size):
                low = segment * batch_index
                high = segment * (batch_index + 1)
                sample_value = random.uniform(low, high)
                data_idx = self._tree.get(sample_value)
                indices.append(data_idx)
                priorities.append(self._tree.leaf_priority(data_idx))

        batch = [self._tree[idx] for idx in indices]
        states = torch.stack([t.state for t in batch])
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.stack([t.next_state for t in batch])
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

        if total <= 0.0:
            prob = 1.0 / n
            weights = np.full(batch_size, (n * prob) ** (-beta), dtype=np.float64)
        else:
            probs = np.array(priorities, dtype=np.float64) / total
            probs = np.maximum(probs, 1e-12)
            weights = (n * probs) ** (-beta)

        weights /= weights.max()
        weight_tensor = torch.tensor(weights, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, indices, weight_tensor

    def update_priorities(
        self,
        indices: list[int],
        td_errors: torch.Tensor,
        priority_epsilon: float,
    ) -> None:
        if len(indices) != td_errors.numel():
            raise ValueError("indices and td_errors must have the same length")
        priorities = (td_errors.detach().abs().cpu().numpy() + priority_epsilon) ** self._alpha
        for data_idx, priority in zip(indices, priorities, strict=True):
            self._tree.update(int(data_idx), float(priority))

    def __len__(self) -> int:
        return self._tree._size
