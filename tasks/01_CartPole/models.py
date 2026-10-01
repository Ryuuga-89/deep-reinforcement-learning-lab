from __future__ import annotations

import torch
import torch.nn as nn


class QNet(nn.Module):
    """MLP Q ネットワーク: 状態 → 各行動の Q 値（活性化なし出力）。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must not be empty")

        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, action_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


def build_qnet(
    state_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...],
) -> QNet:
    return QNet(state_dim, action_dim, hidden_sizes)
