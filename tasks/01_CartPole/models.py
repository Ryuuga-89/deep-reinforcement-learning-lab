from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

QNetType = Literal["mlp", "dueling"]


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


class DuelingQNet(nn.Module):
    """Dueling 構造: 共有 MLP → V(s) と A(s,a) を合成して Q(s,a) を出力。"""

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
        self.feature = nn.Sequential(*layers)
        self.value_stream = nn.Linear(in_dim, 1)
        self.advantage_stream = nn.Linear(in_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


def build_qnet(
    state_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...],
) -> QNet:
    return QNet(state_dim, action_dim, hidden_sizes)


def build_dueling_qnet(
    state_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...],
) -> DuelingQNet:
    return DuelingQNet(state_dim, action_dim, hidden_sizes)


def build_q_model(
    qnet_type: QNetType,
    state_dim: int,
    action_dim: int,
    hidden_sizes: tuple[int, ...],
) -> nn.Module:
    if qnet_type == "mlp":
        return build_qnet(state_dim, action_dim, hidden_sizes)
    if qnet_type == "dueling":
        return build_dueling_qnet(state_dim, action_dim, hidden_sizes)
    raise ValueError(f"unsupported qnet_type: {qnet_type}")
