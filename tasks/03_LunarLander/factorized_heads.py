"""環境次元ごとに独立した離散ヘッド（各ヘッドは独自の levels）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from env import LUNAR_ENV_ACTION_DIM


@dataclass(frozen=True)
class FactorizedHeadSpec:
    levels: tuple[float, ...]

    @property
    def num_choices(self) -> int:
        return len(self.levels)


@dataclass(frozen=True)
class FactorizedHeadConfig:
    heads: tuple[FactorizedHeadSpec, ...]

    @property
    def n_heads(self) -> int:
        return len(self.heads)


_DEFAULT_HEAD_LEVELS: tuple[tuple[float, ...], ...] = (
    (-1.0, 1.0),
    (-1.0, 1.0),
)

_DEFAULT_GYMNASIUM_STYLE: tuple[tuple[float, ...], ...] = (
    (-1.0, 1.0),
    (-1.0, 0.0, 1.0),
)


def _parse_level_list(raw: Any, *, path: str) -> tuple[float, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError(f"{path} must be a list of at least 2 floats")
    levels: list[float] = []
    for item in raw:
        if not isinstance(item, (int, float)):
            raise ValueError(f"{path} entries must be numbers")
        levels.append(float(item))
    return tuple(levels)


def parse_factorized_model_options(options: dict[str, Any]) -> FactorizedHeadConfig:
    allowed = {"head_levels", "preset"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(
            f"unknown model.options for factorized model: {sorted(unknown)}. "
            f"allowed: {sorted(allowed)}"
        )
    preset = options.get("preset", "binary_pair")
    if preset == "binary_pair":
        default_levels: tuple[tuple[float, ...], ...] = _DEFAULT_HEAD_LEVELS
    elif preset == "gymnasium_style":
        default_levels = _DEFAULT_GYMNASIUM_STYLE
    else:
        raise ValueError(
            f"unknown factorized preset {preset!r}. use binary_pair | gymnasium_style"
        )
    head_levels_raw = options.get("head_levels", default_levels)
    if not isinstance(head_levels_raw, (list, tuple)) or not head_levels_raw:
        raise ValueError("model.options.head_levels must be a non-empty list")
    head_specs = tuple(
        FactorizedHeadSpec(_parse_level_list(levels, path=f"head_levels[{i}]"))
        for i, levels in enumerate(head_levels_raw)
    )
    if len(head_specs) != LUNAR_ENV_ACTION_DIM:
        raise ValueError(
            f"factorized model requires {LUNAR_ENV_ACTION_DIM} heads (env action dims), "
            f"got {len(head_specs)}"
        )
    return FactorizedHeadConfig(heads=head_specs)


def decode_head_actions(
    head_actions: torch.Tensor,
    config: FactorizedHeadConfig,
) -> torch.Tensor:
    """head_actions: (batch, n_heads) long → env actions (batch, n_heads) float."""
    if head_actions.dim() != 2 or head_actions.shape[1] != config.n_heads:
        raise ValueError(
            f"head_actions must be (batch, {config.n_heads}), got {tuple(head_actions.shape)}"
        )
    batch = head_actions.shape[0]
    out = torch.zeros(batch, config.n_heads, dtype=torch.float32, device=head_actions.device)
    for h, spec in enumerate(config.heads):
        idx = head_actions[:, h].long()
        levels = torch.tensor(spec.levels, dtype=torch.float32, device=head_actions.device)
        out[:, h] = levels[idx]
    return out


def num_factorized_rollout_dims(config: FactorizedHeadConfig) -> int:
    return config.n_heads
