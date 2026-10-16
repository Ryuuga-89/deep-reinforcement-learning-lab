from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "mlp_actor_critic_discrete"


@dataclass(frozen=True)
class ModelConfig:
    id: str
    options: dict[str, Any] = field(default_factory=dict)


def parse_model_config(block: Any, path: Path) -> ModelConfig:
    if block is None:
        raise ValueError(f"model block is required (id): {path}")
    if not isinstance(block, dict):
        raise ValueError(f"model must be a mapping: {path}")

    allowed = {"id", "options"}
    unknown = set(block) - allowed
    if unknown:
        raise ValueError(
            f"unknown model keys in {path}: {sorted(unknown)} "
            "(policy is no longer configured here; use model.id)"
        )

    raw_id = block.get("id", DEFAULT_MODEL_ID)
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError(f"model.id must be a non-empty string: {path}")

    options_raw = block.get("options", {})
    if options_raw is None:
        options_raw = {}
    if not isinstance(options_raw, dict):
        raise ValueError(f"model.options must be a mapping: {path}")

    return ModelConfig(
        id=raw_id.strip(),
        options=dict(options_raw),
    )
