"""離散モデル用: ロールアウト index → 連続 env 行動 (K, 2) の写像テーブル。"""

from __future__ import annotations

from typing import Any

import numpy as np

from env import LUNAR_ENV_ACTION_DIM

# Gymnasium discrete LunarLander と力学一致（main / lateral）
GYMNASIUM_4: np.ndarray = np.array(
    [
        [-1.0, 0.0],
        [-1.0, -1.0],
        [1.0, 0.0],
        [-1.0, 1.0],
    ],
    dtype=np.float32,
)

MAIN_ON_OFF: np.ndarray = np.array(
    [
        [-1.0, 0.0],
        [1.0, 0.0],
    ],
    dtype=np.float32,
)

LATERAL_THREE: np.ndarray = np.array(
    [
        [-1.0, 0.0],
        [-1.0, -1.0],
        [-1.0, 1.0],
    ],
    dtype=np.float32,
)

# 主 ON/OFF × 横 {左, なし, 右}
MAIN_LATERAL_6: np.ndarray = np.array(
    [
        [-1.0, -1.0],
        [-1.0, 0.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=np.float32,
)

LUNAR_ACTION_TABLES: dict[str, np.ndarray] = {
    "gymnasium_4": GYMNASIUM_4,
    "main_on_off": MAIN_ON_OFF,
    "lateral_three": LATERAL_THREE,
    "main_lateral_6": MAIN_LATERAL_6,
}

DEFAULT_DISCRETE_TABLE = "gymnasium_4"
_DISCRETE_OPTION_KEYS = frozenset({"table"})


def list_action_table_names() -> tuple[str, ...]:
    return tuple(sorted(LUNAR_ACTION_TABLES))


def _validate_table(table: np.ndarray, name: str) -> np.ndarray:
    if table.ndim != 2 or table.shape[1] != LUNAR_ENV_ACTION_DIM:
        raise ValueError(
            f"action table {name!r} must have shape (K, {LUNAR_ENV_ACTION_DIM}), "
            f"got {table.shape}"
        )
    if table.shape[0] < 2:
        raise ValueError(f"action table {name!r} must have K >= 2, got {table.shape[0]}")
    return table.astype(np.float32, copy=False)


def action_table_by_name(name: str) -> np.ndarray:
    table = LUNAR_ACTION_TABLES.get(name)
    if table is None:
        known = list_action_table_names()
        raise ValueError(f"unknown action table {name!r}. known: {known}")
    return _validate_table(table, name)


def parse_discrete_model_options(options: dict[str, Any]) -> str:
    unknown = set(options) - _DISCRETE_OPTION_KEYS
    if unknown:
        raise ValueError(
            f"unknown model.options keys for discrete model: {sorted(unknown)}. "
            f"allowed: {sorted(_DISCRETE_OPTION_KEYS)}"
        )
    raw = options.get("table", DEFAULT_DISCRETE_TABLE)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("model.options.table must be a non-empty string")
    return raw.strip()


def resolve_action_table(options: dict[str, Any]) -> np.ndarray:
    name = parse_discrete_model_options(options)
    return action_table_by_name(name)


def num_rollout_actions(options: dict[str, Any]) -> int:
    return int(resolve_action_table(options).shape[0])
