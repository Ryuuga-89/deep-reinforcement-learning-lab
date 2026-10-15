from __future__ import annotations

REGISTERED_ALGORITHMS: frozenset[str] = frozenset(
    {"dqn", "ddqn", "per_dqn", "per_ddqn"}
)


def validate_algorithm(name: str) -> None:
    if name not in REGISTERED_ALGORITHMS:
        raise ValueError(
            f"unsupported algorithm: {name!r}. "
            f"registered: {sorted(REGISTERED_ALGORITHMS)}"
        )
