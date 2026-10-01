from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from gymnasium.core import Env

from deep_reinforcement_learning_lab.algorithms.names import validate_algorithm
from deep_reinforcement_learning_lab.config.experiment_config import ExperimentConfig


class EnvFactory(Protocol):
    def __call__(self, *, render_mode: str | None = None) -> Env: ...


@dataclass(frozen=True)
class TrainingContext:
    experiment: ExperimentConfig
    config_path: Path
    task_dir: Path
    make_env: EnvFactory


TrainingRunner = Callable[[TrainingContext], float]

_RUNNERS: dict[str, TrainingRunner] = {}


def register_algorithm(name: str, runner: TrainingRunner) -> None:
    validate_algorithm(name)
    _RUNNERS[name] = runner


def run_training(context: TrainingContext) -> float:
    algorithm = context.experiment.algorithm
    validate_algorithm(algorithm)
    runner = _RUNNERS.get(algorithm)
    if runner is None:
        raise ValueError(
            f"no training runner registered for algorithm: {algorithm!r}. "
            "Register one with register_algorithm() from the task entrypoint."
        )
    return runner(context)
