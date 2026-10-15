"""CartPole タスク用 TensorBoard ロギング。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from deep_reinforcement_learning_lab.config import (
    ExperimentConfig,
    archive_experiment_yaml,
    experiment_flat_fields,
    format_experiment_flat_fields,
)

if TYPE_CHECKING:
    from deep_reinforcement_learning_lab.algorithms.DQN import DQN

from env import noise_obs_reference_std_text

_CONFIG_STEP = 0
_HPARAMS_RUN_NAME = "."
_FINAL_EVAL_HPARAM = "hparam/final_mean_episode_length"


def _experiment_log_fields(experiment: ExperimentConfig) -> dict[str, str | int | float | bool]:
    flat = dict(experiment_flat_fields(experiment))
    if experiment.subtask == "noise":
        flat["subtask_options/noise_obs_ref_std"] = noise_obs_reference_std_text()
    return flat


class TrainingLogger:
    """CartPole DQN 学習用 TensorBoard ロガー（Train/ と Eval/ でタグを分離）。"""

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_experiment_start(self, experiment: ExperimentConfig, config_path: Path) -> None:
        resolved = config_path.resolve()
        yaml_text = resolved.read_text(encoding="utf-8")
        flat = _experiment_log_fields(experiment)

        archive_experiment_yaml(self.log_dir, config_path)
        self._writer.add_text("Config/path", str(resolved), global_step=_CONFIG_STEP)
        self._writer.add_text(
            "Config/yaml",
            f"```yaml\n{yaml_text}```",
            global_step=_CONFIG_STEP,
        )
        self._writer.add_text(
            "Config/summary",
            format_experiment_flat_fields(flat),
            global_step=_CONFIG_STEP,
        )

    def log_experiment_end(
        self,
        experiment: ExperimentConfig,
        final_mean_episode_length: float,
    ) -> None:
        self._writer.add_hparams(
            _experiment_log_fields(experiment),
            {_FINAL_EVAL_HPARAM: final_mean_episode_length},
            run_name=_HPARAMS_RUN_NAME,
        )

    def log_train_episode(
        self,
        env_step: int,
        episode_reward: float,
        episode_length: int,
    ) -> None:
        self._writer.add_scalar("Train/episode_reward", episode_reward, env_step)
        self._writer.add_scalar("Train/episode_length", episode_length, env_step)

    def log_train_epsilon(self, env_step: int, agent: DQN) -> None:
        epsilon = max(
            agent.epsilon_end,
            agent.epsilon_start - agent.epsilon_rate * agent.env_step,
        )
        self._writer.add_scalar("Train/epsilon", epsilon, env_step)

    def log_train_loss(self, gradient_step: int, loss: float) -> None:
        self._writer.add_scalar("Train/loss", loss, gradient_step)

    def log_eval_mean_episode_length(
        self,
        env_step: int,
        mean_episode_length: float,
    ) -> None:
        self._writer.add_scalar("Eval/mean_episode_length", mean_episode_length, env_step)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def attach_loss_logging(agent: DQN, logger: TrainingLogger) -> None:
    """DQN.update をラップし、勾配更新ごとに loss を TensorBoard へ記録する。"""

    original_update = agent.update

    def update_with_logging() -> float:
        loss = original_update()
        logger.log_train_loss(agent.gradient_step, loss)
        return loss

    agent.update = update_with_logging  # type: ignore[method-assign]
