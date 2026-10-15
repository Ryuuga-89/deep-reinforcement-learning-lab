"""Pendulum タスク用 TensorBoard ロギング。"""

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
    from deep_reinforcement_learning_lab.algorithms.PPO import ContinuousPPO

_CONFIG_STEP = 0
_HPARAMS_RUN_NAME = "."
_FINAL_EVAL_HPARAM = "hparam/final_mean_episode_return"


class TrainingLogger:
    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_experiment_start(self, experiment: ExperimentConfig, config_path: Path) -> None:
        resolved = config_path.resolve()
        yaml_text = resolved.read_text(encoding="utf-8")
        flat = dict(experiment_flat_fields(experiment))

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
        final_mean_episode_return: float,
    ) -> None:
        self._writer.add_hparams(
            experiment_flat_fields(experiment),
            {_FINAL_EVAL_HPARAM: final_mean_episode_return},
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

    def log_train_ppo_losses(
        self,
        gradient_step: int,
        *,
        policy_loss: float,
        value_loss: float,
        entropy_loss: float,
        total_loss: float,
    ) -> None:
        self._writer.add_scalar("Train/policy_loss", policy_loss, gradient_step)
        self._writer.add_scalar("Train/value_loss", value_loss, gradient_step)
        self._writer.add_scalar("Train/entropy_loss", entropy_loss, gradient_step)
        self._writer.add_scalar("Train/total_loss", total_loss, gradient_step)

    def log_eval_mean_episode_return(
        self,
        env_step: int,
        mean_episode_return: float,
    ) -> None:
        self._writer.add_scalar(
            "Eval/mean_episode_return",
            mean_episode_return,
            env_step,
        )

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def attach_ppo_loss_logging(agent: ContinuousPPO, logger: TrainingLogger) -> None:
    original_update = agent.update

    def update_with_logging():
        losses = original_update()
        logger.log_train_ppo_losses(
            agent.gradient_step,
            policy_loss=losses.policy_loss,
            value_loss=losses.value_loss,
            entropy_loss=losses.entropy_loss,
            total_loss=losses.total,
        )
        return losses

    agent.update = update_with_logging  # type: ignore[method-assign]
