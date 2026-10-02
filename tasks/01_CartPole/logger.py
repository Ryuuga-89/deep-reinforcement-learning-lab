"""CartPole タスク用 TensorBoard ロギング。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deep_reinforcement_learning_lab.algorithms.DQN import DQN


class TrainingLogger:
    """CartPole DQN 学習用 TensorBoard ロガー（Train/ と Eval/ でタグを分離）。"""

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(log_dir))

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
