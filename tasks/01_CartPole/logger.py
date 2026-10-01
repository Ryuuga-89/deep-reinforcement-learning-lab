"""CartPole タスク用 TensorBoard ロギング。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deep_reinforcement_learning_lab.algorithms.DQN import DQN

class TrainingLogger:
    """CartPole DQN 学習用 TensorBoard ロガー。"""

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_episode(self, env_step: int, reward: float) -> None:
        self._writer.add_scalar("Reward/episode", reward, env_step)

    def log_eval(self, env_step: int, mean_reward: float) -> None:
        self._writer.add_scalar("Reward/eval", mean_reward, env_step)

    def log_loss(self, gradient_step: int, loss: float) -> None:
        self._writer.add_scalar("Loss/train", loss, gradient_step)

    def log_epsilon(self, env_step: int, agent: DQN) -> None:
        epsilon = max(
            agent.epsilon_end,
            agent.epsilon_start - agent.epsilon_rate * agent.env_step,
        )
        self._writer.add_scalar("Exploration/epsilon", epsilon, env_step)

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
        logger.log_loss(agent.gradient_step, loss)
        return loss

    agent.update = update_with_logging  # type: ignore[method-assign]
