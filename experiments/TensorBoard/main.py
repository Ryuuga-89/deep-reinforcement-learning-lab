from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

RUN_DIR = Path(__file__).resolve().parent / "runs" / "cartpole"
NUM_EPISODES = 10


def main() -> None:
    print("=== TensorBoard 動作実験 ===")
    print(f"ログ保存先: {RUN_DIR}")
    print()

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    with SummaryWriter(log_dir=str(RUN_DIR)) as writer:
        for episode in range(1, NUM_EPISODES + 1):
            reward = float(episode * 10)
            loss = 1.0 / episode
            lr = 0.001 * (0.9 ** (episode - 1))

            writer.add_scalar("Reward/train", reward, episode)
            writer.add_scalar("Loss/train", loss, episode)
            writer.add_scalar("LearningRate", lr, episode)

            print(
                f"episode {episode:2d}  "
                f"reward={reward:5.1f}  loss={loss:.3f}  lr={lr:.6f}"
            )

    event_files = sorted(RUN_DIR.glob("events.out.tfevents.*"))
    print()
    print("=== サマリー ===")
    print(f"記録エピソード数: {NUM_EPISODES}")
    if event_files:
        print(f"イベントファイル: {event_files[-1]}")
    else:
        print("警告: イベントファイルが生成されませんでした。")
    print()
    print("可視化:")
    print("  uv run tensorboard --logdir experiments/TensorBoard/runs")


if __name__ == "__main__":
    main()
