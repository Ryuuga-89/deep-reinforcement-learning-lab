from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

ENV_ID = "CartPole-v1"
SEED = 42
MAX_STEPS = 1000
VIDEO_DIR = Path(__file__).resolve().parent / "videos"
RECORD_EPISODE = 0  # このエピソード番号だけ動画に記録する
SIMULATION_FPS = 50  # CartPole-v1 の render_fps
PLAYBACK_SPEED = 0.2  # 1.0=等速, 小さいほどスロー（0.2 なら 5 倍スロー）
VIDEO_FPS = max(1, int(SIMULATION_FPS * PLAYBACK_SPEED))

env = gym.make(ENV_ID, render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder=str(VIDEO_DIR),
    name_prefix=ENV_ID,
    episode_trigger=lambda episode_id: episode_id == RECORD_EPISODE,
    fps=VIDEO_FPS,
)

print("=== Gymnasium 動作実験 ===")
print(f"環境 ID: {ENV_ID}")
print(f"観測空間 (observation_space): {env.observation_space}")
print(f"行動空間 (action_space): {env.action_space}")
print(f"動画保存先: {VIDEO_DIR}")
print(f"記録対象: エピソード {RECORD_EPISODE + 1}")
print(f"動画 FPS: {VIDEO_FPS}（等速比 x{PLAYBACK_SPEED}）")
print()

observation, info = env.reset(seed=SEED)
print(f"reset(seed={SEED})")
print(f"  初期観測: {observation}")
print(f"  info: {info}")
print()

step_count = 0
episode_count = 0
episode_step = 0
episode_reward = 0.0
total_reward = 0.0

for step_count in range(1, MAX_STEPS + 1):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    episode_step += 1
    episode_reward += reward
    total_reward += reward

    if step_count <= 3:
        print(f"step {step_count}")
        print(f"  action: {action}")
        print(f"  observation: {observation}")
        print(f"  reward: {reward}")
        print(f"  terminated: {terminated}, truncated: {truncated}")
        print(f"  info: {info}")
        print()

    if terminated or truncated:
        episode_count += 1
        reason = "terminated" if terminated else "truncated"
        print(f"エピソード {episode_count} 終了 ({reason})")
        print(f"  ステップ数: {episode_step}")
        print(f"  報酬合計: {episode_reward}")
        print()

        observation, info = env.reset()
        episode_step = 0
        episode_reward = 0.0

env.close()

print("=== サマリー ===")
print(f"総ステップ数: {step_count}")
print(f"完了エピソード数: {episode_count}")
print(f"報酬合計: {total_reward}")
print(f"1ステップあたり平均報酬: {total_reward / step_count:.3f}")

video_files = sorted(VIDEO_DIR.glob("*.mp4"))
if video_files:
    print()
    print("=== 動画 ===")
    for video_path in video_files:
        print(f"  {video_path}")
    print()
    print("手元の PC から取得する例:")
    print(f"  scp k12:{video_files[-1]} .")
else:
    print()
    print("警告: 動画ファイルが生成されませんでした。")
