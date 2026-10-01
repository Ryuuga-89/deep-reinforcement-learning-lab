# 実験設定（ExperimentConfig）

## 方針

実験 YAML は次の4区分のみを持つ。

| 区分 | YAML | コード |
|------|------|--------|
| サブタスク | `subtask: <name>` | 各タスクの `make_subtask_env` レジストリ |
| アルゴリズム | `algorithm: <name>` | `src` の [`algorithms/registry.py`](../../src/deep_reinforcement_learning_lab/algorithms/registry.py) |
| 実験設定 | `run:` ブロック | `RunConfig` |
| ハイパーパラメータ | `hyperparameters:` ブロック | アルゴリズムごとにパース（例: `DQNHyperparameters`） |

`env_id` や報酬の詳細は YAML に書かない。サブタスク名の実装に閉じる。

## 登録済みアルゴリズム（`src`）

| 名前 | ハイパーパラメータ型 | 学習 |
|------|---------------------|------|
| `dqn` | `DQNHyperparameters` | タスク側が `register_algorithm("dqn", ...)` で runner を登録 |

## ローダー

- [`load_experiment_config`](../../src/deep_reinforcement_learning_lab/config/experiment_config.py) が YAML を読み込む。
- 未知のトップレベルキー・ハイパーパラメータキーは拒否する。
- `algorithm` はレジストリの登録名と一致すること。

## 出力

- `run_dir = <task_dir> / "runs" / <subtask> / <algorithm> / <run.name> /`
- 既存 `run_dir` がある場合は学習を開始しない（タスク `main` の契約）。
- TensorBoard はタスク直下の `runs/` を `--logdir` に指定すると、サブタスク・アルゴリズム配下の run をまとめて閲覧できる。

## 動画（CartPole）

- `run.video_enabled` でオン／オフ。オン時のみ `video_interval_steps` で `run_dir/videos/` に MP4 を保存する。
- `run.video_episodes` は完全平方数（既定 16 → `RunConfig.video_grid_side` で辺長）。実装は `tasks/01_CartPole/record.py`。
- 各マスは greedy 1 エピソード。`terminated` マスは暗転。全マス失敗時はホールド後に全暗転で終了。
- 各マス右上に環境ステップを大きく表示（エピソード終了・失敗後はその値で固定）。
- フレーム合成は imageio へ逐次書き出し（完成動画分のフレームリストは保持しない）。
- `run.eval_seed_offset` / `run.video_seed_offset`（既定 10000 / 20000）で eval 用・動画用の `reset` シードを分ける。重複はローダーで拒否。
