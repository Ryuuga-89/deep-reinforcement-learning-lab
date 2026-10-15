# 実験設定（ExperimentConfig）

## 方針

実験 YAML は次の区分を持つ（トップレベル）。

| 区分 | YAML | コード |
|------|------|--------|
| サブタスク | `subtask: <name>` | 各タスクの `make_subtask_env` レジストリ |
| サブタスクオプション（任意） | `subtask_options:` | `SubtaskOptions`（タスクが解釈） |
| アルゴリズム | `algorithm: <name>` | `src` の [`algorithms/registry.py`](../../src/deep_reinforcement_learning_lab/algorithms/registry.py) |
| 実験設定 | `run:` ブロック | `RunConfig` |
| ハイパーパラメータ | `hyperparameters:` ブロック | アルゴリズムごとにパース（例: `DQNHyperparameters`） |

`env_id` や報酬の式そのものはサブタスク実装に閉じる。CartPole では次を `subtask_options` で変える（サブタスクごとに必要なキーだけ指定）。

| サブタスク | キー |
|------------|------|
| `center` | `center_bonus_coef`（中央ボーナス係数） |
| `move` | `move_alpha`（目標位置ボーナス係数）、`move_beta`（目標切替の間隔、1–500） |
| `noise` | `noise_rho`（観測ガウスノイズの無次元スケール ρ） |

## 登録済みアルゴリズム（`src`）

| 名前 | ハイパーパラメータ型 | 学習 |
|------|---------------------|------|
| `dqn` | `DQNHyperparameters` | タスク側が `register_algorithm("dqn", ...)` で runner を登録 |
| `ddqn` | `DQNHyperparameters`（DQN と同一） | タスク側が `register_algorithm("ddqn", ...)` で runner を登録 |
| `per_dqn` | `PERHyperparameters` | タスク側が `register_algorithm("per_dqn", ...)` で runner を登録 |
| `per_ddqn` | `PERHyperparameters` | タスク側が `register_algorithm("per_ddqn", ...)` で runner を登録 |
| `ppo` | `PPOHyperparameters` | タスク側が `register_algorithm("ppo", ...)` で runner を登録 |

## PPO ハイパーパラメータ（`hyperparameters:`）

| キー | 説明 |
|------|------|
| `gamma`, `learning_rate`, `gae_lambda`, `clip_coef`, `entropy_coef`, `value_coef` | 学習の基本 |
| `rollout_steps` | **各並列 env あたり**の収集ステップ数（ベクトル step 数） |
| `n_envs` | 並列環境数（省略時 1）。1 更新あたりの遷移数は `n_envs × rollout_steps` |
| `n_epochs`, `minibatch_size` | 各ロールアウトのミニバッチ学習 |
| `hidden_sizes`, `ac_type` | Actor-Critic 構造（`shared` / `separate`） |
| `clip_value_coef`, `max_grad_norm` | 省略時は価値クリップ = `clip_coef`、勾配クリップ 0.5 |

ローダーは `n_envs * rollout_steps >= minibatch_size` および `minibatch_size >= 2` を要求する。

## DQN 系ハイパーパラメータ（`hyperparameters:`）

`dqn` / `ddqn` は `DQNHyperparameters`、`per_dqn` / `per_ddqn` はその拡張 `PERHyperparameters`（DQN キーに加えて PER 用キー）。

| キー | 説明 |
|------|------|
| `gamma`, `learning_rate`, `buffer_size`, `batch_size` | 学習の基本 |
| `epsilon_start`, `epsilon_end`, `epsilon_decay_steps` | ε-greedy（減衰は環境ステップ） |
| `loss_type` | `mse` または `huber`（省略時はローダー既定） |
| `hidden_sizes` | 共有 MLP の各層出力次元 |
| `qnet_type` | `mlp`（既定）または `dueling`。Dueling は **別 `algorithm` ではなく** Q ネット構造の選択（CartPole は `models.py`） |
| `target_update_interval` | ハード同期の間隔（**勾配更新回数**）。`tau` と排他 |
| `tau` | ソフト同期（Polyak）係数。`target_update_interval` と排他 |

PER 追加キー: `per_alpha`, `per_beta_start`, `per_beta_end`, `priority_epsilon`。β のアニーリングはタスク runner が `beta_anneal_steps`（CartPole では `run.max_env_steps` を渡す）に対し **勾配更新回数**で線形増加する。

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
- `run.eval_seed_offset` / `run.video_seed_offset`（YAML 省略時は 10000 / 20000）で eval 用・動画用の `reset` シードを分ける（`seed + offset + episode`）。重複はローダーで拒否。CartPole の多くの設定では `0` / `100` を明示している。
