# 02 Pendulum

## タスクの内容

倒立振子の振子部分のみを制御し、直立状態に近づける（Gymnasium `Pendulum-v1`）。

### 観測空間

長さ 3 の連続ベクトル（`Box(3,)`）。

| インデックス | 意味 |
|:---:|:---|
| 0 | cos(θ) |
| 1 | sin(θ) |
| 2 | 角速度 θ̇ |

### 行動空間

`Box(1,)` — トルク ∈ [-2, 2]（連続）。

## 実験する内容

1. [`configs/default/baseline.yaml`](configs/default/baseline.yaml) の PPO で学習・評価・ログ・動画・チェックポイントが一通り動くことを確認する。
2. 比較用 YAML は `configs/default/` に追加し、**baseline から 1 軸だけ**変える。

実験メモは [`report/report.md`](report/report.md)。設定の正本は `configs/`。

## 実験設定の階層（YAML）

CartPole と同様。本タスクは `subtask: default` のみ。

| 区分 | 説明 |
|------|------|
| `subtask` | 環境カテゴリ（`env.py` レジストリ） |
| `algorithm` | `ppo`（`main.py` で runner 登録） |
| `run` | シード、ステップ数、ログ・動画・checkpoint |
| `hyperparameters` | PPO 固有（`n_envs`, `rollout_steps`, `clip_coef`, `ac_type` など） |

### 並列環境（VecEnv）

学習は Gymnasium `SyncVectorEnv` で **`hyperparameters.n_envs`** 本の `Pendulum-v1` を同時に進める。

| キー | 意味 |
|------|------|
| `n_envs` | 並列環境数（省略時 1） |
| `rollout_steps` | **各 env あたり**のベクトル step 数（SB3 の `n_steps` 相当） |
| （暗黙） | 1 回の PPO 更新あたりの遷移数 = `n_envs × rollout_steps` |

`run.max_env_steps` と `log_interval_steps` は **全 env の step 合計**（ベクトル 1 step で `+n_envs`）。Eval・動画は従来どおり **単一 env**。

`configs/default/` の比較は **1 軸だけ** baseline から変える（詳細は [`report/report.md`](report/report.md)）。

**収集の形** — 1 更新あたりの遷移数は `n_envs × rollout_steps = 4096` で揃える（baseline は 4×1024）。

| `run.name` | 意図 | `n_envs` | `rollout_steps` |
|------------|------|----------|-----------------|
| `baseline` | 中間 | 4 | 1024 |
| `more_n_envs` | 環境数↑・ロールアウト短 | 16 | 256 |
| `less_n_envs` | 環境数↓・ロールアウト長 | 1 | 4096 |

**更新の周回** — 収集は baseline と同じ 4×1024。

| `run.name` | `n_epochs` |
|------------|------------|
| `baseline` | 10 |
| `more_n_epochs` | 20 |
| `less_n_epochs` | 5 |

出力: `runs/<subtask>/<algorithm>/<run.name>/`

共通仕様: [`docs/implementation_notes/experiment_config.md`](../../docs/implementation_notes/experiment_config.md)

## コード構成

| ファイル | 役割 |
|---------|------|
| `main.py` | 設定読込・PPO 起動 |
| `train_ppo.py` | PPO 学習ループ |
| `policy.py` | deterministic 行動・Eval 平均リターン |
| `checkpoint.py` | `ac.pt` / `ac_best.pt` |
| `record.py` | 学習中のグリッド動画 |
| `models.py` | Actor-Critic（tanh ガウス方策） |
| `env.py` | `Pendulum-v1` |
| `logger.py` | TensorBoard |
| `console.py` | 標準出力 |

## 実行方法

```bash
uv run python tasks/02_Pendulum/main.py
uv run python tasks/02_Pendulum/main.py tasks/02_Pendulum/configs/default/baseline.yaml
```

## TensorBoard

```bash
uv run tensorboard --logdir tasks/02_Pendulum/runs
```

| タグ | 内容 |
|------|------|
| `Train/episode_reward` | 学習エピソードの累積報酬 |
| `Train/policy_loss` 等 | PPO 更新時 |
| `Eval/mean_episode_return` | deterministic 方策の平均エピソードリターン |

Eval / 動画のシードは `run.eval_seed_offset` / `run.video_seed_offset` で分離する。

## 設定の契約

* 既存 `run_dir` がある場合は学習を開始しない（`run.name` を変えるか削除）。
* `bootstrap_done` は **`terminated` のみ** True（`truncated` では価値をブートストラップ）。
* `video_interval_steps` は `log_interval_steps` の倍数。
