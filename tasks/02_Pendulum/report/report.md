# 02 Pendulum

行動が連続のため、PPOを試してみる。
PPOを簡単に回してみるという目的で、複数seedによる厳密な実験などはしない。

PPOアルゴリズムの整理は [note.md](../../../src/deep_reinforcement_learning_lab/algorithms/PPO/note.md) に記載している。

## 実験の内容

CartPoleと同様、yamlファイルで全設定を管理する。

ただしサブタスクは `default`（Gymnasium `Pendulum-v1`）のみ。環境の説明は [note.md](../note.md)。

実験は **サブタスク → アルゴリズム（`ppo`）→ `run.name`** の順で `runs/` に出力される。

学習は `SyncVectorEnv` で並列環境を回す。Eval と Visualize は **単一環境**で、学習中の方策を **決定論的**（ガウス平均 → tanh）に動かす。

### Eval

`run.eval_episodes` 本（baseline では 100）のエピソードを、`seed + eval_seed_offset + episode` で `reset` し、各エピソードの **ステップ報酬の合計**を取る。その平均を `Eval/mean_episode_return` として TensorBoard に記録する。

報酬は環境の負のコストの和なので、**理論上の最大は 0**（直立・静止・トルク 0）。**大きいほどよい**（0 に近いほどよい）。1 エピソードは最大 200 ステップで終了する。

学習ログの `Train/episode_reward` は VecEnv 上の **確率方策**の 1 本のエピソードであり、Eval とは別物なので混同しない。

### Visualize

CartPole と同様、指定ステップごとに複数エピソードをグリッド動画として保存する（baseline では 16 本）。シード列は eval と `video_seed_offset` で分離している。

### 比較実験（1 軸だけ変える）

いずれも baseline から **1 項目だけ**変えた設定を `configs/default/` に置く。

**収集の形**（1 更新あたりの遷移数は `n_envs × rollout_steps = 4096` で揃える）

| `run.name` | `n_envs` | `rollout_steps` |
|------------|----------|-----------------|
| `baseline` | 4 | 1024 |
| `more_n_envs` | 16 | 256 |
| `less_n_envs` | 1 | 4096 |

**更新の周回**（収集は baseline と同じ 4×1024）

| `run.name` | `n_epochs` |
|------------|------------|
| `baseline` | 10 |
| `more_n_epochs` | 20 |
| `less_n_epochs` | 5 |

## 実験メモ

全ての結果は

```bash
uv run tensorboard --logdir tasks/02_Pendulum/runs
```

で確認可能。

### 1. default

#### 目的

PPO を触ってどんな感じか見てみる。連続行動・VecEnv・ロールアウト長と `n_epochs` の感覚をつかむ。

#### ベースライン

まず [baseline](../configs/default/baseline.yaml) を実行する。Actor-Critic は共有 MLP（64×64）、`max_env_steps: 200000`、ログ・Eval・動画は 16384 環境ステップごと。

#### 収集の比較（メモ）

- `more_n_envs` / `less_n_envs` を baseline と並べて `Eval/mean_episode_return` を見る。
- サンプル数は同じでも、短いロールアウト多数本 vs 1 本の長いロールアウトでは **データの時間相関**が違う点に注意。

less_n_envsでは学習が遅く、安定しない状況が見られた。
ただ最終的にはEvalは全て同じような水準にまで達しており、問題なく学習が行われているとみることができる。

lossを見ると、環境数が多い方が明確にlossが0に近い(他は負)。

また、DQNよりEvalが安定している。DQNで見られたEvalの乱高下がなく、クリップ方策が適切に機能していることがわかる。

#### エポック数の比較（メモ）

- `more_n_epochs` / `less_n_epochs` を同様に比較する。
- 同じロールアウトを何周学習するかの違い。Train の loss だけ下がって Eval が伸びないパターンが出たらメモする。

epochが大きいrunにおいては、Evalは他のrunの水準まで到達しなかった。収集データに対する過学習が、逆に学習を阻害している。
