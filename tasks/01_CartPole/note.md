# 01 CartPole

## タスクの内容

倒立振子を立て続ける。

### 観測空間

`CartPole-v1` の観測は長さ 4 の連続ベクトル（`Box(4,)`）。`reset` / `step` の戻り値 `observation` がそのまま入る。

| インデックス | 名前 | 意味 | エピソード継続の目安 |
|:---:|:---|:---|:---|
| 0 | Cart Position | レール上のカート位置（右が正） | \|x\| < 2.4 |
| 1 | Cart Velocity | カートの水平速度（右が正） | — |
| 2 | Pole Angle | 垂直からの棒の傾き [rad]（右倒れが正） | \|θ\| < 0.2095 rad（約 ±12°） |
| 3 | Pole Angular Velocity | 棒の角速度 [rad/s]（角度が増える方向が正） | — |

符号の向き（右方向を正とする）:

* 0: 正 → カートが右、負 → 左
* 1: 正 → 右へ移動、負 → 左へ移動
* 2: 正 → 棒が右に倒れかけ、負 → 左に倒れかけ（直立時は 0 付近）
* 3: 正 → 角度が増加方向に回転、負 → 減少方向

補足:

* `reset()` 直後は各要素が一様乱数でおおよそ ±0.05（わずかにずれた初期状態）
* `observation_space` の上下限（例: 位置 ±4.8、角度 ±24°）は表現範囲であり、上表の失敗条件とは一致しない

### 行動空間

`Discrete(2)` — 各ステップで **左か右の2択** のみ。力の大きさは選べず、方向だけ決める。

| 行動 | 意味 |
|:---:|:---|
| 0 | カートを **左** に押す（固定の大きさの力） |
| 1 | カートを **右** に押す（固定の大きさの力） |

補足:

* Gymnasium 内部では力の大きさ `force_mag = 10.0`（環境パラメータ。学習時に通常は変更しない）
* `env.step(action)` には `0` または `1` の整数（または `action_space.sample()` 相当）を渡す
* 棒そのものには直接力を加えられない。カートを動かして間接的に棒を立てる

## 実験する内容

1. 教科書的なハイパラの DQN（`configs/baseline.yaml`）で、学習・評価・ログが一通り動くことを確認する。
2. 必要になったら `configs/` に別 YAML を追加し、**baseline から1軸だけ**変えて比較する（学習率、モデルサイズ、報酬など）。

## コード構成

| ファイル | 役割 |
|---------|------|
| `main.py` | 設定読込・学習ループのエントリ |
| `policy.py` | greedy 行動・Eval 平均報酬 |
| `checkpoint.py` | `qnet.pt` / `qnet_best.pt` と best 追跡 |
| `record.py` | 学習中の greedy 動画 |
| `console.py` | 標準出力の整形 |
| `models.py` | `hidden_sizes` から Q ネットを構築 |
| `env.py` | `reward_mode` に応じた Gymnasium 環境 |
| `logger.py` | TensorBoard ログ |
| `configs/baseline.yaml` | 既定の `TrainConfig` |

## 実行方法

```bash
# デフォルト（baseline.yaml）
uv run python tasks/01_CartPole/main.py

# 明示的に baseline を指定
uv run python tasks/01_CartPole/main.py tasks/01_CartPole/configs/baseline.yaml
```

## 設定ファイル（`configs/baseline.yaml`）

比較実験用の追加 YAML はいったん置かず、**baseline だけ**を基準とする。

| 項目 | 値 | 補足 |
|------|-----|------|
| `run_name` | `baseline` | ログは `runs/baseline/` |
| `max_env_steps` | 200000 | 環境ステップで学習終了 |
| `gamma` | 0.99 | |
| `learning_rate` | 0.001 | Adam |
| `buffer_size` / `batch_size` | 50000 / 128 | |
| `epsilon_start` / `epsilon_end` | 1.0 / 0.01 | ε-greedy |
| `epsilon_decay_steps` | 20000 | 総ステップの 10% で ε を 0.01 まで線形減衰 |
| `target_update_interval` | 1000 | **勾配更新**ごとのターゲット Q 同期 |
| `hidden_sizes` | [128, 128] | 2 層 MLP + ReLU |
| `loss_type` | `huber` | Smooth L1 |
| `reward_mode` | `default` | Gymnasium 既定の +1/ステップ |
| `log_interval_steps` | 10000 | eval・標準出力 |
| `video_interval_steps` | 20000 | `log_interval_steps` の倍数 |

`run_name` を変えれば TensorBoard のサブディレクトリも変わる（`runs/<run_name>/`）。比較用に新しい YAML を足すときは、上記以外は baseline と揃え、変えるキーだけを書き換える。

## TensorBoard

```bash
uv run tensorboard --logdir tasks/01_CartPole/runs
```

主なスカラー（横軸は原則 **環境ステップ**）: `Reward/episode`（エピソード終了時）、`Reward/eval`（`log_interval_steps` ごと）、`Loss/train`（勾配更新）、`Exploration/epsilon`。

## 設定の契約

* **学習の終了** — `max_env_steps` 環境ステップで打ち切り（エピソード数は可変）。
* **`video_interval_steps` が 0 より大きいとき** — `log_interval_steps` の倍数であること。
* このとき eval ログ・動画・標準出力の 1 行は同じ環境ステップで揃う。
* **`video_interval_steps: 0`** — 録画なし。

## 再現性・学習の注意

* エピソード `i` の初期状態は `reset(seed=seed + i)` で固定している（`i` は 1 始まり）。
* DQN の `done` は **倒れたとき（`terminated`）のみ** bootstrap を止める。時間切れ（`truncated`）はエピソードは終了するが TD ターゲットでは次状態を見る。
* `env.py` は `reward_mode: sutton_barto` も選べるが、baseline は `default`。報酬モードを変えた run 同士では TensorBoard の報酬を直接比較しない。
