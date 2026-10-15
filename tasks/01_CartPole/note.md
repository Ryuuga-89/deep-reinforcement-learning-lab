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

1. 教科書的なハイパラの DQN（[`configs/default/baseline.yaml`](configs/default/baseline.yaml)）で、学習・評価・ログが一通り動くことを確認する。
2. 必要になったら `configs/<サブタスク>/` に別 YAML を追加し、**baseline から1軸だけ**変えて比較する（例: `hyperparameters.hidden_sizes`、`algorithm`、`subtask`）。

実験結果のメモは [`report/report.md`](report/report.md)。設定の正本はサブタスク別に次のように置く。

| ディレクトリ | 内容 |
|--------------|------|
| `configs/default/` | 通常環境（`subtask: default`）での DQN 派生・ハイパラ比較 |
| `configs/center/` | `subtask: center` |
| `configs/move/` | `subtask: move` |
| `configs/noise/` | `subtask: noise` |

## 実験設定の階層（YAML）

1 つの YAML が 1 回の学習 run を表す。トップレベルは次の区分（`subtask_options` は任意）。

| 区分 | YAML | 説明 |
|------|------|------|
| サブタスク | `subtask: <name>` | 環境のカテゴリ（実装は `env.py` のレジストリ） |
| サブタスクオプション | `subtask_options:` | サブタスク固有の環境パラメータ（`center` の `center_bonus_coef`、`move` の `move_alpha` / `move_beta`、`noise` の `noise_rho`） |
| アルゴリズム | `algorithm: <name>` | `src` に登録された RL アルゴリズム（本タスク `main.py` で runner 登録） |
| 実験設定 | `run:` | `name`（出力ディレクトリ名）、`seed`、`max_env_steps`、ログ・動画・checkpoint |
| ハイパーパラメータ | `hyperparameters:` | 選んだアルゴリズム固有（DQN なら γ、ε、ネットサイズなど） |

報酬の式はサブタスク実装に含める。`center` ではボーナス係数、`move` では係数と切替間隔を YAML で変えられる。

リポジトリ共通の説明: [`docs/implementation_notes/experiment_config.md`](../../docs/implementation_notes/experiment_config.md)

### 登録済みサブタスク（本タスク）

| `subtask` | 内容 |
|-----------|------|
| `default` | `CartPole-v1`、Gymnasium 既定の +1/ステップ報酬 |
| `sutton_barto` | `CartPole-v1`、`sutton_barto_reward=True` |
| `center` | `CartPole-v1`、報酬 `1 + c·(1 - min(|x|/x_threshold, 1))`（`subtask_options.center_bonus_coef` = \(c\)、省略時 1.0） |
| `move` | `CartPole-v1`、報酬 `1 + α·(1 - min(|x - x*|/x_threshold, 1))`。`x*` は軌道を 0–1 とした目標（0.25 → 左、0.75 → 右）。エピソード内ステップ `t` が `3β` 未満は 0.25、以降は `3β, 5β, 7β, ...` で 0.75 と 0.25 を交互に切替（`move_alpha` = α、省略時 1.0、`move_beta` = β、省略時 100、1–500） |
| `noise` | `CartPole-v1`、報酬は default と同じ。エージェントが受け取る観測のみ \(\tilde{o}_i = \mathrm{clip}(o_i + \epsilon_i)\)、\(\epsilon_i \sim \mathcal{N}(0, (\rho s_i)^2)\)（成分・ステップ独立）。\(s_i\) は `env.py` の `NOISE_OBS_REFERENCE_STD`（default 環境・ランダム方策で集めた観測の次元ごと std）。`noise_rho` = ρ、省略時 0.1。学習・Eval・動画はいずれもノイズ付き観測。 |

### 登録済みアルゴリズム（本タスク）

| `algorithm` | 学習 |
|-------------|------|
| `dqn` | `train_dqn.py` |
| `ddqn` | `train_dqn.py`（Double DQN） |
| `per_dqn` | `train_dqn.py`（DQN + PER） |
| `per_ddqn` | `train_dqn.py`（DDQN + PER） |

Dueling 構造は別アルゴリズム名ではなく、`hyperparameters.qnet_type: dueling`（省略時 `mlp`）で Q ネットを差し替える。詳細は [`docs/implementation_notes/experiment_config.md`](../../docs/implementation_notes/experiment_config.md)。

## コード構成

| ファイル | 役割 |
|---------|------|
| `main.py` | 設定読込・アルゴリズム registry 起動 |
| `train_dqn.py` | DQN 学習ループ |
| `policy.py` | greedy 行動・Eval 平均エピソード長 |
| `checkpoint.py` | `qnet.pt` / `qnet_best.pt` と best 追跡 |
| `record.py` | 学習中の greedy グリッド動画（imageio ストリーミング書き出し） |
| `console.py` | 標準出力の整形 |
| `models.py` | `qnet_type` と `hidden_sizes` から Q ネットを構築 |
| `env.py` | サブタスク名 → Gymnasium 環境 |
| `logger.py` | TensorBoard ログ |
| `configs/**/*.yaml` | `ExperimentConfig`（サブタスク別ディレクトリ） |

## 実行方法

```bash
# デフォルト（configs/default/baseline.yaml）
uv run python tasks/01_CartPole/main.py

# 設定ファイルを指定
uv run python tasks/01_CartPole/main.py tasks/01_CartPole/configs/default/large_model.yaml
```

## 設定例（`configs/default/baseline.yaml`）

```yaml
subtask: default
algorithm: dqn

run:
  name: baseline
  seed: 42
  max_env_steps: 200000
  ...

hyperparameters:
  gamma: 0.99
  hidden_sizes: [128, 128]
  ...
```

比較用 YAML（例: `configs/default/large_model.yaml`）では `run.name` と変更したいハイパーパラメータだけ baseline と差を付ける。

## TensorBoard

```bash
uv run tensorboard --logdir tasks/01_CartPole/runs
```

TensorBoard タグは **Train/**（学習中・ε-greedy）と **Eval/**（greedy 評価）で分ける。

| タグ | タイミング | 横軸 |
|------|------------|------|
| `Train/episode_reward` | 学習エピソード終了 | 環境ステップ |
| `Train/episode_length` | 同上 | 環境ステップ |
| `Train/epsilon` | 同上 | 環境ステップ |
| `Train/loss` | 勾配更新 | 勾配ステップ |
| `Eval/mean_episode_length` | `log_interval_steps` ごと＋終了時 | 環境ステップ |

**実験設定** — 学習開始時に `run_dir/experiment.yaml` を保存し、**TEXT** タブへ `Config/*` を書く。学習終了時に **HPARAMS** へ `add_hparams`（`run_name='.'` で同一 `run_dir`、子フォルダなし）と最終 `hparam/final_mean_episode_length` を記録する。複数 run の比較は `--logdir tasks/01_CartPole/runs` で行う（TensorBoard 公式の「1 trial = 1 ログディレクトリ」）。

## 設定の契約

* **`run_dir` の重複** — 出力先は `runs/<subtask>/<algorithm>/<run.name>/`（例: `runs/default/dqn/baseline/`）。このディレクトリが **既に存在する場合は学習を開始せず終了**する。再実行するには `run.name` を変えるか、既存ディレクトリを削除する。
* **学習の終了** — `run.max_env_steps` 環境ステップで打ち切り（エピソード数は可変）。
* **動画可視化** — `run.video_enabled: true` のときだけ、`video_interval_steps` ごとに `videos/train_step{環境ステップ:07d}.mp4` を出力する（`false` なら録画しない）。エンコードは **imageio** で合成フレームをメモリに溜めず逐次書き出す（ロールアウト分の RGB は録画時に保持）。
* **`video_interval_steps`** — `log_interval_steps` の倍数であること。`video_enabled: true` のときは正の整数が必須。
* 動画は **√N×√N グリッド**（既定 `video_episodes: 16` → 4×4）。各マスが 1 エピソード（greedy）。
* **マス内オーバーレイ** — 各マス右上にそのエピソードの環境ステップ（0 始まり）を大きく表示。全マス同時にカウントアップする。`terminated`（失敗）したマスは失敗フレームのステップで停止し、暗転後もその数字を表示する。`truncated`（上限到達）のマスは最終フレームのステップで停止する。
* **暗転** — `terminated` したマスはその瞬間から暗転し、他マスは続行する。上限 500 到達（`truncated`）のマスは暗転しない。
* **全マス `terminated` で終わったとき** — 最後に倒れたマスのあと **ホールド用フレーム**を付け、全マス暗転の状態で終了する（明るいマスが残って切れるのを防ぐ）。
* **Eval と可視化の環境シードは分離**する。`run.eval_rollout_seed(i)` / `run.video_rollout_seed(i)`（いずれも `seed + 各オフセット + episode`）。オフセットは異なり、シード列が重ならないこと。ローダー省略時の既定は `eval_seed_offset=10000`・`video_seed_offset=20000` だが、本タスクの baseline 系 YAML では `0` / `100` を明示している。チェックポイント間では同じオフセット系列を使い、学習ステップはシードに入れない。
* eval ログ・動画・標準出力の 1 行は同じ環境ステップで揃う。

## 再現性・学習の注意

* Eval / 動画の初期状態は `run.eval_rollout_seed(episode)` / `run.video_rollout_seed(episode)`（`seed + 各オフセット + episode`）で固定する。
* DQN の `done` は **倒れたとき（`terminated`）のみ** bootstrap を止める。時間切れ（`truncated`）はエピソードは終了するが TD ターゲットでは次状態を見る。
* `default` / `sutton_barto` / `center` / `move` などサブタスク間では報酬スケールが異なるため、TensorBoard の報酬は直接比較しない。
