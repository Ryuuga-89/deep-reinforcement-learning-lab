# 03 LunarLander

## タスクの内容

月面着陸船を着陸パッド上に安全に着地させる（Gymnasium Box2D）。

### 観測空間

長さ 8 の連続ベクトル（`Box(8,)`）。離散・連続サブタスクで同一。

| インデックス | 意味 |
|:---:|:---|
| 0 | x 位置 |
| 1 | y 位置（高度） |
| 2 | x 速度 |
| 3 | y 速度 |
| 4 | 船体角度 |
| 5 | 角速度 |
| 6 | 左脚接地（0/1） |
| 7 | 右脚接地（0/1） |

### 物理環境と行動

**`subtask` は常に `default`。** 物理 sim は常に `LunarLanderContinuous-v3`。環境が受け取る行動は常に **`Box(2)`**（主・横スロットル）。

離散か連続かは **モデル（`model.id`）** が決める。env は常に `Box(2)` のみ。

離散か連続かは **`model.id` / `model.options` の組み合わせごとに別条件**として試す。現行 id は次の 3 つ。

* **`lunar_binned`** — env 次元ごとの独立ヘッド（`head_levels` または `preset`）。`log π` はヘッド和。[`factorized_heads.py`](factorized_heads.py), [`factorized_ppo.py`](factorized_ppo.py)
* **`lunar_table`** — 単一 `Categorical(K)` + [`lunar_action_tables.py`](lunar_action_tables.py) の K 点写像（`options.table`）。用意済みは `gymnasium_4`、`main_lateral_6`、`main5_lat3`、`main_on_off`、`lateral_three`
* **`lunar_continuous`** — tanh-Gaussian。`baseline_continuous` など `run.name` で識別

旧チェックポイントの `mlp_actor_critic_*` / `mlp_factorized_discrete` は読込時に上の id へ解決する。

成功の目安: 1 エピソードの累積報酬が **200 以上**（Gymnasium `reward_threshold`）。

## 実験する内容

1. [`configs/default/baseline_*_preview.yaml`](configs/default/) で単発・動画付きの動作確認。
2. 統計用は `*_train.yaml`（トップレベル `seeds:` 付き）。**条件ごとに `run.name` を変える**（例: `table_gymnasium_4`、`table_main_lateral_6`、`table_main5_lat3`）。
3. 比較用 YAML は既存 train をコピーし、**1 軸だけ**変える（`run.name` も必ず変える）。PPO の **`entropy_coef` は条件間で揃える**（default baseline は `0.01`）。

実験メモは [`report/report.md`](report/report.md)。

## 実験設定の階層（YAML）

CartPole と同様。

| 区分 | 説明 |
|------|------|
| `subtask` | `default`（LunarLander タスク固定） |
| `algorithm` | `ppo` |
| `model` | **必須** — `id`（`lunar_table` / `lunar_binned` / `lunar_continuous`）。登録は [`ac_model_registry.py`](ac_model_registry.py) |
| `run` | シード、ステップ数、ログ・動画・checkpoint |
| `hyperparameters` | PPO 固有（`ac_type` / `hidden_sizes` など） |

`model` 例:

```yaml
model:
  id: lunar_binned
  options:
    preset: binary_pair   # 省略可。主・横それぞれ 2 値
    # head_levels:        # preset を上書きする場合
    #   - [-1, 1]
    #   - [-1, 0, 1]      # gymnasium_style preset の既定
```

`preset: gymnasium_style` は主 2 値 × 横 3 値（**独立**なので旧 4 択と同じ joint にはならない）。

新モデル追加: `register_model(..., ppo_kind="discrete"|"continuous"|"factorized")`。環境へは `rollout_actions_to_env` / `deterministic_action` が **常に `float32[2]`**。

### 学習ループの違い（モデル id に紐づく `ppo_kind`）

| `ppo_kind` | 並列 | 備考 |
|------------|------|------|
| `factorized` | `n_envs` ≥ 1 | `FactorizedPPO` + ヘッドごと独立 Categorical（ロールアウトは `(n_heads,)` インデックス） |
| `discrete` | `n_envs` ≥ 1 | 単一 `Categorical(K)` + テーブル写像（比較用） |
| `continuous` | `n_envs` ≥ 1 | `ContinuousPPO` + `VectorContinuousFixedRolloutBuffer` |

Eval・動画は **単一連続 env** の deterministic 方策（モデルの `deterministic_action` → `Box(2)`）。

出力: `runs/<subtask>/<algorithm>/<run.name>/seed_XXXX/`（`XXXX` は 4 桁の学習 seed）

共通仕様: [`docs/implementation_notes/experiment_config.md`](../../docs/implementation_notes/experiment_config.md)

## 依存

Box2D は `uv sync` で `gymnasium[box2d]` として入る。ビルドに `swig` が必要な環境では事前インストールが必要な場合がある。

## コード構成

| ファイル | 役割 |
|---------|------|
| `main.py` | 設定読込・PPO 起動 |
| `train_ppo.py` | 離散 / 連続の PPO 学習ループ |
| `policy.py` | deterministic 行動・Eval 平均リターン |
| `checkpoint.py` | `ac.pt` / `ac_best.pt`（`policy_kind` 付き） |
| `record.py` | 学習中のグリッド動画 |
| `config_loader.py` | `model` ブロック + 共通 `ExperimentConfig` 読込 |
| `model_config.py` | `ModelConfig` / `model.id` パース |
| `ac_model_registry.py` | `model.id` → ビルダ登録 |
| `models.py` | 離散 / 連続 AC + 離散→連続写像 |
| `env.py` | 連続 LunarLander のみ |
| `logger.py` | TensorBoard |
| `console.py` | 標準出力 |
| `run_paths.py` | `seed_XXXX` 付き `run_dir` 解決 |
| `run_seeds.py` | 複数 seed 学習スイープ |
| `eval_export.py` | 学習終了時 `eval_final.json` |
| `analyze_runs.py` | seed 間の集約・95% CI・任意の paired 比較 |
| `plot_exp1.py` / `plot_exp2.py` | レポート用の成功率図 |
| `export_exp1_rollout_gif.py` / `export_exp2_rollout_gif.py` | seed 42 の比較 GIF |

## 実行方法

```bash
# 単発（動画あり preview）。引数なしは baseline_continuous_preview
uv run python tasks/03_LunarLander/main.py
uv run python tasks/03_LunarLander/main.py \
  tasks/03_LunarLander/configs/default/table_gymnasium_4_preview.yaml

# 統計用: seed 42–51（動画オフ train YAML の seeds: を一括実行）
uv run python tasks/03_LunarLander/run_seeds.py \
  tasks/03_LunarLander/configs/default/baseline_continuous_train.yaml
uv run python tasks/03_LunarLander/run_seeds.py \
  tasks/03_LunarLander/configs/default/table_gymnasium_4_train.yaml

# 途中再開
uv run python tasks/03_LunarLander/run_seeds.py \
  tasks/03_LunarLander/configs/default/table_gymnasium_4_train.yaml --skip-existing

# 集約（主張用の比較は success_rate。--holm は出した指標全体を補正する）
uv run python tasks/03_LunarLander/analyze_runs.py \
  --run-dir tasks/03_LunarLander/runs/default/ppo/table_gymnasium_4 \
  --compare tasks/03_LunarLander/runs/default/ppo/table_main_lateral_6

```

### 設定ファイルの使い分け

| ファイル | 用途 | 出力 `run.name` |
|----------|------|-----------------|
| `*_preview.yaml` | 単発・動画あり | `*_preview` |
| `*_train.yaml` | 統計用（動画オフ・`seeds:` で multiseed） | YAML の `run.name` |

preview は **`run.name` が `*_preview`** のため、train の出力と混ざらない。`main.py` は `run.seed` のみ使用（`seeds` は無視）。

## TensorBoard

```bash
uv run tensorboard --logdir tasks/03_LunarLander/runs
```

| タグ | 内容 |
|------|------|
| `Train/episode_reward` | 学習中のエピソード累積報酬 |
| `Eval/success_rate` | 累積報酬 ≥ 200 のエピソード割合（Gymnasium solved 基準） |
| `Eval/land_success_rate` / `crash_rate` / `timeout_rate` | 終了理由 3 分類（着陸 +100 / クラッシュ −100 / 1000 ステップ切れ） |
| `Eval/mean_episode_return` | deterministic Eval の平均リターン（補助） |
| `Eval/mean_main_fuel_on_land` / `mean_side_fuel_on_land` | **着陸成功エピソードのみ**の燃料ペナルティ累積（環境と同じ 0.30×主 / 0.03×横） |
| `Eval/mean_abs_tilt_rad_on_land` | 着陸成功時の \|角度\| [rad]（観測 index 4） |
| `Eval/mean_abs_angular_speed_on_land` | 着陸成功時の \|角速度\|（観測 index 5、環境のスケール済み） |
| `Train/policy_loss` など | PPO 更新（勾配ステップ） |

best チェックポイントは **成功率 → 同率なら平均リターン** の順で更新する。

## 設定の契約

* **旧 run** — `runs/discrete/` / `runs/continuous/` 配下は旧設定のログ。現在は `runs/default/ppo/<run.name>/` と `model.id` を正本とする。
* **`run_dir` の重複** — `.../seed_XXXX/` が既にある場合は学習を開始しない（`run_seeds.py` は `--skip-existing` でスキップ可）。
* **`eval_final.json`** — 各 seed の **best ckpt 再評価**メトリクス（`eval_source: best_checkpoint`）。seed 間統計の正本。
* **GAE の `bootstrap_done`** — `terminated` のみ（墜落）。`truncated`（1000 ステップ上限）はエピソード終了だが bootstrap は次状態を見る。
* **動画** — 各マス左上に **累積燃料ペナルティ**（`fuel 12.34`＝環境と同じ 0.30×主＋0.03×横の合計）、右上に環境ステップ。`terminated`（墜落）マスは暗転。着陸成功は暗転しない。`truncated` は暗転しない。
* **Eval / 動画シード** — `eval_seed_offset` / `video_seed_offset`（baseline では `0` / `100`）。
