# 03 LunarLander

同一の連続物理環境（`LunarLanderContinuous-v3`）上で、離散 4 択方策と連続 2 次元方策の PPO を比較する。
本タスクでは **複数 seed（既定 42–51、N=10）** で学習し、統計的に妥当性を評価する。

## 設定の正本

`subtask` は **`default`**。方策は **`model.id`**（連続 / テーブル離散 / 因子分解離散）。出力は `run.name` で分離する。

| 比較軸 | 推奨 `model.id` |
|--------|-----------------|
| 連続スロットル | `mlp_actor_critic_continuous` |
| 独立ヘッド離散（粒度は `head_levels`） | `mlp_factorized_discrete` |
| 旧 Gymnasium 4 択（joint K） | `mlp_actor_critic_discrete` + `options.table: gymnasium_4` |

| 用途 | 離散 | 連続 |
|------|------|------|
| 単発（動画） | [`baseline_discrete_preview.yaml`](../configs/default/baseline_discrete_preview.yaml) | [`baseline_continuous_preview.yaml`](../configs/default/baseline_continuous_preview.yaml) |
| 統計用学習 | [`baseline_discrete_train.yaml`](../configs/default/baseline_discrete_train.yaml) | [`baseline_continuous_train.yaml`](../configs/default/baseline_continuous_train.yaml) |
| multiseed | [`baseline_discrete_multiseed.yaml`](../configs/default/baseline_discrete_multiseed.yaml) | [`baseline_continuous_multiseed.yaml`](../configs/default/baseline_continuous_multiseed.yaml) |

環境・モデル契約は [note.md](../note.md)。

## 統計の流れ

1. `run_seeds.py` で `runs/default/ppo/baseline_discrete/seed_XXXX/`（連続は `baseline_continuous`）を生成
2. 各 seed に `eval_final.json`（`model_id` / `policy_kind` 付き）
3. `analyze_runs.py --run-dir runs/default/ppo/baseline_discrete`
4. `--compare runs/default/ppo/baseline_continuous` で paired 比較

### 主指標

- **成功率** `success_rate`: Eval エピソードの累積報酬 ≥ 200 の割合（Gymnasium solved）
- 補助: `mean_episode_return`、終了理由 3 分類、着陸成功時の燃料・姿勢

## 実験メモ

### 離散 vs 連続

離散 4 択モデル（`mlp_actor_critic_discrete`）と連続モデル（`mlp_actor_critic_continuous`）の学習安定性を比較する。物理 sim と env の action space は共通（連続 2 次元）。

（新実験結果はここに追記。）
