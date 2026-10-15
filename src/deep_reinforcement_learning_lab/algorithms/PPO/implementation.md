# PPO - Implementation

## 原則

学習時にのみ利用され、

1. Actor-Critic による行動サンプリングと価値推定
2. ロールアウトバッファによるオンポリシーデータの保持
3. GAE によるアドバンテージ計算
4. クリップ付き方策損失・クリップ付き価値損失・エントロピーによる更新

を担当する。推論時（評価・録画）はタスク側が Actor の greedy 方策を使う。

PPO は環境を関知しない。特定のモデル構造にも依存せず、`action_logits` / `state_value` を提供する `nn.Module` を受け取る。

テンソル・モデルはすべて CPU 上で扱う。

## クラス `PPO`

### コンストラクタ

- `ac_model`, `optimizer`, `rollout_buffer`
- `gamma`, `gae_lambda`, `clip_coef`, `entropy_coef`, `value_coef`
- `rollout_steps`, `n_epochs`, `minibatch_size`
- `clip_value_coef`（省略時は `clip_coef` と同値）
- `max_grad_norm`（既定 0.5）

### 主要メソッド

- `select_action(state) -> int` — Categorical サンプリング。`log_prob` と `value_old` を内部保持。
- `step(state, action, reward, next_state, bootstrap_done) -> RolloutLosses | None` — 1 遷移をバッファへ。満杯なら GAE + `update()`。
- `finish_rollout(last_state)` — 末尾ブートストラップと advantage / return 算出。advantage はバッファ全体で正規化。
- `update() -> RolloutLosses` — 複数エポックのミニバッチ学習後バッファを clear。
- `flush_if_pending(last_state)` — 学習終端の部分ロールアウトを処理。

`bootstrap_done` は **terminated のみ** True（truncated では価値ブートストラップを継続）。

### 損失

- 方策: clipped surrogate（PPO）
- 価値: **クリップ付き MSE（PPO2 既定）** — 収集時の `values_old` を基準にクリップ
- エントロピー: 最大化（損失は負の平均エントロピー）

拡張は `_policy_loss` / `_value_loss` / `_compute_gae` のオーバーライド。

## `RolloutBuffer`

抽象クラス。具体実装 `FixedRolloutBuffer` は固定 `rollout_steps` 容量（部分ロールアウトも `len < rollout_steps` で GAE 可能）。
