# 03 LunarLander

PPOを利用した強化学習について、手法やモデルによる性能差を統計的に分析する。
今回は複数 seed による分析を実施する。

## 統計の扱い（最低限の体裁）

- **主指標**: `success_rate`（eval 累積報酬 ≥ 200 の割合）。事前にここだけを主張の根拠にする。
- **集約**: `analyze_runs.py` が seed ごとの `eval_final.json` から平均・SD・95% CI（t）を出す。
- **2 条件比較**: 共通 seed 上の **paired t 検定**、差分の 95% CI、Cohen’s d_z。副指標は探索的；必要なら `--holm` で Holm 補正付き p を併記。
- **評価**: multiseed 比較の正本は **best チェックポイント（`ac_best.pt`）の再評価**（`eval_final.json` の `eval_source: best_checkpoint`）。seed 数は多いほどよい（目安 n≥10）。
- **ハイパラ**: 条件比較では `entropy_coef` などを揃える（repo の baseline 連続・離散は **`0.01`**）。離散はパターンごとに `run.name` を分け、単一の「代表離散」は置かない。

```bash
# 例: ある離散条件（run.name）と連続 baseline の paired 比較
uv run python tasks/03_LunarLander/analyze_runs.py \
  --run-dir tasks/03_LunarLander/runs/default/ppo/<discrete_run_name> \
  --compare tasks/03_LunarLander/runs/default/ppo/baseline_continuous \
  --holm
```

## 実験メモ

### 出力の離散化による性能・学習の変化

行動空間が連続の時、あえてモデルの選択肢を減らすアプローチが考えられる。
これはモデルの出力範囲を制限することで、学習をより早く進行させることを狙っている。

これによってどのように性能が変化するかと、どのように学習が変化するかを分析する。

分析対象は次の3つ
* 連続: 通常通り、連続領域に対する出力を行う
* テーブル: あらかじめ行動テーブルを用意しておき、そのどれを行動とするかを分類タスクで決める
* ビニング: 連続であった出力範囲を離散化し、出力させる。

出力モジュール以外のモデル構造と、ハイパーパラメータ、環境シードは一致させる。

