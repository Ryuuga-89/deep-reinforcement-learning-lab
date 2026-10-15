# Deep Reinforcement Learning Lab

## 概要

深層強化学習についての理論的学習を受けて、実際に手を動かして実装、実験するためのリポジトリ。
研究開発用というよりは、いくつかの古典的なタスクに対して実装や実験を行い、学びを得るのが目的となる。

## 目標

* Gymnasiumライブラリを利用して、強化学習における「環境」を理解する。
* 強化学習のアルゴリズムとして、DQN / PPO をそれぞれ実装する。
* PyTorchでモデルを記述し、強化学習を実施してみる。
* 色々と試行錯誤して、強化学習について知る。

## 対象となるタスク

1. 倒立振子(CartPole)
2. 振子(Pendulum)
3. 月面着陸(LunarLander)
4. 2Dダンジョン(2D_Dungeon)

1-3はGymnasiumライブラリの実装を利用する。
4は自作する。

## 方針

* AIコーディングエージェントを利用する。ただしコードの内容を自分で理解できるように、AGENTS.mdにて「指示したスコープ外の変更を行わないこと」「一度の変更で変更するファイルは1つまで」と制限する。
* 深層強化学習の実装に関しては、環境はGymnasium / NN周りはPyTorch を利用する。その他の高級ライブラリは基本的に利用しない。
* 環境管理にはuvを利用する。

## 実験成果物とレポート

学習の出力と GitHub 上で共有する資料は分ける。

| 場所 | 内容 | Git |
|------|------|-----|
| `tasks/<task>/configs/` | 実験の正本（YAML）。再実行はここを指定する | コミットする |
| `tasks/<task>/runs/` | 学習の一次出力（TensorBoard イベント、チェックポイント、学習中に生成した動画など） | **管理しない**（`.gitignore` で除外） |
| `tasks/<task>/report/` | 実験メモ（`report.md`）と、レポートから参照する画像・動画（例: `report/assets/`） | コミットする（メディアは必要な分だけ） |

* `runs/` はローカル（または各自の環境）での分析用。容量が大きく、設定から再生成できるものはリポジトリに載せない。
* レポートで見せたい図や動画は、`runs/` から必要なファイルだけ `report/` 配下にコピーし、`report.md` は **そのパスだけ** を参照する（`runs/` への相対リンクは使わない）。
* 再現の記述は `configs/` の YAML と `run.name`（および実行コマンド）を書く。`runs/` 内の `experiment.yaml` は実行時スナップショットであり、共有の正本ではない。
* `experiments/` 配下の試行用出力も同様に `runs/` や `videos/` は Git 管理外とする。

## リポジトリ構造

```plaintext
deep-reinforcement-learning-lab
├─ AGENTS.md
├─ README.md
├─ src/ 再利用するコードベース
├─ docs/
│  └─ implementation_notes/ 実装に関するメモ
├─ tasks/ 各タスク（configs / runs / report など）
│  └─ 01_CartPole/  （他タスクは順次追加）
└─ experiments/ ライブラリの動作実験など
```