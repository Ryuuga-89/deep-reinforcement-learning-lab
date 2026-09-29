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

## リポジトリ構造

```plaintext
deep-reinforcement-learning-lab
├─ AGENTS.md
├─ REDAME.md
├─ src/ 再利用するコードベース
├─ docs/
│  ├─ study.md 学びをメモする
│  └─ implementation_notes/ 実装に関するメモ
├─ tasks/ 各タスク
│  ├─ 01_CartPole/
│  ├─ 02_Penduram/
│  ├─ 03_LunarLander/
│  └─ 04_2D_Dungion/
└─ experiments/ ライブラリの動作実験など
```