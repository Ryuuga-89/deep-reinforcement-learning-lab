# TensorBoard

## 概要

軽量な学習ロギングツール
元々TensorFlow用に作られていたが、現在ではPyTorchでも使える

## ロギング

まずSummaryWriterをインスタンス化する。これはディレクトリへのパスを引数にもち、指定されたパスへログを書き出す。

SummaryWriter.add_scalar()は、ログを書き込むために最も利用されるメソッドである。

```python
writer.add_scalar(
    "表示名",
    値,
    step,
)

writer.add_scalar("Reward/train", reward, episode)
writer.add_scalar("Loss/train", loss, step)
writer.add_scalar("LearningRate", lr, step)
```

ロギング終了時には、SummaryWriter.close()を行うことが望ましい。
あるいはwith文を利用する。

## 可視化

学習を実行すると、指定されたrunディレクトリに

```plaintext
runs/
└── cartpole/
    └── events.out.tfevents....
```

といったファイルが生成される。

この状態で

```bash
tensorboard --logdir runs
```

を実行すると、可視化をブラウザで確認できる。
