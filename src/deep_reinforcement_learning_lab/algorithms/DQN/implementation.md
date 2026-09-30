# DQN - Implementation

## 原則

学習時にのみ利用され
1. モデルを用いたサンプリング
2. リプレイバッファを用いたリプレイデータの保持
3. タイミングに応じたバッチデータ作成
4. メインネットワークの学習
5. ターゲットネットワークの同期
を担当する。

一方で推論時には一切登場しない。

DQNは環境を関知しない。
また特定のモデルに依存しない。

テンソル・モデルはすべて CPU 上で扱う。

## クラス定義

以下のようなクラスとして定義する。

インスタンス時の引数

* q_model(nn.Module): 対象のモデル。入力次元と出力次元は環境に準拠する。ターゲットネットワークはインスタンス化時にこれをコピーして内部で保持する（学習ループ側は渡さない）。
* optimizer(torch.optim.Optimizer): 最適化器
* replay_buffer(ReplayBuffer, 自作クラス): リプレイバッファ
* gamma(float): 割引率
* epsilon_start(float): イプシロンの初期値
* epsilon_end(float): イプシロンの終了値
* epsilon_rate: イプシロンの減衰率。環境ステップ `t` ごとに `ε = max(ε_end, ε_start - rate * t)`。
* batch_size(int): バッチサイズ
* target_update_interval_step(int): ハード更新時、ターゲットを同期する間隔（**勾配更新回数**）。
* tau(float, optional): ターゲットネットワークの更新のための混合率。定義時は `update()` のたびにソフト更新。未定義時はハード更新（上記インターバルごとに上書き）。
* loss_type(str, optional): 損失関数（MSE / Huber など、切り替え可能）。

主要メソッド

* select_action(self, state) -> int: 現在の1状態から行動を1つサンプリング（環境1ステップにつき1回）。`state` は `torch.Tensor`（`[*state_shape]` または `[1, *state_shape]` のみ）。出力値は行動の index。
* step(self, state, action, reward, next_state, done) -> None: 外部ループから渡された1ステップ分の遷移を処理する。バッファへの追加やステップカウンタの更新など。末尾で `len(buffer) >= batch_size` のとき `update()` を呼ぶ。
* update(self) -> float: TD誤差を計算して、メインネットワークの重みを勾配降下法で1回更新する（戻り値はバッチ平均 loss）。またこの内部で必要に応じて `_update_target_network()` を呼び出す。
* _update_target_network(self) -> None: ターゲットネットワークの更新を実施

### ReplayBuffer

リプレイバッファを定義するための抽象クラス。
全ての具体的なバッファはこれを継承するものとする。

インスタンス時の引数

* buffer_size(int): バッファサイズ

主要メソッド(全て抽象メソッドで、具体クラスで実装する)

* push(self, state, action, reward, next_state, done) -> None: 1ステップ分の遷移データをバッファに追加する。
* sample(self, batch_size) -> tuple[torch.tensor]: 指定されたバッチサイズのデータをサンプリングしてタプルにまとめて返す。それぞれのデータはtorch.tensorでまとめられる。返却順の推奨: `(states, actions, rewards, next_states, dones)`。`actions` は `torch.long`、`dones` は `(1 - done)` 用に `float` など、実装で dtype を揃える。
* \_\_len\_\_(self) -> int: 格納されているデータ数を返す
* buffer_size: プロパティ

#### UniformReplayBuffer

`ReplayBuffer` の具体実装。満杯時は古い遷移から FIFO で破棄するリングバッファ。
