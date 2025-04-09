<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Trainer

[`Trainer`] クラスは、ほとんどの標準的なユースケースに対して、PyTorch で機能を完全にトレーニングするための API を提供します。これは、[サンプル スクリプト](https://github.com/huggingface/transformers/tree/main/examples) のほとんどで使用されています。

[`Trainer`] をインスタンス化する前に、トレーニング中にカスタマイズのすべてのポイントにアクセスするために [`TrainingArguments`] を作成します。

この API は、複数の GPU/TPU での分散トレーニング、[NVIDIA Apex](https://github.com/NVIDIA/apex) および PyTorch のネイティブ AMP による混合精度をサポートします。

[`Trainer`] には、上記の機能をサポートする基本的なトレーニング ループが含まれています。カスタム動作を挿入するには、それらをサブクラス化し、次のメソッドをオーバーライドします。

- **get_train_dataloader** -- トレーニング データローダーを作成します。
- **get_eval_dataloader** -- 評価用データローダーを作成します。
- **get_test_dataloader** -- テスト データローダーを作成します。
- **log** -- トレーニングを監視しているさまざまなオブジェクトに関する情報をログに記録します。
- **create_optimizer_and_scheduler** -- オプティマイザと学習率スケジューラが渡されなかった場合にセットアップします。
  初期化。 `create_optimizer`メソッドと`create_scheduler`メソッドをサブクラス化またはオーバーライドすることもできることに注意してください。
  別々に。
- **create_optimizer** -- init で渡されなかった場合にオプティマイザーをセットアップします。
- **create_scheduler** -- init で渡されなかった場合、学習率スケジューラを設定します。
- **compute_loss** - トレーニング入力のバッチの損失を計算します。
- **training_step** -- トレーニング ステップを実行します。
- **prediction_step** -- 評価/テスト ステップを実行します。
- **evaluate** -- 評価ループを実行し、メトリクスを返します。
- **predict** -- テスト セットの予測 (ラベルが使用可能な場合はメトリクスも含む) を返します。

<Tip warning={true}>

[`Trainer`] クラスは 🤗 Transformers モデル用に最適化されており、驚くべき動作をする可能性があります
他の機種で使用する場合。独自のモデルで使用する場合は、次の点を確認してください。

- モデルは常に [`~utils.ModelOutput`] のタプルまたはサブクラスを返します。
- `labels` 引数が指定され、その損失が最初の値として返される場合、モデルは損失を計算できます。
  タプルの要素 (モデルがタプルを返す場合)
- モデルは複数のラベル引数を受け入れることができます ([`TrainingArguments`] で `label_names` を使用して、その名前を [`Trainer`] に示します) が、それらのいずれにも `"label"` という名前を付ける必要はありません。

</Tip>

以下は、加重損失を使用するように [`Trainer`] をカスタマイズする方法の例です (不均衡なトレーニング セットがある場合に役立ちます)。

```python
from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

PyTorch [`Trainer`] のトレーニング ループの動作をカスタマイズするもう 1 つの方法は、トレーニング ループの状態を検査できる [callbacks](コールバック) を使用することです (進行状況レポート、TensorBoard または他の ML プラットフォームでのログ記録など)。決定（早期停止など）。

## Trainer

[[autodoc]] Trainer
    - all

## Seq2SeqTrainer

[[autodoc]] Seq2SeqTrainer
    - evaluate
    - predict

## TrainingArguments

[[autodoc]] TrainingArguments
    - all

## Seq2SeqTrainingArguments

[[autodoc]] Seq2SeqTrainingArguments
    - all

## Checkpoints

デフォルトでは、[`Trainer`] はすべてのチェックポイントを、
[`TrainingArguments`] を使用しています。これらは、xxx を含む`checkpoint-xxx`という名前のサブフォルダーに保存されます。
それはトレーニングの段階でした。

チェックポイントからトレーニングを再開するには、次のいずれかを使用して [`Trainer.train`] を呼び出します。

- `resume_from_checkpoint=True` は最新のチェックポイントからトレーニングを再開します
- `resume_from_checkpoint=checkpoint_dir` ディレクトリ内の特定のチェックポイントからトレーニングを再開します
  合格した。

さらに、`push_to_hub=True` を使用すると、モデル ハブにチェックポイントを簡単に保存できます。デフォルトでは、すべて
中間チェックポイントに保存されたモデルは別のコミットに保存されますが、オプティマイザーの状態は保存されません。適応できます
[`TrainingArguments`] の `hub-strategy` 値を次のいずれかにします。

- `"checkpoint"`: 最新のチェックポイントも last-checkpoint という名前のサブフォルダーにプッシュされます。
  `trainer.train(resume_from_checkpoint="output_dir/last-checkpoint")` を使用してトレーニングを簡単に再開します。
- `"all_checkpoints"`: すべてのチェックポイントは、出力フォルダーに表示されるようにプッシュされます (したがって、1 つのチェックポイントが得られます)
  最終リポジトリ内のフォルダーごとのチェックポイント フォルダー)

## Logging

デフォルトでは、[`Trainer`] はメインプロセスに `logging.INFO` を使用し、レプリカがある場合には `logging.WARNING` を使用します。

これらのデフォルトは、[`TrainingArguments`] の 5 つの `logging` レベルのいずれかを使用するようにオーバーライドできます。
引数:

- `log_level` - メインプロセス用
- `log_level_replica` - レプリカ用

さらに、[`TrainingArguments`] の `log_on_each_node` が `False` に設定されている場合、メイン ノードのみが
メイン プロセスのログ レベル設定を使用すると、他のすべてのノードはレプリカのログ レベル設定を使用します。

[`Trainer`] は、`transformers` のログ レベルをノードごとに個別に設定することに注意してください。
[`Trainer.__init__`]。したがって、他の機能を利用する場合は、これをより早く設定することをお勧めします (次の例を参照)。
[`Trainer`] オブジェクトを作成する前の `transformers` 機能。

これをアプリケーションで使用する方法の例を次に示します。

```python
[...]
logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# set the main code and the modules it uses to the same log-level according to the node
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

trainer = Trainer(...)
```

そして、メイン ノードと他のすべてのノードで重複する可能性が高いものを出力しないように警告するだけを表示したい場合は、
警告: 次のように実行できます。

```bash
my_app.py ... --log_level warning --log_level_replica error
```

マルチノード環境で、各ノードのメインプロセスのログを繰り返したくない場合は、次のようにします。
上記を次のように変更します。

```bash
my_app.py ... --log_level warning --log_level_replica error --log_on_each_node 0
```

その後、最初のノードのメイン プロセスのみが「警告」レベルでログに記録され、メイン ノード上の他のすべてのプロセスはログに記録されます。
ノードと他のノード上のすべてのプロセスは「エラー」レベルでログに記録されます。

アプリケーションをできるだけ静かにする必要がある場合は、次のようにします。

```bash
my_app.py ... --log_level error --log_level_replica error --log_on_each_node 0
```

(マルチノード環境の場合は `--log_on_each_node 0` を追加します)

## Randomness

[`Trainer`] によって生成されたチェックポイントから再開する場合、すべての努力がその状態を復元するために行われます。
_python_、_numpy_、および _pytorch_ の RNG 状態は、そのチェックポイントを保存した時点と同じ状態になります。
これにより、「停止して再開」というスタイルのトレーニングが、ノンストップトレーニングに可能な限り近づけられるはずです。

ただし、さまざまなデフォルトの非決定的な pytorch 設定により、これは完全に機能しない可能性があります。フルをご希望の場合は
決定論については、[ランダム性のソースの制御](https://pytorch.org/docs/stable/notes/randomness) を参照してください。ドキュメントで説明されているように、これらの設定の一部は
物事を決定論的にするもの (例: `torch.backends.cudnn.deterministic`) は物事を遅くする可能性があるため、これは
デフォルトでは実行できませんが、必要に応じて自分で有効にすることができます。

## Specific GPUs Selection

どの GPU をどのような順序で使用するかをプログラムに指示する方法について説明します。

[`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.Parallel.DistributedDataParallel.html) を使用して GPU のサブセットのみを使用する場合、使用する GPU の数を指定するだけです。 。たとえば、GPU が 4 つあるが、最初の 2 つを使用したい場合は、次のようにします。

```bash
torchrun --nproc_per_node=2  trainer-program.py ...
```

[`accelerate`](https://github.com/huggingface/accelerate) または [`deepspeed`](https://github.com/deepspeedai/DeepSpeed) がインストールされている場合は、次を使用して同じことを達成することもできます。の一つ：

```bash
accelerate launch --num_processes 2 trainer-program.py ...
```

```bash
deepspeed --num_gpus 2 trainer-program.py ...
```

これらのランチャーを使用するために、Accelerate または [Deepspeed 統合](deepspeed) 機能を使用する必要はありません。


これまでは、プログラムに使用する GPU の数を指示できました。次に、特定の GPU を選択し、その順序を制御する方法について説明します。

次の環境変数は、使用する GPU とその順序を制御するのに役立ちます。

**`CUDA_VISIBLE_DEVICES`**

複数の GPU があり、そのうちの 1 つまたはいくつかの GPU だけを使用したい場合は、環境変数 `CUDA_VISIBLE_DEVICES` を使用する GPU のリストに設定します。

たとえば、4 つの GPU (0、1、2、3) があるとします。物理 GPU 0 と 2 のみで実行するには、次のようにします。

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun trainer-program.py ...
```

したがって、pytorch は 2 つの GPU のみを認識し、物理 GPU 0 と 2 はそれぞれ `cuda:0` と `cuda:1` にマッピングされます。

順序を変更することもできます。

```bash
CUDA_VISIBLE_DEVICES=2,0 torchrun trainer-program.py ...
```

ここでは、物理 GPU 0 と 2 がそれぞれ`cuda:1`と`cuda:0`にマッピングされています。

上記の例はすべて `DistributedDataParallel` 使用パターンのものですが、同じ方法が [`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) でも機能します。


```bash
CUDA_VISIBLE_DEVICES=2,0 python trainer-program.py ...
```

GPU のない環境をエミュレートするには、次のようにこの環境変数を空の値に設定するだけです。

```bash
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

他の環境変数と同様に、これらをコマンド ラインに追加する代わりに、次のようにエクスポートすることもできます。

```bash
export CUDA_VISIBLE_DEVICES=0,2
torchrun trainer-program.py ...
```

ただし、この方法では、以前に環境変数を設定したことを忘れて、なぜ間違った GPU が使用されているのか理解できない可能性があるため、混乱を招く可能性があります。したがって、このセクションのほとんどの例で示されているように、同じコマンド ラインで特定の実行に対してのみ環境変数を設定するのが一般的です。

**`CUDA_DEVICE_ORDER`**

物理デバイスの順序を制御する追加の環境変数 `CUDA_DEVICE_ORDER` があります。選択肢は次の 2 つです。

1. PCIe バス ID 順 (`nvidia-smi` の順序と一致) - これがデフォルトです。

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

2. GPU コンピューティング能力順に並べる

```bash
export CUDA_DEVICE_ORDER=FASTEST_FIRST
```

ほとんどの場合、この環境変数を気にする必要はありませんが、古い GPU と新しい GPU が物理的に挿入されているため、遅い古いカードが遅くなっているように見えるような偏ったセットアップを行っている場合には、非常に役立ちます。初め。これを解決する 1 つの方法は、カードを交換することです。ただし、カードを交換できない場合 (デバイスの冷却が影響を受けた場合など)、`CUDA_DEVICE_ORDER=FASTEST_FIRST`を設定すると、常に新しい高速カードが最初に配置されます。ただし、`nvidia-smi`は依然として PCIe の順序でレポートするため、多少混乱するでしょう。

順序を入れ替えるもう 1 つの解決策は、以下を使用することです。

```bash
export CUDA_VISIBLE_DEVICES=1,0
```

この例では 2 つの GPU だけを使用していますが、もちろん、コンピューターに搭載されている数の GPU にも同じことが当てはまります。

また、この環境変数を設定する場合は、`~/.bashrc` ファイルまたはその他の起動設定ファイルに設定して、忘れるのが最善です。

## Trainer Integrations

[`Trainer`] は、トレーニングを劇的に改善する可能性のあるライブラリをサポートするように拡張されました。
時間とはるかに大きなモデルに適合します。

現在、サードパーティのソリューション [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) および [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html) をサポートしています。論文 [ZeRO: メモリの最適化兆パラメータ モデルのトレーニングに向けて、Samyam Rajbhandari、Jeff Rasley、Olatunji Ruwase、Yuxiong He 著](https://arxiv.org/abs/1910.02054)。

この提供されるサポートは、この記事の執筆時点では新しくて実験的なものです。 DeepSpeed と PyTorch FSDP のサポートはアクティブであり、それに関する問題は歓迎しますが、FairScale 統合は PyTorch メインに統合されているため、もうサポートしていません ([PyTorch FSDP 統合](#pytorch-fully-sharded-data-parallel))

<a id='zero-install-notes'></a>

### CUDA Extension Installation Notes

この記事の執筆時点では、Deepspeed を使用するには、CUDA C++ コードをコンパイルする必要があります。

すべてのインストールの問題は、[Deepspeed](https://github.com/deepspeedai/DeepSpeed/issues) の対応する GitHub の問題を通じて対処する必要がありますが、ビルド中に発生する可能性のある一般的な問題がいくつかあります。
CUDA 拡張機能を構築する必要がある PyTorch 拡張機能。

したがって、次の操作を実行中に CUDA 関連のビルドの問題が発生した場合は、次のとおりです。

```bash
pip install deepspeed
```

まず次の注意事項をお読みください。

これらのノートでは、`pytorch` が CUDA `10.2` でビルドされた場合に何をすべきかの例を示します。あなたの状況が次のような場合
異なる場合は、バージョン番号を目的のバージョンに調整することを忘れないでください。

#### Possible problem #1

Pytorch には独自の CUDA ツールキットが付属していますが、これら 2 つのプロジェクトをビルドするには、同一バージョンの CUDA が必要です。
システム全体にインストールされます。

たとえば、Python 環境に `cudatoolkit==10.2` を指定して `pytorch` をインストールした場合は、次のものも必要です。
CUDA `10.2` がシステム全体にインストールされました。

正確な場所はシステムによって異なる場合がありますが、多くのシステムでは`/usr/local/cuda-10.2`が最も一般的な場所です。
Unix システム。 CUDA が正しく設定され、`PATH`環境変数に追加されると、
次のようにしてインストール場所を指定します。


```bash
which nvcc
```

CUDA がシステム全体にインストールされていない場合は、最初にインストールしてください。お気に入りを使用して手順を見つけることができます
検索エンジン。たとえば、Ubuntu を使用している場合は、[ubuntu cuda 10.2 install](https://www.google.com/search?q=ubuntu+cuda+10.2+install) を検索するとよいでしょう。

#### Possible problem #2

もう 1 つの考えられる一般的な問題は、システム全体に複数の CUDA ツールキットがインストールされている可能性があることです。たとえばあなた
がある可能性があり：

```bash
/usr/local/cuda-10.2
/usr/local/cuda-11.0
```

この状況では、`PATH` および `LD_LIBRARY_PATH` 環境変数に以下が含まれていることを確認する必要があります。
目的の CUDA バージョンへの正しいパス。通常、パッケージ インストーラーは、これらに、
最後のバージョンがインストールされました。適切なパッケージが見つからないためにパッケージのビルドが失敗するという問題が発生した場合は、
CUDA バージョンがシステム全体にインストールされているにもかかわらず、前述の 2 つを調整する必要があることを意味します
環境変数。

まず、その内容を見てみましょう。

```bash
echo $PATH
echo $LD_LIBRARY_PATH
```

それで、中に何が入っているかがわかります。

`LD_LIBRARY_PATH` が空である可能性があります。

`PATH` は実行可能ファイルが存在する場所をリストし、`LD_LIBRARY_PATH` は共有ライブラリの場所を示します。
探すことです。どちらの場合も、前のエントリが後のエントリより優先されます。 `:` は複数を区切るために使用されます
エントリ。

ここで、ビルド プログラムに特定の CUDA ツールキットの場所を指示するには、最初にリストされる希望のパスを挿入します。
やっていること：

```bash
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```

既存の値を上書きするのではなく、先頭に追加することに注意してください。

もちろん、必要に応じてバージョン番号やフルパスを調整します。割り当てたディレクトリが実際に機能することを確認してください
存在する。 `lib64` サブディレクトリは、`libcudart.so` などのさまざまな CUDA `.so` オブジェクトが存在する場所です。
システムでは別の名前が付けられますが、現実を反映するように調整してください。

#### Possible problem #3

一部の古い CUDA バージョンは、新しいコンパイラでのビルドを拒否する場合があります。たとえば、あなたは`gcc-9`を持っていますが、それが必要です
`gcc-7`。

それにはさまざまな方法があります。

最新の CUDA ツールキットをインストールできる場合は、通常、新しいコンパイラがサポートされているはずです。

あるいは、既に所有しているコンパイラに加えて、下位バージョンのコンパイラをインストールすることもできます。
すでに存在しますが、デフォルトではないため、ビルドシステムはそれを認識できません。 「gcc-7」がインストールされているが、
ビルドシステムが見つからないというメッセージを表示する場合は、次の方法で解決できる可能性があります。

```bash
sudo ln -s /usr/bin/gcc-7  /usr/local/cuda-10.2/bin/gcc
sudo ln -s /usr/bin/g++-7  /usr/local/cuda-10.2/bin/g++
```

ここでは、`/usr/local/cuda-10.2/bin/gcc` から `gcc-7` へのシンボリックリンクを作成しています。
`/usr/local/cuda-10.2/bin/` は `PATH` 環境変数内にある必要があります (前の問題の解決策を参照)。
`gcc-7` (および `g++7`) が見つかるはずで、ビルドは成功します。

いつものように、状況に合わせて例のパスを編集してください。

### PyTorch Fully Sharded Data parallel

より大きなバッチ サイズで巨大なモデルのトレーニングを高速化するには、完全にシャード化されたデータ並列モデルを使用できます。
このタイプのデータ並列パラダイムでは、オプティマイザーの状態、勾配、パラメーターをシャーディングすることで、より多くのデータと大規模なモデルをフィッティングできます。
この機能とその利点の詳細については、[完全シャーディング データ並列ブログ](https://pytorch.org/blog/introducing-pytorch-full-sharded-data-Parallel-api/) をご覧ください。
最新の PyTorch の Fully Sharded Data Parallel (FSDP) トレーニング機能を統合しました。
必要なのは、設定を通じて有効にすることだけです。

**FSDP サポートに必要な PyTorch バージョン**: PyTorch Nightly (リリース後にこれを読んだ場合は 1.12.0)
FSDP を有効にしたモデルの保存は、最近の修正でのみ利用できるためです。

**使用法**：

- 配布されたランチャーが追加されていることを確認してください
まだ使用していない場合は、`-m torch.distributed.launch --nproc_per_node=NUMBER_OF_GPUS_YOU_HAVE`を使用します。

- **シャーディング戦略**:
  - FULL_SHARD : データ並列ワーカー/GPU にわたるシャード オプティマイザーの状態 + 勾配 + モデル パラメーター。
    このためには、コマンドライン引数に`--fsdp full_shard`を追加します。
  - SHARD_GRAD_OP : シャード オプティマイザーの状態 + データ並列ワーカー/GPU 全体の勾配。
    このためには、コマンドライン引数に`--fsdp shard_grad_op`を追加します。
  - NO_SHARD : シャーディングなし。このためには、コマンドライン引数に`--fsdp no_shard`を追加します。
- パラメータと勾配を CPU にオフロードするには、
  コマンドライン引数に`--fsdp "full_shard offload"`または`--fsdp "shard_grad_op offload"`を追加します。
- `default_auto_wrap_policy` を使用して FSDP でレイヤーを自動的に再帰的にラップするには、
  コマンドライン引数に`--fsdp "full_shard auto_wrap"`または`--fsdp "shard_grad_op auto_wrap"`を追加します。
- CPU オフロードと自動ラッピングの両方を有効にするには、
  コマンドライン引数に`--fsdp "full_shard offload auto_wrap"`または`--fsdp "shard_grad_op offload auto_wrap"`を追加します。
- 残りの FSDP 構成は、`--fsdp_config <path_to_fsdp_config.json>`を介して渡されます。それは、次のいずれかの場所です。
  FSDP json 構成ファイル (例: `fsdp_config.json`)、またはすでにロードされている json ファイルを `dict` として使用します。
  - 自動ラッピングが有効な場合は、トランスベースの自動ラップ ポリシーまたはサイズ ベースの自動ラップ ポリシーを使用できます。
    - トランスフォーマーベースの自動ラップポリシーの場合、構成ファイルで `fsdp_transformer_layer_cls_to_wrap` を指定することをお勧めします。指定しない場合、使用可能な場合、デフォルト値は `model._no_split_modules` になります。
      これは、ラップするトランスフォーマー層クラス名のリスト (大文字と小文字を区別) を指定します (例: [`BertLayer`]、[`GPTJBlock`]、[`T5Block`] ...)。
      重みを共有するサブモジュール (埋め込み層など) が異なる FSDP ラップされたユニットにならないようにする必要があるため、これは重要です。
      このポリシーを使用すると、マルチヘッド アテンションとそれに続くいくつかの MLP レイヤーを含むブロックごとにラッピングが発生します。
      共有埋め込みを含む残りの層は、同じ最も外側の FSDP ユニットにラップされるのが便利です。
      したがって、トランスベースのモデルにはこれを使用してください。
    - サイズベースの自動ラップポリシーの場合は、設定ファイルに`fsdp_min_num_params`を追加してください。
      自動ラッピングのための FSDP のパラメータの最小数を指定します。
  - 設定ファイルで `fsdp_backward_prefetch` を指定できるようになりました。次のパラメータのセットをいつプリフェッチするかを制御します。
    `backward_pre` と `backward_pos` が利用可能なオプションです。
    詳細については、`torch.distributed.fsdp.full_sharded_data_Parallel.BackwardPrefetch`を参照してください。
  - 設定ファイルで `fsdp_forward_prefetch` を指定できるようになりました。次のパラメータのセットをいつプリフェッチするかを制御します。
    `True`の場合、FSDP はフォワード パスでの実行中に、次に来るオールギャザーを明示的にプリフェッチします。
  - 設定ファイルで `limit_all_gathers` を指定できるようになりました。
    `True`の場合、FSDP は CPU スレッドを明示的に同期して、実行中のオールギャザが多すぎるのを防ぎます。
  - `activation_checkpointing`を設定ファイルで指定できるようになりました。
    `True`の場合、FSDP アクティベーション チェックポイントは、FSDP のアクティベーションをクリアすることでメモリ使用量を削減する手法です。
    特定のレイヤーを処理し、バックワード パス中にそれらを再計算します。事実上、これは余分な計算時間を犠牲にします
    メモリ使用量を削減します。

**注意すべき注意点がいくつかあります**
- これは `generate` と互換性がないため、 `--predict_with_generate` とも互換性がありません
  すべての seq2seq/clm スクリプト (翻訳/要約/clm など)。
  問題 [#21667](https://github.com/huggingface/transformers/issues/21667) を参照してください。

### PyTorch/XLA Fully Sharded Data parallel

TPU ユーザーの皆様に朗報です。 PyTorch/XLA は FSDP をサポートするようになりました。
最新の Fully Sharded Data Parallel (FSDP) トレーニングがすべてサポートされています。
詳細については、[FSDP を使用した Cloud TPU での PyTorch モデルのスケーリング](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/) および [PyTorch/XLA 実装 を参照してください。 FSDP の](https://github.com/pytorch/xla/tree/master/torch_xla/distributed/fsdp)
必要なのは、設定を通じて有効にすることだけです。

**FSDP サポートに必要な PyTorch/XLA バージョン**: >=2.0

**使用法**：

`--fsdp "full shard"` を、`--fsdp_config <path_to_fsdp_config.json>` に加えられる次の変更とともに渡します。
- PyTorch/XLA FSDP を有効にするには、`xla`を`True`に設定する必要があります。
- `xla_fsdp_settings` 値は、XLA FSDP ラッピング パラメータを格納する辞書です。
  オプションの完全なリストについては、[こちら](
  https://github.com/pytorch/xla/blob/master/torch_xla/distributed/fsdp/xla_full_sharded_data_Parallel.py)。
- `xla_fsdp_grad_ckpt`。 `True`の場合、ネストされた XLA FSDP でラップされた各レイヤー上で勾配チェックポイントを使用します。
  この設定は、xla フラグが true に設定されており、自動ラッピング ポリシーが指定されている場合にのみ使用できます。
  `fsdp_min_num_params` または `fsdp_transformer_layer_cls_to_wrap`。
- トランスフォーマー ベースの自動ラップ ポリシーまたはサイズ ベースの自動ラップ ポリシーのいずれかを使用できます。
  - トランスフォーマーベースの自動ラップポリシーの場合、構成ファイルで `fsdp_transformer_layer_cls_to_wrap` を指定することをお勧めします。指定しない場合、使用可能な場合、デフォルト値は `model._no_split_modules` になります。
    これは、ラップするトランスフォーマー層クラス名のリスト (大文字と小文字を区別) を指定します (例: [`BertLayer`]、[`GPTJBlock`]、[`T5Block`] ...)。
    重みを共有するサブモジュール (埋め込み層など) が異なる FSDP ラップされたユニットにならないようにする必要があるため、これは重要です。
    このポリシーを使用すると、マルチヘッド アテンションとそれに続くいくつかの MLP レイヤーを含むブロックごとにラッピングが発生します。
    共有埋め込みを含む残りの層は、同じ最も外側の FSDP ユニットにラップされるのが便利です。
    したがって、トランスベースのモデルにはこれを使用してください。
  - サイズベースの自動ラップポリシーの場合は、設定ファイルに`fsdp_min_num_params`を追加してください。
    自動ラッピングのための FSDP のパラメータの最小数を指定します。

### Using Trainer for accelerated PyTorch Training on Mac 

PyTorch v1.12 リリースにより、開発者と研究者は Apple シリコン GPU を利用してモデル トレーニングを大幅に高速化できます。
これにより、プロトタイピングや微調整などの機械学習ワークフローを Mac 上でローカルで実行できるようになります。
PyTorch のバックエンドとしての Apple の Metal Performance Shaders (MPS) はこれを可能にし、新しい `"mps"` デバイス経由で使用できます。
これにより、計算グラフとプリミティブが MPS Graph フレームワークと MPS によって提供される調整されたカーネルにマッピングされます。
詳細については、公式ドキュメント [Mac での Accelerated PyTorch Training の紹介](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) を参照してください。
および [MPS バックエンド](https://pytorch.org/docs/stable/notes/mps.html)。

<Tip warning={false}>

MacOS マシンに PyTorch >= 1.13 (執筆時点ではナイトリー バージョン) をインストールすることを強くお勧めします。
トランスベースのモデルのモデルの正確性とパフォーマンスの向上に関連する主要な修正が行われています。
詳細については、https://github.com/pytorch/pytorch/issues/82707 を参照してください。

</Tip>

**Apple Silicon チップを使用したトレーニングと推論の利点**

1. ユーザーがローカルで大規模なネットワークやバッチ サイズをトレーニングできるようにします
2. ユニファイド メモリ アーキテクチャにより、データ取得の遅延が短縮され、GPU がメモリ ストア全体に直接アクセスできるようになります。
したがって、エンドツーエンドのパフォーマンスが向上します。
3. クラウドベースの開発に関連するコストや追加のローカル GPU の必要性を削減します。

**前提条件**: mps サポートを備えたトーチをインストールするには、
この素晴らしいメディア記事 [GPU アクセラレーションが M1 Mac の PyTorch に登場](https://medium.com/towards-data-science/gpu-acceleration-comes-to-pytorch-on-m1-macs-195c399efcc1) に従ってください。 。

**使用法**：
`mps` デバイスは、`cuda` デバイスが使用される方法と同様に利用可能な場合、デフォルトで使用されます。
したがって、ユーザーによるアクションは必要ありません。
たとえば、以下のコマンドを使用して、Apple Silicon GPU を使用して公式の Glue テキスト分類タスクを (ルート フォルダーから) 実行できます。

```bash
export TASK_NAME=mrpc

python examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
```

**注意すべきいくつかの注意事項**

1. 一部の PyTorch 操作は mps に実装されていないため、エラーがスローされます。
これを回避する 1 つの方法は、環境変数 `PYTORCH_ENABLE_MPS_FALLBACK=1` を設定することです。
これらの操作では CPU にフォールバックします。ただし、それでも UserWarning がスローされます。
2. 分散セットアップ`gloo`および`nccl`は、`mps`デバイスでは動作しません。
これは、現在「mps」デバイス タイプの単一 GPU のみを使用できることを意味します。

最後に、覚えておいてください。 🤗 `Trainer` は MPS バックエンドのみを統合するため、
MPS バックエンドの使用に関して問題や質問がある場合は、
[PyTorch GitHub](https://github.com/pytorch/pytorch/issues) に問題を提出してください。

## Using Accelerate Launcher with Trainer

加速してトレーナーにパワーを与えましょう。ユーザーが期待することに関しては、次のとおりです。
- トレーナー引数に対して FSDP、DeepSpeed などのトレーナー インテレーションを変更せずに使用し続けることができます。
- トレーナーで Accelerate Launcher を使用できるようになりました (推奨)。

トレーナーで Accelerate Launcher を使用する手順:
1. 🤗 Accelerate がインストールされていることを確認してください。Accelerate がないと `Trainer` を使用することはできません。そうでない場合は、`pip install accelerate`してください。 Accelerate のバージョンを更新する必要がある場合もあります: `pip install activate --upgrade`
2. `accelerate config`を実行し、アンケートに記入します。以下は加速設定の例です。
  ａ． DDP マルチノード マルチ GPU 構成:
    ```yaml
    compute_environment: LOCAL_MACHINE                                                                                             
    distributed_type: MULTI_GPU                                                                                                    
    downcast_bf16: 'no'
    gpu_ids: all
    machine_rank: 0 #change rank as per the node
    main_process_ip: 192.168.20.1
    main_process_port: 9898
    main_training_function: main
    mixed_precision: fp16
    num_machines: 2
    num_processes: 8
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ```

  b. FSDP config:
    ```yaml
    compute_environment: LOCAL_MACHINE
    distributed_type: FSDP
    downcast_bf16: 'no'
    fsdp_config:
      fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
      fsdp_backward_prefetch_policy: BACKWARD_PRE
      fsdp_forward_prefetch: true
      fsdp_offload_params: false
      fsdp_sharding_strategy: 1
      fsdp_state_dict_type: FULL_STATE_DICT
      fsdp_sync_module_states: true
      fsdp_transformer_layer_cls_to_wrap: BertLayer
      fsdp_use_orig_params: true
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16
    num_machines: 1
    num_processes: 2
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ```
  c.ファイルを指す DeepSpeed 構成:
    ```yaml
    compute_environment: LOCAL_MACHINE
    deepspeed_config:
      deepspeed_config_file: /home/user/configs/ds_zero3_config.json
      zero3_init_flag: true
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    machine_rank: 0
    main_training_function: main
    num_machines: 1
    num_processes: 4
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ``` 

  d.加速プラグインを使用した DeepSpeed 構成:

    ```yaml
    compute_environment: LOCAL_MACHINE                                                                                             
    deepspeed_config:                                                                                                              
      gradient_accumulation_steps: 1
      gradient_clipping: 0.7
      offload_optimizer_device: cpu
      offload_param_device: cpu
      zero3_init_flag: true
      zero_stage: 2
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16
    num_machines: 1
    num_processes: 4
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ```

3. 加速設定またはランチャー引数によって上記で処理された引数以外の引数を使用して、トレーナー スクリプトを実行します。
以下は、上記の FSDP 構成で`accelerate launcher`を使用して`run_glue.py`を実行する例です。 

```bash
cd transformers

accelerate launch \
./examples/pytorch/text-classification/run_glue.py \
--model_name_or_path google-bert/bert-base-cased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir /tmp/$TASK_NAME/ \
--overwrite_output_dir
```

4. `accelerate launch`するための cmd 引数を直接使用することもできます。上の例は次のようにマッピングされます。

```bash
cd transformers

accelerate launch --num_processes=2 \
--use_fsdp \
--mixed_precision=bf16 \
--fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP  \
--fsdp_transformer_layer_cls_to_wrap="BertLayer" \
--fsdp_sharding_strategy=1 \
--fsdp_state_dict_type=FULL_STATE_DICT \
./examples/pytorch/text-classification/run_glue.py
--model_name_or_path google-bert/bert-base-cased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir /tmp/$TASK_NAME/ \
--overwrite_output_dir
```

詳細については、🤗 Accelerate CLI ガイドを参照してください: [🤗 Accelerate スクリプトの起動](https://huggingface.co/docs/accelerate/basic_tutorials/launch)。

移動されたセクション:

[ <a href="./deepspeed#deepspeed-trainer-integration">DeepSpeed</a><a id="deepspeed"></a>
| <a href="./deepspeed#deepspeed-installation">Installation</a><a id="installation"></a>
| <a href="./deepspeed#deepspeed-multi-gpu">Deployment with multiple GPUs</a><a id="deployment-with-multiple-gpus"></a>
| <a href="./deepspeed#deepspeed-one-gpu">Deployment with one GPU</a><a id="deployment-with-one-gpu"></a>
| <a href="./deepspeed#deepspeed-notebook">Deployment in Notebooks</a><a id="deployment-in-notebooks"></a>
| <a href="./deepspeed#deepspeed-config">Configuration</a><a id="configuration"></a>
| <a href="./deepspeed#deepspeed-config-passing">Passing Configuration</a><a id="passing-configuration"></a>
| <a href="./deepspeed#deepspeed-config-shared">Shared Configuration</a><a id="shared-configuration"></a>
| <a href="./deepspeed#deepspeed-zero">ZeRO</a><a id="zero"></a>
| <a href="./deepspeed#deepspeed-zero2-config">ZeRO-2 Config</a><a id="zero-2-config"></a>
| <a href="./deepspeed#deepspeed-zero3-config">ZeRO-3 Config</a><a id="zero-3-config"></a>
| <a href="./deepspeed#deepspeed-nvme">NVMe Support</a><a id="nvme-support"></a>
| <a href="./deepspeed#deepspeed-zero2-zero3-performance">ZeRO-2 vs ZeRO-3 Performance</a><a id="zero-2-vs-zero-3-performance"></a>
| <a href="./deepspeed#deepspeed-zero2-example">ZeRO-2 Example</a><a id="zero-2-example"></a>
| <a href="./deepspeed#deepspeed-zero3-example">ZeRO-3 Example</a><a id="zero-3-example"></a>
| <a href="./deepspeed#deepspeed-optimizer">Optimizer</a><a id="optimizer"></a>
| <a href="./deepspeed#deepspeed-scheduler">Scheduler</a><a id="scheduler"></a>
| <a href="./deepspeed#deepspeed-fp32">fp32 Precision</a><a id="fp32-precision"></a>
| <a href="./deepspeed#deepspeed-amp">Automatic Mixed Precision</a><a id="automatic-mixed-precision"></a>
| <a href="./deepspeed#deepspeed-bs">Batch Size</a><a id="batch-size"></a>
| <a href="./deepspeed#deepspeed-grad-acc">Gradient Accumulation</a><a id="gradient-accumulation"></a>
| <a href="./deepspeed#deepspeed-grad-clip">Gradient Clipping</a><a id="gradient-clipping"></a>
| <a href="./deepspeed#deepspeed-weight-extraction">Getting The Model Weights Out</a><a id="getting-the-model-weights-out"></a>
]
