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

# DeepSpeed の統合

[DeepSpeed](https://github.com/microsoft/DeepSpeed) は、[ZeRO 論文](https://arxiv.org/abs/1910.02054) で説明されているすべてを実装します。現在、次のものを完全にサポートしています。

1. オプティマイザーの状態分割 (ZeRO ステージ 1)
2. 勾配分割 (ZeRO ステージ 2)
3. パラメーターの分割 (ZeRO ステージ 3)
4. カスタム混合精度トレーニング処理
5. 一連の高速 CUDA 拡張ベースのオプティマイザー
6. CPU および NVMe への ZeRO オフロード

ZeRO-Offload には独自の専用ペーパーがあります: [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)。 NVMe サポートについては、論文 [ZeRO-Infinity: Breaking the GPU] で説明されています。
極端なスケールの深層学習のためのメモリ ウォール](https://arxiv.org/abs/2104.07857)。

DeepSpeed ZeRO-2 は、その機能が推論には役に立たないため、主にトレーニングのみに使用されます。

DeepSpeed ZeRO-3 は、巨大なモデルを複数の GPU にロードできるため、推論にも使用できます。
単一の GPU では不可能です。

🤗 Transformers は、2 つのオプションを介して [DeepSpeed](https://github.com/microsoft/DeepSpeed) を統合します。

1. [`Trainer`] によるコア DeepSpeed 機能の統合。何でもやってくれるタイプです
   統合の場合 - カスタム構成ファイルを指定するか、テンプレートを使用するだけで、他に何もする必要はありません。たいていの
   このドキュメントではこの機能に焦点を当てています。
2. [`Trainer`] を使用せず、DeepSpeed を統合した独自のトレーナーを使用したい場合
   `from_pretrained` や `from_config` などのコア機能には、重要な機能の統合が含まれています。
   ZeRO ステージ 3 以降の「zero.Init」などの DeepSpeed の部分。この機能を活用するには、次のドキュメントをお読みください。
   [非トレーナー DeepSpeed 統合](#nontrainer-deepspeed-integration)。

統合されているもの:

トレーニング：

1. DeepSpeed ZeRO トレーニングは、ZeRO-Infinity (CPU および NVME オフロード) を使用して完全な ZeRO ステージ 1、2、および 3 をサポートします。

推論：

1. DeepSpeed ZeRO Inference は、ZeRO-Infinity による ZeRO ステージ 3 をサポートします。トレーニングと同じ ZeRO プロトコルを使用しますが、
   オプティマイザと lr スケジューラは使用せず、ステージ 3 のみが関連します。詳細については、以下を参照してください。
   [ゼロ推論](#ゼロ推論)。

DeepSpeed Inference もあります。これは、Tensor Parallelism の代わりに Tensor Parallelism を使用するまったく異なるテクノロジーです。
ZeRO (近日公開)。

<a id='deepspeed-trainer-integration'></a>


## Trainer Deepspeed Integration


<a id='deepspeed-installation'></a>

### Installation

Install the library via pypi:

```bash
pip install deepspeed
```

or via `transformers`' `extras`:

```bash
pip install transformers[deepspeed]
```

または、[DeepSpeed の GitHub ページ](https://github.com/microsoft/deepspeed#installation) で詳細を確認してください。
[高度なインストール](https://www.deepspeed.ai/tutorials/advanced-install/)。

それでもビルドに苦労する場合は、まず [CUDA 拡張機能のインストール ノート](trainer#cuda-extension-installation-notes) を必ず読んでください。

拡張機能を事前ビルドせず、実行時に拡張機能がビルドされることに依存しており、上記の解決策をすべて試した場合
それが役に立たなかった場合、次に試すべきことは、モジュールをインストールする前にモジュールを事前にビルドすることです。

DeepSpeed のローカル ビルドを作成するには:

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

NVMe オフロードを使用する場合は、上記の手順に「DS_BUILD_AIO=1」を含める必要があります (また、
*libaio-dev* システム全体にインストールします)。

`TORCH_CUDA_ARCH_LIST` を編集して、使用する GPU カードのアーキテクチャのコードを挿入します。すべてを仮定すると
あなたのカードは同じで、次の方法でアーチを取得できます。

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"
```

したがって、`8, 6`を取得した場合は、`TORCH_CUDA_ARCH_LIST="8.6"`を使用します。複数の異なるカードをお持ちの場合は、すべてをリストすることができます
それらのうち、`TORCH_CUDA_ARCH_LIST="6.1;8.6"`が好きです

複数のマシンで同じセットアップを使用する必要がある場合は、バイナリ ホイールを作成します。

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 \
python setup.py build_ext -j8 bdist_wheel
```

`dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`のようなものが生成されるので、これをインストールできます
`pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`としてローカルまたは他のマシンにインストールします。

繰り返しますが、`TORCH_CUDA_ARCH_LIST`をターゲット アーキテクチャに合わせて調整することを忘れないでください。

NVIDIA GPU の完全なリストと、それに対応する **コンピューティング機能** (この記事の Arch と同じ) を見つけることができます。
コンテキスト) [ここ](https://developer.nvidia.com/cuda-gpus)。

以下を使用して、pytorch が構築されたアーチを確認できます。

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

ここでは、インストールされている GPU の 1 つのアーチを見つける方法を説明します。たとえば、GPU 0 の場合:

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; \
print(torch.cuda.get_device_properties(torch.device('cuda')))"
```

出力が次の場合:

```bash
_CudaDeviceProperties(name='GeForce RTX 3090', major=8, minor=6, total_memory=24268MB, multi_processor_count=82)
```

そうすれば、このカードのアーチが`8.6`であることがわかります。

`TORCH_CUDA_ARCH_LIST` を完全に省略することもできます。そうすれば、ビルド プログラムが自動的にクエリを実行します。
ビルドが行われる GPU のアーキテクチャ。これは、ターゲット マシンの GPU と一致する場合もあれば、一致しない場合もあります。
目的のアーチを明示的に指定することをお勧めします。

提案されたことをすべて試してもまだビルドの問題が発生する場合は、GitHub の問題に進んでください。
[ディープスピード](https://github.com/microsoft/DeepSpeed/issues)、


### Deployment with multiple GPUs

DeepSpeed 統合をデプロイするには、[`Trainer`] コマンド ライン引数を調整して新しい引数 `--deepspeed ds_config.json` を含めます。ここで、`ds_config.json` は DeepSpeed 構成ファイルです。
   [こちら](https://www.deepspeed.ai/docs/config-json/)に記載されています。ファイル名はあなた次第です。
   DeepSpeed の`add_config_arguments`ユーティリティを使用して、必要なコマンド ライン引数をコードに追加することをお勧めします。
   詳細については、[DeepSpeed の引数解析](https://deepspeed.readthedocs.io/en/latest/initialize.html#argument-parsing) ドキュメントを参照してください。

ここで選択したランチャーを使用できます。 pytorch ランチャーを引き続き使用できます。

```bash
torch.distributed.run --nproc_per_node=2 your_program.py <normal cl args> --deepspeed ds_config.json
```

または、`deepspeed`によって提供されるランチャーを使用します。

```bash
deepspeed --num_gpus=2 your_program.py <normal cl args> --deepspeed ds_config.json
```

ご覧のとおり、引数は同じではありませんが、ほとんどのニーズではどちらでも機能します。の
さまざまなノードと GPU を構成する方法の詳細については、[こちら](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) を参照してください。

`deepspeed`ランチャーを使用し、利用可能なすべての GPU を使用したい場合は、`--num_gpus`フラグを省略するだけです。

以下は、利用可能なすべての GPU をデプロイする DeepSpeed で`run_translation.py`を実行する例です。

```bash
deepspeed examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

DeepSpeed のドキュメントには、「--deepspeed --deepspeed_config ds_config.json」が表示される可能性が高いことに注意してください。
DeepSpeed 関連の引数が 2 つありますが、簡単にするためであり、処理すべき引数がすでに非常に多いためです。
この 2 つを 1 つの引数に結合しました。

実際の使用例については、この [投稿](https://github.com/huggingface/transformers/issues/8771#issuecomment-759248400) を参照してください。

<a id='deepspeed-one-gpu'></a>


### Deployment with one GPU

1 つの GPU で DeepSpeed をデプロイするには、[`Trainer`] コマンド ライン引数を次のように調整します。

```bash
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero2.json \
--model_name_or_path t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

これは複数の GPU の場合とほぼ同じですが、ここでは、DeepSpeed に 1 つの GPU だけを使用するように明示的に指示します。
`--num_gpus=1`。デフォルトでは、DeepSpeed は指定されたノード上で認識できるすべての GPU をデプロイします。起動する GPU が 1 つだけの場合
の場合、この引数は必要ありません。次の [ドキュメント](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) では、ランチャー オプションについて説明しています。

1 つの GPU だけで DeepSpeed を使用したいのはなぜですか?

1. 一部の計算とメモリをホストの CPU と RAM に委任できる ZeRO オフロード機能を備えているため、
   モデルのニーズに合わせてより多くの GPU リソースを残しておきます。より大きなバッチ サイズ、または非常に大きなモデルのフィッティングを可能にする
   普通は合わないでしょう。
2. スマートな GPU メモリ管理システムを提供し、メモリの断片化を最小限に抑えます。
   より大きなモデルとデータ バッチ。

次に構成について詳しく説明しますが、単一の GPU で大幅な改善を実現するための鍵は次のとおりです。
DeepSpeed を使用するには、構成ファイルに少なくとも次の構成が必要です。

```json
{
  "zero_optimization": {
     "stage": 2,
     "offload_optimizer": {
         "device": "cpu",
         "pin_memory": true
     },
     "allgather_partitions": true,
     "allgather_bucket_size": 2e8,
     "reduce_scatter": true,
     "reduce_bucket_size": 2e8,
     "overlap_comm": true,
     "contiguous_gradients": true
  }
}
```

これにより、オプティマイザーのオフロードやその他の重要な機能が有効になります。バッファ サイズを試してみるとよいでしょう。
詳細については、以下のディスカッションを参照してください。

このタイプのデプロイメントの実際的な使用例については、この [投稿](https://github.com/huggingface/transformers/issues/8771#issuecomment-759176685) を参照してください。

このドキュメントで詳しく説明されているように、CPU および NVMe オフロードを備えた ZeRO-3 を試すこともできます。

ノート：

- GPU 0 とは異なる特定の GPU で実行する必要がある場合、`CUDA_VISIBLE_DEVICES` を使用して制限することはできません。
  利用可能な GPU の表示範囲。代わりに、次の構文を使用する必要があります。

  ```bash
  deepspeed --include localhost:1 examples/pytorch/translation/run_translation.py ...
  ```

  この例では、DeepSpeed に GPU 1 (2 番目の GPU) を使用するように指示します。

<a id='deepspeed-multi-node'></a>

### 複数のノードを使用したデプロイメント

このセクションの情報は DeepSpeed 統合に固有のものではなく、あらゆるマルチノード プログラムに適用できます。ただし、DeepSpeed は、SLURM 環境でない限り、他のランチャーよりも使いやすい「deepspeed」ランチャーを提供します。

このセクションでは、それぞれ 8 GPU を備えた 2 つのノードがあると仮定します。また、最初のノードには `ssh hostname1` を使用して、2 番目のノードには `ssh hostname2` を使用して接続できます。両方ともパスワードなしでローカルの ssh 経由で相互に接続できる必要があります。もちろん、これらのホスト (ノード) 名を、作業している実際のホスト名に変更する必要があります。

#### The torch.distributed.run launcher


たとえば、`torch.distributed.run` を使用するには、次のようにします。

```bash
python -m torch.distributed.run --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=hostname1 \
--master_port=9901 your_program.py <normal cl args> --deepspeed ds_config.json
```

各ノードに SSH で接続し、それぞれのノードで同じコマンドを実行する必要があります。急ぐ必要はありません。ランチャーは両方のノードが同期するまで待機します。

詳細については、[torchrun](https://pytorch.org/docs/stable/elastic/run.html) を参照してください。ちなみに、これは pytorch の数バージョン前の`torch.distributed.launch`を置き換えたランチャーでもあります。