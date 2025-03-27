<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Efficient Training on Multiple GPUs

単一のGPUでのトレーニングが遅すぎる場合や、モデルの重みが単一のGPUのメモリに収まらない場合、複数のGPUを使用したセットアップが必要となります。単一のGPUから複数のGPUへの切り替えには、ワークロードを分散するためのある種の並列処理が必要です。データ、テンソル、またはパイプラインの並列処理など、さまざまな並列処理技術があります。ただし、すべてに適した一つの解決策は存在せず、最適な設定は使用するハードウェアに依存します。この記事は、おそらく他のフレームワークにも適用される主要な概念に焦点を当てつつ、PyTorchベースの実装に焦点を当てています。

<Tip>

**注意**: [単一GPUセクション](perf_train_gpu_one) で紹介された多くの戦略（混合精度トレーニングや勾配蓄積など）は一般的であり、モデルのトレーニングに一般的に適用されます。したがって、マルチGPUやCPUトレーニングなどの次のセクションに入る前に、それを確認してください。

</Tip>

まず、さまざまな1D並列処理技術とその利点および欠点について詳しく説明し、それらを2Dおよび3D並列処理に組み合わせてさらに高速なトレーニングを実現し、より大きなモデルをサポートする方法を検討します。さまざまな他の強力な代替手法も紹介されます。

## Concepts

以下は、この文書で後で詳しく説明される主要な概念の簡単な説明です。

1. **DataParallel (DP)** - 同じセットアップが複数回複製され、各セットアップにデータのスライスが供給されます。処理は並行して行われ、各セットアップはトレーニングステップの最後に同期されます。
2. **TensorParallel (TP)** - 各テンソルは複数のチャンクに分割され、単一のGPUにテンソル全体が存在するのではなく、テンソルの各シャードが指定されたGPUに存在します。処理中に、各シャードは別々に並行して処理され、異なるGPUで同期され、ステップの最後に結果が同期されます。これは水平並列処理と呼ばれるもので、分割は水平レベルで行われます。
3. **PipelineParallel (PP)** - モデルは垂直（レイヤーレベル）に複数のGPUに分割され、モデルの単一または複数のレイヤーが単一のGPUに配置されます。各GPUはパイプラインの異なるステージを並行して処理し、バッチの小さなチャンクで作業します。
4. **Zero Redundancy Optimizer (ZeRO)** - TPといくらか似たようなテンソルのシャーディングを実行しますが、前向きまたは後向きの計算のためにテンソル全体が再構築されるため、モデルを変更する必要はありません。また、GPUメモリが制限されている場合に補償するためのさまざまなオフロード技術をサポートします。
5. **Sharded DDP** - Sharded DDPは、さまざまなZeRO実装で使用される基本的なZeROコンセプトの別名です。

各コンセプトの詳細に深入りする前に、大規模なインフラストラクチャで大規模なモデルをトレーニングする際の大まかな決定プロセスを見てみましょう。

## Scalability Strategy

**⇨ シングルノード / マルチGPU**
* モデルが単一のGPUに収まる場合：

    1. DDP - 分散データ並列
    2. ZeRO - 状況と使用される構成に応じて速いかどうかが異なります

* モデルが単一のGPUに収まらない場合：

    1. PP
    2. ZeRO
    3. TP

    非常に高速なノード内接続（NVLINKまたはNVSwitchなど）があれば、これらの3つはほぼ同じ速度になるはずで、これらがない場合、PPはTPまたはZeROよりも速くなります。TPの程度も差を生じるかもしれません。特定のセットアップでの勝者を見つけるために実験することが最善です。

    TPはほとんどの場合、単一ノード内で使用されます。つまり、TPサイズ <= ノードごとのGPU数です。

* 最大のレイヤーが単一のGPUに収まらない場合：

    1. ZeROを使用しない場合 - TPを使用する必要があります。PP単独では収まらないでしょう。
    2. ZeROを使用する場合 - "シングルGPU"のエントリと同じものを参照してください

**⇨ マルチノード / マルチGPU**

* ノード間の高速接続がある場合：

    1. ZeRO - モデルへのほとんどの変更が不要です
    2. PP+TP+DP - 通信が少なく、モデルへの大規模な変更が必要です

* ノード間の接続が遅く、GPUメモリがまだ不足している場合：

    1. DP+PP+TP+ZeRO-1

## Data Parallelism

2つのGPUを持つほとんどのユーザーは、`DataParallel`（DP）と`DistributedDataParallel`（DDP）によって提供されるトレーニング速度の向上をすでに享受しています。これらはほぼ自明に使用できるPyTorchの組み込み機能です。一般的に、すべてのモデルで動作するDDPを使用することをお勧めします。DPは一部のモデルで失敗する可能性があるためです。[PyTorchのドキュメンテーション](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html)自体もDDPの使用を推奨しています。

### DP vs DDP

`DistributedDataParallel`（DDP）は通常、`DataParallel`（DP）よりも高速ですが、常にそうとは限りません：
* DPはPythonスレッドベースですが、DDPはマルチプロセスベースです。そのため、GIL（Global Interpreter Lock）などのPythonスレッドの制約がないためです。
* 一方、GPUカード間の遅い相互接続性は、DDPの場合に実際には遅い結果をもたらす可能性があります。

以下は、2つのモード間のGPU間通信の主な違いです：

[DDP](https://pytorch.org/docs/master/notes/ddp.html):

- 開始時、メインプロセスはモデルをGPU 0から他のGPUに複製します。
- それから各バッチごとに:
   1. 各GPUは各自のミニバッチのデータを直接消費します。
   2. `backward`中、ローカル勾配が準備できると、それらはすべてのプロセスで平均化されます。

[DP](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html):

各バッチごとに:
   1. GPU 0はデータバッチを読み取り、それから各GPUにミニバッチを送信します。
   2. GPU 0から各GPUに最新のモデルを複製します。
   3. `forward`を実行し、各GPUからGPU 0に出力を送信し、損失を計算します。
   4. GPU 0からすべてのGPUに損失を分散し、`backward`を実行します。
   5. 各GPUからGPU 0に勾配を送信し、それらを平均化します。

DDPはバッチごとに行う通信は勾配の送信のみであり、一方、DPはバッチごとに5つの異なるデータ交換を行います。

DPはプロセス内でデータをPythonスレッドを介してコピーしますが、DDPは[torch.distributed](https://pytorch.org/docs/master/distributed.html)を介してデータをコピーします。

DPではGPU 0は他のGPUよりもはるかに多くの作業を行うため、GPUの未使用率が高くなります。

DDPは複数のマシン間で使用できますが、DPの場合はそうではありません。

DPとDDPの他にも違いがありますが、この議論には関係ありません。

これら2つのモードを深く理解したい場合、この[記事](https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/)を強くお勧めします。素晴らしいダイアグラムを含み、さまざまなハードウェアでの複数のベンチマークとプロファイラの出力を示し、知っておく必要があるすべての微妙なニュアンスを説明しています。

実際のベンチマークを見てみましょう：

| Type   | NVlink | Time |
| :----- | -----  | ---: |
| 2:DP   | Y      | 110s |
| 2:DDP  | Y      | 101s |
| 2:DDP  | N      | 131s |


解析：

ここで、DPはNVlinkを使用したDDPに比べて約10％遅く、NVlinkを使用しないDDPに比べて約15％高速であることが示されています。

実際の違いは、各GPUが他のGPUと同期する必要があるデータの量に依存します。同期するデータが多いほど、遅いリンクが合計の実行時間を遅くする可能性が高くなります。

以下は完全なベンチマークコードと出力です：

`NCCL_P2P_DISABLE=1`を使用して、対応するベンチマークでNVLink機能を無効にしました。


```bash

# DP
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
python examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path openai-community/gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 110.5948, 'train_samples_per_second': 1.808, 'epoch': 0.69}

# DDP w/ NVlink
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path openai-community/gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVlink
rm -r /tmp/test-clm; NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path openai-community/gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

ハードウェア: 2x TITAN RTX、各24GB + 2つのNVLink（`nvidia-smi topo -m`で `NV2`）

ソフトウェア: `pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`

## ZeRO Data Parallelism

ZeROパワードデータ並列処理（ZeRO-DP）は、次の[ブログ投稿](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)のダイアグラムで説明されています。
![DeepSpeed-Image-1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png)

これは理解が難しいかもしれませんが、実際にはこの概念は非常にシンプルです。これは通常の`DataParallel`（DP）ですが、完全なモデルパラメータ、勾配、およびオプティマイザの状態を複製する代わりに、各GPUはそれぞれのスライスのみを保存します。そして、実行時に、特定のレイヤーに必要な完全なレイヤーパラメータが必要な場合、すべてのGPUが同期して、お互いに不足している部分を提供します。それがすべてです。

3つのレイヤーからなる単純なモデルを考えてみましょう。各レイヤーには3つのパラメータがあります：


```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```

レイヤーLaには、重みa0、a1、およびa2があります。

3つのGPUがある場合、Sharded DDP（= Zero-DP）はモデルを3つのGPUに次のように分割します：


```
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

これは、典型的なディープニューラルネットワーク（DNN）のダイアグラムを想像すると、テンソル並列処理と同様の水平スライスであるようなものです。垂直スライスは、異なるGPUに完全な層グループを配置する方法です。しかし、これは単なる出発点に過ぎません。

これから、各GPUは通常のデータ並列処理（DP）と同様に、通常のミニバッチを受け取ります：

```
x0 => GPU0
x1 => GPU1
x2 => GPU2
```

最初に、入力データはレイヤーLaに適用されます。

GPU0に焦点を当てましょう：x0は、その前向きパスを実行するためにa0、a1、a2のパラメータが必要ですが、GPU0にはa0しかありません。GPU1からa1を、GPU2からa2を受け取り、モデルの各部分をまとめます。

同様に、GPU1はミニバッチx1を受け取り、a1しか持っていませんが、a0とa2のパラメータが必要です。これらはGPU0とGPU2から取得します。

GPU2もx2を受け取ります。a0とa1はGPU0とGPU1から受け取り、a2とともに完全なテンソルを再構築します。

3つのGPUは完全なテンソルを再構築し、前向き計算が行われます。

計算が完了すると、不要になったデータは削除されます。計算中だけ使用され、再構築は事前にフェッチを使用して効率的に行われます。

そして、このプロセス全体がレイヤーLb、次に前向きでLc、そして逆方向でLc -> Lb -> Laに対して繰り返されます。

私にとって、これは効率的なグループでの重みの分散戦略のように聞こえます：

1. 人Aはテントを持っています。
2. 人Bはストーブを持っています。
3. 人Cは斧を持っています。

今、彼らは毎晩持っているものを共有し、他の人から持っていないものをもらい、朝には割り当てられたタイプのギアを詰めて旅を続けます。これがSharded DDP / Zero DPです。

この戦略を、各人が独自のテント、ストーブ、斧を持って運ばなければならないシンプルな戦略と比較してみてください。これがPyTorchのDataParallel（DPおよびDDP）です。

このトピックの文献を読む際に、以下の類義語に出会うかもしれません：Sharded、Partitioned。

ZeROがモデルの重みを分割する方法に注意を払うと、これはテンソルパラレリズムと非常に似ているように見えます。これは後で議論される垂直モデルパラレリズムとは異なり、各レイヤーの重みをパーティション/シャーディングします。

Implementations:

- [DeepSpeed](https://www.deepspeed.ai/tutorials/zero/) ZeRO-DP stages 1+2+3
- [`transformers` integration](main_classes/trainer#trainer-integrations)


## Naive Model Parallelism (Vertical) and Pipeline Parallelism

ナイーブモデルパラレリズム（MP）は、モデルの層を複数のGPUに分散させる方法です。このメカニズムは比較的単純で、希望する層を`.to()`メソッドを使用して特定のデバイスに切り替えるだけです。これにより、データがこれらの層を通過するたびに、データも層と同じデバイスに切り替えられ、残りの部分は変更されません。

私たちはこれを「垂直MP」と呼びます。なぜなら、ほとんどのモデルがどのように描かれるかを思い出すと、層を垂直にスライスするからです。たとえば、以下の図は8層のモデルを示しています：


```
===================  ===================
|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
===================  ===================
        gpu0                 gpu1
```

我々は、モデルを垂直に2つに分割し、レイヤー0から3をGPU0に配置し、レイヤー4から7をGPU1に配置しました。

データがレイヤー0から1、1から2、2から3に移動する間は通常のモデルと同じです。しかし、データがレイヤー3からレイヤー4に移動する必要がある場合、GPU0からGPU1への移動が発生し、通信のオーバーヘッドが発生します。参加しているGPUが同じコンピュートノード（例：同じ物理マシン）にある場合、このコピーは非常に高速ですが、異なるコンピュートノード（例：複数のマシン）にある場合、通信のオーバーヘッドは大幅に増加する可能性があります。

その後、レイヤー4から5、6から7までは通常のモデルと同様に動作し、7番目のレイヤーが完了すると、データをしばしばレイヤー0に戻す必要があります（またはラベルを最後のレイヤーに送信します）。これで損失を計算し、オプティマイザが作業を開始できます。

問題点：
- 主な欠点、およびなぜこれを「単純な」MPと呼ぶのかは、1つを除いてすべてのGPUがどんな瞬間でもアイドル状態であることです。したがって、4つのGPUを使用する場合、単純なMPは、1つのGPUのメモリ容量を4倍にするのとほぼ同じであり、ハードウェアの残りを無視します。さらに、データのコピーのオーバーヘッドがあることを忘れてはいけません。したがって、4枚の6GBのカードは、データのコピーのオーバーヘッドがない1枚の24GBのカードと同じサイズを収容できるでしょうが、後者はトレーニングをより迅速に完了します。ただし、たとえば40GBのカードがあり、45GBのモデルを収める必要がある場合、勾配とオプティマイザの状態のためにほとんど収めることができません。
- 共有の埋め込みは、GPU間でコピーする必要があるかもしれません。

パイプライン並列処理（PP）は、ほぼ単純なMPと同じですが、GPUがアイドル状態になる問題を解決し、入力バッチをマイクロバッチに分割し、パイプラインを人工的に作成することにより、異なるGPUが計算プロセスに同時に参加できるようにします。

以下は、[GPipe論文](https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html)からの図で、上部には単純なMP、下部にはPPが示されています：

![mp-pp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-gpipe-bubble.png)

この図から、PPがGPUがアイドル状態の領域である「バブル」を少なく持つことがわかります。アイドル状態の部分は「バブル」と呼ばれます。

図の両方の部分は、4つのGPUがパイプラインに参加している4の次元の並列性を示しています。つまり、4つのパイプステージF0、F1、F2、F3のフォワードパスがあり、逆順のバックワードパスB3、B2、B1、B0があります。

PPは調整する新しいハイパーパラメータを導入します。それは `chunks` で、同じパイプステージを通じて連続して送信されるデータのチャンクの数を定義します。たとえば、下の図では `chunks=4` が表示されています。GPU0はチャンク0、1、2、3（F0,0、F0,1、F0,2、F0,3）で同じフォワードパスを実行し、他のGPUが作業を開始し始めるのを待ってから、GPU0はチャンク3、2、1、0（B0,3、B0,2、B0,1、B0,0）で逆順パスを実行します。

注意すべきは、概念的にはこれが勾配蓄積ステップ（GAS）と同じコンセプトであることです。PyTorchは `chunks` を使用し、DeepSpeedは同じハイパーパラメータをGASと呼びます。

`chunks` の導入により、PPはマイクロバッチ（MBS）の概念を導入します。DPはグローバルデータバッチサイズをミニバッチに分割します。したがって、DPの次数が4で、グローバルバッチサイズが1024の場合、4つのミニバッチ（それぞれ256）に分割されます（1024/4）。そして、`chunks`（またはGAS）の数が32である場合、マイクロバッチサイズは8になります（256/32）。各パイプラインステージは1つのマイクロバッチで作業します。

DP + PPセットアップのグローバルバッチサイズを計算するには、`mbs*chunks*dp_degree`（`8*32*4=1024`）を行います。

図に戻りましょう。

`chunks=1` であれば、非効率な単純なMPになります。非常に大きな `chunks` 値を使用すると、非常に小さなマイクロバッチサイズになり、効率があまり高くないかもしれません。したがって、GPUの効率的な利用を最大化する値を見つけるために実験する必要があります。これは、バブルのサイズを最小限にすることに対応する、すべての参加GPUにわたる高い並行GPU利用を可能にするためです。

2つのソリューショングループがあります。従来のパイプラインAPIソリューションと、ユーザーのモデルを大幅に変更する必要があるより現代的なソリューションです。

従来のパイプラインAPIソリューション：
- PyTorch
- DeepSpeed
- Megatron-LM

現代的なソリューション：
- Varuna
- Sagemaker

従来のパイプラインAPIソリューションの問題点：
- モデルをかなり変更する必要があるため、Pipelineはモジュールの通常のフローを`nn.Sequential`シーケンスに再書き込む必要があり、モデルの設計を変更することが必要です。
- 現在、Pipeline APIは非常に制限的です。最初のパイプラインステージに渡されるPython変数のセットがある場合、回避策を見つける必要があります。現在、パイプラインインターフェースでは、唯一のテンソルまたはテンソルのタプルを入力と出力として要求しています。これらのテンソルはバッチサイズを最初の次元として持っている必要があります。パイプラインはミニバッチをマイクロバッチに分割します。可能な改善点については、こちらの議論が行われています：https://github.com/pytorch/pytorch/pull/50693
- パイプステージのレベルでの条件付き制御フローは不可能です。例えば、T5のようなエンコーダーデコーダーモデルは、条件付きエンコーダーステージを処理するために特別な回避策が必要です。
- 各レイヤーを配置する必要があるため、1つのモデルの出力が他のモデルの入力になるようにします。

VarunaとSageMakerとの実験はまだ行っていませんが、彼らの論文によれば、上記で述べた問題のリストを克服し、ユーザーのモデルにははるかに小さな変更しか必要としないと報告されています。

実装：

- [Pytorch](https://pytorch.org/docs/stable/pipeline.html) (initial support in pytorch-1.8, and progressively getting improved in 1.9 and more so in 1.10). Some [examples](https://github.com/pytorch/pytorch/blob/master/benchmarks/distributed/pipeline/pipe.py)
- [DeepSpeed](https://www.deepspeed.ai/tutorials/pipeline/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) has an internal implementation - no API.
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972) - this is a proprietary solution that can only be used on AWS.
- [OSLO](https://github.com/tunib-ai/oslo) - この実装は、Hugging Face Transformersに基づいています。

🤗 Transformersのステータス: この執筆時点では、いずれのモデルも完全なPP（パイプライン並列処理）をサポートしていません。GPT2モデルとT5モデルは単純なMP（モデル並列処理）サポートを持っています。主な障害は、モデルを`nn.Sequential`に変換できず、すべての入力がテンソルである必要があることです。現在のモデルには、変換を非常に複雑にする多くの機能が含まれており、これらを削除する必要があります。

他のアプローチ：

DeepSpeed、Varuna、およびSageMakerは、[交互にパイプラインを実行](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features.html)するコンセプトを使用しています。ここでは、バックワードパスを優先させてバブル（アイドル時間）をさらに最小限に抑えます。

Varunaは、最適なスケジュールを発見するためにシミュレーションを使用してスケジュールをさらに改善しようとします。

OSLOは、`nn.Sequential`の変換なしでTransformersに基づくパイプライン並列処理を実装しています。

## Tensor Parallelism

テンソル並列処理では、各GPUがテンソルのスライスのみを処理し、全体が必要な操作のためにのみ完全なテンソルを集約します。

このセクションでは、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)論文からのコンセプトと図を使用します：[GPUクラスタでの効率的な大規模言語モデルトレーニング](https://arxiv.org/abs/2104.04473)。

どのトランスフォーマの主要な構築要素は、完全に接続された`nn.Linear`に続く非線形アクティベーション`GeLU`です。

Megatronの論文の表記法に従って、行列の乗算部分を`Y = GeLU(XA)`と書くことができます。ここで、`X`と`Y`は入力ベクトルと出力ベクトルで、`A`は重み行列です。

行列の計算を行列形式で見ると、行列乗算を複数のGPUで分割できる方法が簡単に理解できます：
![Parallel GEMM](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_gemm.png)

重み行列`A`を`N`個のGPUに対して列ごとに分割し、並列で行列乗算`XA_1`から`XA_n`を実行すると、`N`個の出力ベクトル`Y_1、Y_2、...、Y_n`が得られ、それらを独立して`GeLU`に供給できます：
![独立したGeLU](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-independent-gelu.png)

この原理を使用して、最後まで同期が必要ないまま、任意の深さのMLPを更新できます。Megatron-LMの著者はそのための有用なイラストを提供しています：
![並列シャード処理](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_shard_processing.png)

マルチヘッドアテンションレイヤーを並列化することはさらに簡単です。それらは既に複数の独立したヘッドを持っているため、本質的に並列です！
![並列セルフアテンション](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_self_attention.png)

特別な考慮事項：TPには非常に高速なネットワークが必要であり、したがって1つのノードを超えてTPを実行しないことがお勧めされません。実際には、1つのノードに4つのGPUがある場合、最大のTP度数は4です。TP度数8が必要な場合は、少なくとも8つのGPUを持つノードを使用する必要があります。

このセクションは、元のより詳細な[TPの概要](https://github.com/huggingface/transformers/issues/10321#issuecomment-783543530)に基づいています。
by [@anton-l](https://github.com/anton-l)。

SageMakerは、より効率的な処理のためにTPとDPを組み合わせて使用します。

代替名：
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)はこれを「テンソルスライシング」と呼びます。詳細は[DeepSpeedの特徴](https://www.deepspeed.ai/training/#model-parallelism)をご覧ください。

実装例:
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)には、モデル固有の内部実装があります。
- [parallelformers](https://github.com/tunib-ai/parallelformers)（現時点では推論のみ）。
- [SageMaker](https://arxiv.org/abs/2111.05972) - これはAWSでのみ使用できるプロプライエタリなソリューションです。
- [OSLO](https://github.com/tunib-ai/oslo)には、Transformersに基づいたテンソル並列実装があります。

🤗 Transformersの状況:
- コア: まだコアには実装されていません。
- ただし、推論が必要な場合、[parallelformers](https://github.com/tunib-ai/parallelformers)はほとんどのモデルに対してサポートを提供します。これがコアに実装されるまで、これを使用できます。そして、トレーニングモードもサポートされることを期待しています。
- Deepspeed-Inferenceでは、BERT、GPT-2、およびGPT-NeoモデルをCUDAカーネルベースの高速推論モードでサポートしています。詳細は[こちら](https://www.deepspeed.ai/tutorials/inference-tutorial/)をご覧ください。

## DP+PP

DeepSpeedの[パイプラインチュートリアル](https://www.deepspeed.ai/tutorials/pipeline/)からの次の図は、DPをPPと組み合わせる方法を示しています。

![dp-pp-2d](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero-dp-pp.png)

ここで重要なのは、DPランク0がGPU2を見えなくし、DPランク1がGPU3を見えなくすることです。DPにとって、存在するのはGPU 0 と 1 のみで、それらの2つのGPUのようにデータを供給します。GPU0はPPを使用してGPU2に一部の負荷を「秘密裏に」オフロードし、GPU1も同様にGPU3を支援に引き入れます。

各次元には少なくとも2つのGPUが必要ですので、ここでは少なくとも4つのGPUが必要です。

実装例:
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/tunib-ai/oslo)

🤗 Transformersの状況: まだ実装されていません

## DP+PP+TP

さらに効率的なトレーニングを行うために、3Dパラレリズムを使用し、PPをTPとDPと組み合わせます。これは次の図で示されています。

![dp-pp-tp-3d](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png)

この図は[3Dパラレリズム：兆パラメータモデルへのスケーリング](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)というブログ投稿から取得されたもので、おすすめの読み物です。

各次元には少なくとも2つのGPUが必要ですので、ここでは少なくとも8つのGPUが必要です。

実装例:
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) - DeepSpeedには、さらに効率的なDPであるZeRO-DPと呼ばれるものも含まれています。
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Varuna](https://github.com/microsoft/varuna)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/tunib-ai/oslo)

🤗 Transformersの状況: まだ実装されていません。PPとTPがないため。

## ZeRO DP+PP+TP

DeepSpeedの主要な機能の1つはZeROで、これはDPの拡張機能です。これについてはすでに「ZeROデータ並列化」で説明されています。通常、これは単独で動作する機能で、PPやTPは必要ありません。しかし、PPとTPと組み合わせることもできます。

ZeRO-DPがPPと組み合わされる場合、通常はZeROステージ1（オプティマイザーシャーディング）のみが有効になります。

ZeROステージ2（勾配シャーディング）をパイプライン並列化と組み合わせて使用する理論的な可能性はありますが、性能に悪影響を及ぼします。各マイクロバッチごとに勾配をシャーディングする前に、勾配を集約するための追加のリダクションスキャッター集計が必要で、通信オーバーヘッドが発生する可能性があります。パイプライン並列化の性質上、小さなマイクロバッチが使用され、計算の集中度（マイクロバッチサイズ）をバランスにかけ、パイプラインバブル（マイクロバッチ数）を最小限に抑えることに焦点が当てられています。したがって、これらの通信コストは影響を及ぼすでしょう。

さらに、PPには通常よりも少ない層が含まれており、メモリの節約はそれほど大きくありません。PPは既に勾配サイズを「1/PP」に削減するため、勾配シャーディングの節約は純粋なDPよりもはるかに重要ではありません。

ZeROステージ3も同様の理由で適していません - より多くのノード間通信が必要です。

そして、ZeROを持っているので、もう一つの利点はZeRO-Offloadです。これはステージ1オプティマイザーステートをCPUにオフロードできます。

実装例:
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)と[BigScienceからのMegatron-Deepspeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed)は、前者のリポジトリのフォークです。
- [OSLO](https://github.com/tunib-ai/oslo)

重要な論文:

- [DeepSpeedとMegatronを使用したMegatron-Turing NLG 530Bのトレーニング](https://arxiv.org/abs/2201.11990)

🤗 Transformersの状況: まだ実装されていません。PPとTPがないため。


## FlexFlow

[FlexFlow](https://github.com/flexflow/FlexFlow)は、わずかに異なるアプローチで並列化の問題を解決します。

論文: [Zhihao Jia、Matei Zaharia、Alex Aikenによる "Deep Neural Networksのデータとモデルの並列化を超えて"](https://arxiv.org/abs/1807.05358)

FlexFlowは、サンプル-オペレータ-属性-パラメータの4D並列化を行います。

1. サンプル = データ並列化（サンプル単位の並列化）
2. オペレータ = 単一の操作をいくつかのサブ操作に並列化
3. 属性 = データ並列化（長さ方向の並列化）
4. パラメータ = モデル並列化（次元に関係なく、水平または垂直）

例:
* サンプル

シーケンス長512の10バッチを考えてみましょう。これらをサンプル次元で2つのデバイスに並列化すると、10 x 512が5 x 2 x 512になります。

* オペレータ

層正規化を行う場合、まずstdを計算し、次にmeanを計算し、データを正規化できます。オペレータの並列化により、stdとmeanを並列に計算できます。したがって、オペレータ次元で2つのデバイス（cuda:0、cuda:1）に並列化すると、最初に入力データを両方のデバイスにコピーし、cuda:0でstdを計算し、cuda:1でmeanを同時に計算します。

* 属性

10バッチの512長があります。これらを属性次元で2つのデバイスに並列化すると、10 x 512が10 x 2 x 256になります。

* パラメータ

これはテンソルモデルの並列化または単純な層ごとのモデルの並列化と似ています。

このフレームワークの重要性は、（1）GPU/TPU/CPU対（2）RAM/DRAM対（3）高速内部接続/低速外部接続などのリソースを取り、これらすべてをアルゴリズムによって自動的に最適化することです。どの並列化をどこで使用するかをアルゴリズム的に決定します。

非常に重要な側面の1つは、FlexFlowは静的で固定のワークロードを持つモデルのために設計されており、動的な動作を持つモデルはイテレーションごとに異なる並列化戦略を好む場合があることです。

したがって、このフレームワークの約束は非常に魅力的です。選択したクラスタで30分間のシミュレーションを実行し、この特定の環境を最適に利用するための最良の戦略を提供します。部分を追加/削除/置換すると、それに対して実行して再最適化プランを作成します。その後、トレーニングできます。異なるセットアップには独自の最適化があります。

🤗 Transformersの現在の状況: まだ統合されていません。すでに[transformers.utils.fx](https://github.com/huggingface/transformers/blob/master/src/transformers/utils/fx.py)を使用してモデルがFXトレース可能であるため、FlexFlowを動作させるために必要な手順を誰かが見つける必要があります。

## Which Strategy To Use When

ここでは、どの並列化戦略をいつ使用するかの非常におおまかなアウトラインを示します。各リストの最初が通常よりも速いことが一般的です。

**⇨ 単一GPU**

* モデルが単一GPUに収まる場合：

    1. 通常の使用

* モデルが単一GPUに収まらない場合：

    1. ZeRO + CPUをオフロードし、オプションでNVMeをオフロード
    2. 上記に加えて、最大のレイヤーが単一GPUに収まらない場合、[Memory Centric Tiling](https://deepspeed.readthedocs.io/en/latest/zero3.html#memory-centric-tiling)（詳細は以下参照）を有効化

* 最大のレイヤーが単一GPUに収まらない場合：

    1. ZeROを使用しない場合 - TPを有効化する必要があります。なぜなら、PPだけでは収めることができないからです。
    2. ZeROを使用する場合は、上記の「単一GPU」のエントリと同じものを参照してください

**⇨ 単一ノード/マルチGPU**

* モデルが単一GPUに収まる場合：

    1. DDP - 分散データ並列
    2. ZeRO - 状況と使用される構成に依存して速いかどうかが異なることがあります

* モデルが単一GPUに収まらない場合：

    1. PP
    2. ZeRO
    3. TP

    非常に高速なノード内接続がNVLINKまたはNVSwitchである場合、これらのすべてはほとんど同等の性能です。これらがない場合、PPはTPまたはZeROよりも速くなります。TPの度合いも違いを生じるかもしれません。特定のセットアップで勝者を見つけるために実験するのが最善です。

    TPはほとんど常に単一ノード内で使用されます。つまり、TPサイズ <= ノードあたりのGPUです。

* 最大のレイヤーが単一GPUに収まらない場合：

    1. ZeROを使用しない場合 - TPを使用する必要があります。なぜなら、PPだけでは収めることができないからです。
    2. ZeROを使用する場合は、上記の「単一GPU」のエントリと同じものを参照してください

**⇨ マルチノード/マルチGPU**

* 高速なノード間接続がある場合：

    1. ZeRO - モデルへのほとんどの変更が不要です
    2. PP+TP+DP - 通信が少なく、モデルに大規模な変更が必要です

* 遅いノード間接続があり、GPUメモリが少ない場合：

    1. DP+PP+TP+ZeRO-1

