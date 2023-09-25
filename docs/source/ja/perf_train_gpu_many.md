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


```

# DP
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
python examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 110.5948, 'train_samples_per_second': 1.808, 'epoch': 0.69}

# DDP w/ NVlink
rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVlink
rm -r /tmp/test-clm; NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py \
--model_name_or_path gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
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

- [DeepSpeed](https://www.deepspeed.ai/features/#the-zero-redundancy-optimizer) ZeRO-DP stages 1+2+3
- [`transformers` integration](main_classes/trainer#trainer-integrations)


