<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Custom hardware for training

モデルのトレーニングおよび推論に使用するハードウェアは、パフォーマンスに大きな影響を与えることがあります。GPUについて詳しく知りたい場合は、Tim Dettmerの優れた[ブログ記事](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)をチェックしてみてください。

GPUセットアップの実用的なアドバイスをいくつか見てみましょう。

## GPU
より大きなモデルをトレーニングする場合、基本的には以下の3つのオプションがあります：

- より大きなGPU
- より多くのGPU
- より多くのCPUおよびNVMe（[DeepSpeed-Infinity](main_classes/deepspeed#nvme-support)によるオフロード）

まず、単一のGPUを使用する場合から始めましょう。

### Power and Cooling

高価なハイエンドGPUを購入した場合、正しい電力供給と十分な冷却を提供することが重要です。

**電力**：

一部の高級コンシューマGPUカードには、2つまたは3つのPCI-E 8ピン電源ソケットがあります。カードにあるソケットの数だけ、独立した12V PCI-E 8ピンケーブルが接続されていることを確認してください。同じケーブルの一端にある2つの分岐（またはピッグテールケーブルとしても知られています）を使用しないでください。つまり、GPUに2つのソケットがある場合、PSUからカードに向けて2つのPCI-E 8ピンケーブルを使用し、1つのケーブルの端に2つのPCI-E 8ピンコネクタがあるものは使用しないでください！そうしないと、カードからのパフォーマンスを十分に引き出すことができません。

各PCI-E 8ピン電源ケーブルは、PSU側の12Vレールに接続する必要があり、最大で150Wの電力を供給できます。

一部のカードはPCI-E 12ピンコネクタを使用することがあり、これらは最大で500-600Wの電力を供給できます。

低価格帯のカードは6ピンコネクタを使用することがあり、最大で75Wの電力を供給します。

さらに、カードが必要とする安定した電圧を提供する高品質な電源ユニット（PSU）を使用する必要があります。

もちろん、PSUにはカードを駆動するために十分な未使用の電力が必要です。

**冷却**：

GPUが過熱すると、スロットリングが開始され、フルパフォーマンスを提供しなくなり、過熱しすぎるとシャットダウンすることさえあります。

GPUが重要な負荷の下でどのような温度を目指すべきかを正確に示すことは難しいですが、おそらく+80℃未満であれば良いでしょうが、それより低い方が良いです - おそらく70-75℃が優れた範囲でしょう。スロットリングの開始温度はおそらく84-90℃のあたりからでしょう。スロットリングによるパフォーマンスの低下以外にも、長時間にわたる非常に高い温度はGPUの寿命を短縮する可能性があります。

次に、複数のGPUを持つ際に最も重要な側面の一つである接続について詳しく見てみましょう。

### Multi-GPU Connectivity

複数のGPUを使用する場合、カードの相互接続方法はトータルのトレーニング時間に大きな影響を与える可能性があります。GPUが同じ物理ノードにある場合、次のように実行できます：


```bash
nvidia-smi topo -m
```

もちろん、GPUがどのように相互接続されているかについて説明します。デュアルGPUを搭載し、NVLinkで接続されているマシンでは、おそらく以下のような情報が表示されるでしょう：

```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV2     0-23            N/A
GPU1    NV2      X      0-23            N/A
```

別のNVLinkなしのマシンでは、以下のような状況が発生するかもしれません：

```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-11            N/A
GPU1    PHB      X      0-11            N/A
```

こちらが伝説です：

```
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

最初のレポートである `NV2` では、GPUは2つのNVLinkで接続されており、2番目のレポートである `PHB` では、典型的な消費者向けのPCIe+Bridgeセットアップが行われています。

あなたのセットアップでどの種類の接続性があるかを確認してください。これらの接続方法のいくつかはカード間の通信を速くすることができます（例：NVLink）、他のものは遅くすることができます（例：PHB）。

使用されるスケーラビリティソリューションの種類に応じて、接続速度は大きな影響を与えることも、小さな影響を与えることもあります。GPUがあまり頻繁に同期する必要がない場合、DDPのように、遅い接続の影響はそれほど重要ではありません。しかし、GPUが頻繁にメッセージを送信する必要がある場合、ZeRO-DPのように、高速の接続がより高速なトレーニングを実現するために非常に重要になります。

#### NVlink

[NVLink](https://en.wikipedia.org/wiki/NVLink) は、Nvidiaによって開発された有線のシリアルマルチレーンの近距離通信リンクです。

各新世代では、より高速な帯域幅が提供されます。たとえば、[Nvidia Ampere GA102 GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf) からの引用です。

> Third-Generation NVLink®
> GA102 GPUs utilize NVIDIA’s third-generation NVLink interface, which includes four x4 links,
> with each link providing 14.0625 GB/sec bandwidth in each direction between two GPUs. Four
> links provide 56.25 GB/sec bandwidth in each direction, and 112.5 GB/sec total bandwidth
> between two GPUs. Two RTX 3090 GPUs can be connected together for SLI using NVLink.
> (Note that 3-Way and 4-Way SLI configurations are not supported.)

したがって、`nvidia-smi topo -m` の出力の `NVX` レポートで取得する `X` が高いほど良いです。世代はあなたのGPUアーキテクチャに依存します。

小さなサンプルのwikitextを使用したgpt2言語モデルのトレーニングの実行を比較しましょう。

結果は次のとおりです：

（ここに結果を挿入）

上記のテキストの日本語訳を提供しました。Markdownコードとしてフォーマットしました。どんな他の質問があれば、お気軽にお知らせください！

| NVlink | Time |
| -----  | ---: |
| Y      | 101s |
| N      | 131s |


NVLinkを使用すると、トレーニングが約23％速く完了することがわかります。2番目のベンチマークでは、`NCCL_P2P_DISABLE=1`を使用して、GPUがNVLinkを使用しないように指示しています。

以下は、完全なベンチマークコードと出力です：


```bash
# DDP w/ NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path openai-community/gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 torchrun \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path openai-community/gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

Hardware: 2x TITAN RTX 24GB each + NVlink with 2 NVLinks (`NV2` in `nvidia-smi topo -m`)
Software: `pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`
