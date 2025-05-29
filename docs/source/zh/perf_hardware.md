<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

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


# 训练用的定制硬件

您用来运行模型训练和推断的硬件可能会对性能产生重大影响。要深入了解 GPU，务必查看 Tim Dettmer 出色的[博文](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)。

让我们来看一些关于 GPU 配置的实用建议。

## GPU
当你训练更大的模型时，基本上有三种选择：

- 更大的 GPU
- 更多的 GPU
- 更多的 CPU 和 NVMe（通过[DeepSpeed-Infinity](main_classes/deepspeed#nvme-support)实现）

让我们从只有一块GPU的情况开始。

### 供电和散热

如果您购买了昂贵的高端GPU，请确保为其提供正确的供电和足够的散热。

**供电**：

一些高端消费者级GPU卡具有2个，有时甚至3个PCI-E-8针电源插口。请确保将与插口数量相同的独立12V PCI-E-8针线缆插入卡中。不要使用同一根线缆两端的2个分叉（也称为pigtail cable）。也就是说，如果您的GPU上有2个插口，您需要使用2条PCI-E-8针线缆连接电源和卡，而不是使用一条末端有2个PCI-E-8针连接器的线缆！否则，您无法充分发挥卡的性能。

每个PCI-E-8针电源线缆需要插入电源侧的12V轨上，并且可以提供最多150W的功率。

其他一些卡可能使用PCI-E-12针连接器，这些连接器可以提供最多500-600W的功率。

低端卡可能使用6针连接器，这些连接器可提供最多75W的功率。

此外，您需要选择具有稳定电压的高端电源。一些质量较低的电源可能无法为卡提供所需的稳定电压以发挥其最大性能。

当然，电源还需要有足够的未使用的瓦数来为卡供电。

**散热**：

当GPU过热时，它将开始降频，不会提供完整的性能。如果温度过高，可能会缩短GPU的使用寿命。

当GPU负载很重时，很难确定最佳温度是多少，但任何低于+80度的温度都是好的，越低越好，也许在70-75度之间是一个非常好的范围。降频可能从大约84-90度开始。但是除了降频外，持续的高温可能会缩短GPU的使用寿命。

接下来让我们看一下拥有多个GPU时最重要的方面之一：连接。

### 多GPU连接

如果您使用多个GPU，则卡之间的互连方式可能会对总训练时间产生巨大影响。如果GPU位于同一物理节点上，您可以运行以下代码：

```bash
nvidia-smi topo -m
```

它将告诉您GPU如何互连。在具有双GPU并通过NVLink连接的机器上，您最有可能看到类似以下内容：

```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV2     0-23            N/A
GPU1    NV2      X      0-23            N/A
```

在不同的机器上，如果没有NVLink，我们可能会看到：
```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-11            N/A
GPU1    PHB      X      0-11            N/A
```

这个报告包括了这个输出：

```
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

因此，第一个报告`NV2`告诉我们GPU通过2个NVLink互连，而第二个报告`PHB`展示了典型的消费者级PCIe+Bridge设置。

检查你的设置中具有哪种连接类型。其中一些会使卡之间的通信更快（例如NVLink），而其他则较慢（例如PHB）。

根据使用的扩展解决方案的类型，连接速度可能会产生重大或较小的影响。如果GPU很少需要同步，就像在DDP中一样，那么较慢的连接的影响将不那么显著。如果GPU经常需要相互发送消息，就像在ZeRO-DP中一样，那么更快的连接对于实现更快的训练变得非常重要。


#### NVlink

[NVLink](https://en.wikipedia.org/wiki/NVLink)是由Nvidia开发的一种基于线缆的串行多通道近程通信链接。

每个新一代提供更快的带宽，例如在[Nvidia Ampere GA102 GPU架构](https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)中有这样的引述：

> Third-Generation NVLink®
> GA102 GPUs utilize NVIDIA’s third-generation NVLink interface, which includes four x4 links,
> with each link providing 14.0625 GB/sec bandwidth in each direction between two GPUs. Four
> links provide 56.25 GB/sec bandwidth in each direction, and 112.5 GB/sec total bandwidth
> between two GPUs. Two RTX 3090 GPUs can be connected together for SLI using NVLink.
> (Note that 3-Way and 4-Way SLI configurations are not supported.)

所以，在`nvidia-smi topo -m`输出的`NVX`报告中获取到的更高的`X`值意味着更好的性能。生成的结果将取决于您的GPU架构。

让我们比较在小样本wikitext上训练gpt2语言模型的执行结果。

结果是：


| NVlink | Time |
| -----  | ---: |
| Y      | 101s |
| N      | 131s |


可以看到，NVLink使训练速度提高了约23%。在第二个基准测试中，我们使用`NCCL_P2P_DISABLE=1`告诉GPU不要使用NVLink。

这里是完整的基准测试代码和输出：

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

硬件: 2x TITAN RTX 24GB each + NVlink with 2 NVLinks (`NV2` in `nvidia-smi topo -m`)
软件: `pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`
