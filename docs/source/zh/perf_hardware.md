<!---版权所有 2022 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证的规定，否则您不得使用此文件。您可以在
    http://www.apache.org/licenses/LICENSE-2.0
根据适用法律或书面协议要求，软件在许可证下发布，基于“按原样”发布，不附加任何形式的担保或条件。有关特定语言的许可证条款和限制条件，请参阅许可证。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->

# 用于训练的自定义硬件

您用于运行模型训练和推理的硬件对性能有很大影响。要深入了解 GPU，请务必查看 Tim Dettmer 的出色 [博文](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)。
让我们来看一些关于 GPU 设置的实用建议。

## GPU 

当您训练更大的模型时，您基本上有三个选择：- 更大的 GPU- 更多的 GPU- 更多的 CPU 和 NVMe（由 [DeepSpeed-Infinity](main_classes/deepspeed#nvme-support) 卸载）
让我们从只有一个 GPU 的情况开始。

### 电源和散热

如果您购买了昂贵的高端 GPU，请确保为其提供正确的电源和足够的散热。

**电源：**

一些高端消费级 GPU 卡有 2 个甚至 3 个 PCI-E 8 针电源插座。请确保您插入与插座数量相同的独立 12V PCI-E 8 针电缆到显卡上。不要使用相同电缆末端上的 2 个分支（也称为小头电缆）。也就是说，如果您的 GPU 上有 2 个插座，您需要将 2 根 PCI-E 8 针电缆从电源插入显卡，而不是一根末端带有 2 个 PCI-E 8 针连接器的电缆！否则，您将无法充分发挥显卡的性能。

每根 PCI-E 8 针电源电缆需要插入 PSU 侧的一个 12V 电轨，可提供高达 150W 的功率。

其他一些显卡可能使用 PCI-E 12 针连接器，这些连接器可以提供高达 500-600W 的功率。

低端显卡可能使用 6 针连接器，可提供高达 75W 的功率。

此外，您需要具备稳定电压的高端电源。一些质量较低的电源可能无法为显卡提供所需的稳定电压以实现最佳性能。
当然，电源需要有足够的未使用功率来为显卡供电。

**散热：**

当 GPU 过热时，它将开始降频，无法提供完整的性能，甚至可能在温度过高时关闭。

很难确定在 GPU 负载严重的情况下应该追求的最佳温度，但可能在+80 ° C 以下都是不错的选择，更低的温度是更好的选择 - 也许在 70-75 ° C 的范围内是一个很好的区间。降频可能在 84-90 ° C 左右开始。但除了降低性能外，长时间高温还可能缩短 GPU 的使用寿命。

接下来，让我们来看看拥有多个 GPU 时最重要的方面之一：连接性。

### 多 GPU 连接

如果您使用多个 GPU，卡之间的连接方式对总体训练时间有很大影响。如果 GPU 在同一物理节点上，您可以运行：
```
nvidia-smi topo -m
```

它将告诉您 GPU 之间的连接方式。在具有双 GPU 且使用 NVLink 连接的机器上，您很可能会看到类似以下的输出：
```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV2     0-23            N/A
GPU1    NV2      X      0-23            N/A
```

在没有 NVLink 的不同机器上，可能会看到以下输出：
```
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-11            N/A
GPU1    PHB      X      0-11            N/A
```

报告包括以下说明：
```
  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

因此，第一个报告“NV2”告诉我们 GPU 之间使用 2 个 NVLink 进行连接，第二个报告“PHB”表示我们使用了典型的消费级 PCIe+Bridge 设置。

检查您的设置上使用的连接类型。其中一些将加快卡之间的通信速度（例如 NVLink），而其他一些将减慢通信速度（例如 PHB）。

根据使用的可扩展性解决方案类型，连接速度可能会有很大或很小的影响。如果 GPU 需要很少进行同步，如在 DDP 中，较慢的连接影响将不太显著。如果 GPU 需要经常发送消息，如在 ZeRO-DP 中，更快的连接速度变得非常重要，以实现更快的训练。


#### NVlink

[NVLink](https://en.wikipedia.org/wiki/NVLink) 是由 Nvidia 开发的一种基于线缆的串行多通道近距离通信链接。

每一代新的 GPU 提供更快的带宽，例如下面是来自 [Nvidia Ampere GA102 GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf) 的引用：

> 第三代 NVLink ® > GA102 GPU 使用 NVIDIA 的第三代 NVLink 接口，包括四个 x4 链接，> 每个链接在两个 GPU 之间的每个方向上提供 14.0625 GB/秒的带宽。四个 > 链接在每个方向上提供 56.25 GB/秒的带宽，总带宽为 112.5 GB/秒 > 两个 GPU 之间。两个 RTX 3090 GPU 可以使用 NVLink 连接在一起进行 SLI。
> （请注意，不支持 3 路和 4 路 SLI 配置。）

因此，在 `nvidia-smi topo -m` 的输出中，`NVX` 报告中的较高的 `X` 值越好。代数将取决于您的 GPU 架构。

让我们比较使用 gpt2 语言模型在一个小的 wikitext 样本上进行训练的执行情况。

结果如下：
| NVlink | Time |
| -----  | ---: |
| Y      | 101s |
| N      | 131s |

您可以看到，使用 NVLink 完成训练的时间约快 23%。在第二个基准测试中，我们使用 `NCCL_P2P_DISABLE=1` 告诉 GPU 不使用 NVLink。

以下是完整的基准测试代码和输出：

```bash
# DDP w/ NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train \
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 101.9003, 'train_samples_per_second': 1.963, 'epoch': 0.69}

# DDP w/o NVLink

rm -r /tmp/test-clm; CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 python -m torch.distributed.launch \
--nproc_per_node 2 examples/pytorch/language-modeling/run_clm.py --model_name_or_path gpt2 \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train
--output_dir /tmp/test-clm --per_device_train_batch_size 4 --max_steps 200

{'train_runtime': 131.4367, 'train_samples_per_second': 1.522, 'epoch': 0.69}
```

硬件：2x TITAN RTX 24GB + NVlink（使用 2 个 NVLink 连接）

软件：`pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`