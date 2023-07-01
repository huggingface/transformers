<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“按原样”基础分发的，不附带任何形式的保证或条件，无论是明示的还是隐含的。有关许可证的详细信息
⚠️ 请注意，此文件使用 Markdown 编写，但包含我们 doc-builder（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->

# 多 GPU 高效训练

当在单个 GPU 上训练速度太慢或模型权重无法容纳在单个 GPU 的内存中时，我们使用多 GPU 设置。从单个 GPU 切换到多个 GPU 需要某种形式的并行处理，因为工作需要分布。有多种技术可以实现并行处理，例如数据、张量或流水线并行处理。然而，并无一种解决方案适用于所有情况，最佳设置取决于您运行的硬件。虽然主要概念很可能适用于任何其他框架，但本文重点关注基于 PyTorch 的实现。
<Tip>

 注意：在深入研究以下部分（如多 GPU 或 CPU 训练）之前，大多数在 [单个 GPU 部分](perf_train_gpu_one) 介绍的策略（如混合精度训练或梯度积累）是通用的，并适用于训练模型，所以请务必查看一下。
</Tip>

我们首先详细讨论各种 1D 并行处理技术及其优缺点，然后再看看如何将它们组合成 2D 和 3D 并行处理，以实现更快的训练并支持更大的模型。还将介绍其他强大的替代方法。

## 概念

以下是将在本文档中深入描述的主要概念的简要说明。

1. **数据并行（DP）** - 多个相同设置被复制多次，并且每个设置都被提供数据的一个切片。处理是并行进行的，并且在每个训练步骤结束时，所有设置都进行同步。
2. **张量并行（TP）** - 每个张量被分成多个块，因此每个张量的分片都位于其指定的 GPU 上。在处理过程中，每个分片在不同的 GPU 上并行处理，结果在步骤结束时同步。这可以称为水平并行，因为分割发生在水平层面上。3. **流水线并行（PP）** - 模型在多个 GPU 上垂直（层级）拆分，以便只有一个或几个模型层位于单个 GPU 上。每个 GPU 并行处理管道的不同阶段，并在小批次上工作。4. **Zero Redundancy Optimizer（ZeRO）** - 也执行张量的分片，与 TP 类似，但整个张量在前向或后向计算时重新构造，因此不需要修改模型。它还支持各种卸载技术，以弥补有限的 GPU 内存。
5. **Sharded DDP** - 是各种其他 ZeRO 实现中使用的基础 ZeRO 概念的另一个名称。

在更深入地了解每个概念的具体细节之前，我们首先看一下在大型基础架构上训练大型模型时的大致决策过程。

## 可扩展性策略

**⇨ 单节点/多 GPU*** 模型适合单个 GPU：

    1. DDP - 分布式 DP   
    2. ZeRO - 根据情况和配置，可能更快或更慢
* 模型不适合单个 GPU：
    1. PP    
    2. ZeRO    
    3. TP
    使用非常快的 NVLINK 或 NVSwitch 进行内节点连接时，这三种方法应该基本相当，没有这些，PP 将比 TP 或 ZeRO 快。TP 的程度也可能会有所不同。最好进行实验，找到适合您特定设置的优胜者。
    
    TP 几乎始终在单个节点内使用。即 TP 大小 <= 节点的 GPU 数。

* 最大的层不适合单个 GPU：
    1. 如果不使用 ZeRO - 必须使用 TP，因为单独使用 PP 无法容纳。    
    2. 对于 ZeRO，请参阅上面“单个 GPU”条目

**⇨ 多节点/多 GPU**

* 当您拥有快速的节点间连接时：
    1. ZeRO - 因为它对模型几乎没有修改要求    
    2. PP+TP+DP - 通信较少，但需要对模型进行大规模更改

* 当您具有较慢的节点间连接并且 GPU 内存仍较低时：
    1. DP+PP+TP+ZeRO-1


## 数据并行处理
大多数仅使用 2 个 GPU 的用户已经通过 `DataParallel`（DP）和 `DistributedDataParallel`（DDP）获得了提速。这是 PyTorch 的内置功能。请注意，一般建议使用 DDP，因为它得到了更好的维护，并适用于所有模型，而 DP 可能对某些模型无效。[PyTorch 文档](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html) 本身建议使用 DDP。
### DP vs DDP

`DistributedDataParallel`（DDP）通常比 `DataParallel`（DP）更快，但并非总是如此：

* DDP 是基于多进程的，而不是基于 Python 线程 - 因此它没有 Python 线程的限制，例如全局解释器锁（GIL）

* 另一方面，GPU 卡之间的慢速互连可能导致 DDP 的实际速度较慢

以下是两种模式之间的 GPU 间通信开销的主要区别：
[DDP](https://pytorch.org/docs/master/notes/ddp.html)：

- 在启动时，主进程将模型从 GPU 0 复制到其他 GPU- 然后对于每个批次：  
 1. 每个 GPU 直接使用自己的小批次数据   
 2. 在 `backward` 期间，一旦本地梯度准备好，它们就会在所有进程之间平均

[DP](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html)：
对于每个批次：
   1. GPU 0 读取数据批次，然后将小批次发送到每个 GPU   
   2. 从 GPU 0 复制最新的模型到每个 GPU   
   3. 运行 `forward`，并将每个 GPU 的输出发送到 GPU 0，计算损失   
   4. 将损失从 GPU 0 分发到所有 GPU，运行 `backward`   
   5. 将每个 GPU 的梯度发送到 GPU 0 并对其进行平均
DDP 每批次执行的唯一通信是发送梯度，而 DP 每批次执行 5 个不同的数据交换。

DP 通过 Python 线程在进程内部复制数据，而 DDP 通过 [torch.distributed](https://pytorch.org/docs/master/distributed.html) 复制数据。

在 DP 下，GPU 0 执行的工作比其他 GPU 多得多，从而导致 GPU 的利用率不足。

您可以在多台机器上使用 DDP，但对于 DP 则不适用。
DP 和 DDP 之间还有其他差异，但对于本讨论不相关。

如果你想深入了解这两种模式，强烈推荐阅读这篇 [文章](https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/)。它有很好的图表，包含多个基准测试和各种硬件的分析结果，并解释了可能需要了解的所有细微差别。

让我们看一个实际的基准测试：

| Type   | NVlink | Time |
| :----- | -----  | ---: |
| 2: DP   | Y      | 110s |
| 2: DDP  | Y      | 101s |
| 2: DDP  | N      | 131s |

分析：

在这里，DP 比带有 NVlink 的 DDP 慢约 10 ％，但比没有 NVlink 的 DDP 快约 15 ％。

真正的差异将取决于每个 GPU 需要与其他 GPU 同步的数据量-要同步的数据越多，慢速连接将降低总运行时间。
这是完整的基准测试代码和输出：

对应基准测试中使用了 `NCCL_P2P_DISABLE=1` 来禁用 NVLink 功能。

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

硬件：2x TITAN RTX 24GB + 2 个 NVlink（`NV2` 在 `nvidia-smi topo -m` 中）软件：`pytorch-1.8-to-be` + `cuda-11.0` / `transformers==4.3.0.dev0`

## ZeRO 数据并行

下面的图表描述了 ZeRO 数据并行（ZeRO-DP）的过程，参考自这篇 [博文](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)：![DeepSpeed-Image-1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png)

虽然这个概念可能有些难以理解，但实际上它很简单。这只是通常的 `DataParallel`（DP），只是每个 GPU 存储的是模型的一部分，而不是整个模型参数、梯度和优化器状态。在运行时，当给定层需要完整的层参数时，所有 GPU 会同步相互提供它们缺少的部分-就是这样。

考虑这个简单的模型，包含 3 个层，每个层有 3 个参数：
```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```
第一层 La 有权重 a0、a1 和 a2。
如果我们有 3 个 GPU，Sharded DDP（= Zero-DP）会将模型分割到 3 个 GPU 上：
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

从某种意义上说，这与张量并行类似，如果你想象一下典型的 DNN 图表。垂直切分是将整个层组放在不同的 GPU 上。但这只是一个起点。

现在每个 GPU 都会得到与 DP 相同的小批量：
```
x0 => GPU0
x1 => GPU1
x2 => GPU2
```

输入数据没有改变-它们认为它们将由正常的模型进行处理。

首先，输入数据经过 La 层。

让我们只关注 GPU0：x0 需要 a0、a1、a2 参数来进行前向传递，但 GPU0 只有 a0-它从 GPU1 接收 a1 和从 GPU2 接收 a2，将模型的所有部分组合在一起。
并行地，GPU1 获得小批量 x1，它只有 a1，但需要 a0 和 a2 参数，所以它从 GPU0 和 GPU2 获取这些参数。
GPU2 也是同样的情况，它获得输入 x2。它从 GPU0 和 GPU1 获取 a0 和 a1，并使用它的 a2 重构完整的张量。
所有 3 个 GPU 都获得完整的重构张量，并进行前向传递。

计算完成后，不再需要的数据被丢弃-它们只在计算过程中使用。通过预取方式高效地进行重构。

整个过程对于层 Lb，然后是 Lc 的前向传递，以及后向传递 Lc-> Lb-> La 重复进行。

对我来说，这听起来像是一种高效的分组背包权重分配策略：

1. 人 A 携带帐篷 
2. 人 B 携带炉子 
3. 人 C 携带斧子

每晚他们共享彼此拥有的物品，并从其他人那里获得他们没有的物品，早晨他们整理好自己分配到的类型的装备，继续前行。这就是 Sharded DDP / Zero DP。

将这种策略与每个人都必须携带自己的帐篷、炉子和斧头的简单策略进行比较，后者效率更低。这就是 PyTorch 中的 DataParallel（DP 和 DDP）。

在阅读有关此主题的文献时，您可能会遇到以下同义词：Sharded，Partitioned。

如果您仔细观察 ZeRO 如何分割模型的权重-它看起来与后面将讨论的张量并行非常相似。这是因为它将每个层的权重进行了分区/分片，与将整个层组放在不同的 GPU 上的垂直模型并行不同。

实现：

- [DeepSpeed](https://www.deepspeed.ai/features/#the-zero-redundancy-optimizer) ZeRO-DP stages 1+2+3
- [Fairscale](https://github.com/facebookresearch/fairscale/#optimizer-state-sharding-zero) ZeRO-DP stages 1+2+3
- [`transformers` integration](main_classes/trainer#trainer-integrations)
- [Fairscale](https://github.com/facebookresearch/fairscale/#optimizer-state-sharding-zero) ZeRO-DP stages 1+2+3
- [`transformers` integration](main_classes/trainer#trainer-integrations)

## Naive 模型并行（垂直）和 Pipeline 并行

Naive 模型并行（MP）是将模型层组分布在多个 GPU 上的方法。机制相对简单-将所需的层 `.to()` 到所需的设备上，现在每当数据进出这些层时，将数据切换到与层相同的设备，并保持其余部分不变。

我们将其称为垂直 MP，因为如果你记得大多数模型是如何绘制的，我们将层垂直切分。例如，如果以下图表显示一个包含 8 层的模型：
```
===================  ===================
|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
===================  ===================
        gpu0                 gpu1
```
我们只是在垂直方向上将其切成 2 份，将层 0-3 放在 GPU0 上，将层 4-7 放在 GPU1 上。

现在，当数据从第 0 层传递到第 1 层、从第 1 层传递到第 2 层、从第 2 层传递到第 3 层时，这仅仅是正常的模型。但当数据需要从第 3 层传递到第 4 层时，它需要从 GPU0 传递到 GPU1，这会引入通信开销。如果参与的 GPU 在同一计算节点上（例如同一台物理机），复制操作非常快速，但如果 GPU 位于不同的计算节点上（例如多台机器），通信开销可能会显著增加。

然后，层 4 到层 5，再到层 6 到层 7 就像正常的模型一样，当第 7 层完成时，我们通常需要将数据发送回第 0 层，其中包含标签（或者将标签发送到最后一层）。现在可以计算损失并且优化器可以发挥作用。

问题：- 主要的缺陷，也是为什么这被称为“naive” MP 的原因，是除了一个 GPU 外，其余的 GPU 都处于空闲状态。因此，如果使用 4 个 GPU，几乎与将单个 GPU 的内存增加四倍相同，忽略其他硬件。此外，还有在设备之间复制数据的开销。因此，使用 naive MP，4 个 6GB 的卡可以容纳与 1 个 24GB 的卡相同大小的模型，但后者将完成训练更快，因为它没有数据复制开销。但是，假设您有 40GB 的卡，并且需要适应一个 45GB 的模型，您可以使用 4 个 40GB 的卡（但由于梯度和优化器状态的原因，可能勉强适应）- 共享嵌入可能需要在 GPU 之间来回复制。
Pipeline 并行（PP）与 naive MP 几乎相同，但它通过将传入的批次分成微批次并人为地创建一个管道来解决 GPU 空闲问题，这允许不同的 GPU 同时参与计算过程。

下面这张来自 [GPipe 论文](https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html) 的插图展示了顶部的 naive MP 和底部的 PP：
![mp-pp](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-gpipe-bubble.png)
从底部的图表可以清楚地看出，PP 具有较少的死区，即 GPU 处于闲置状态的部分。这些闲置部分被称为 "泡泡"。
图表的两部分都展示了 4 级的并行性。也就是说，有 4 个 GPU 参与到管道中。因此，有 4 个管道阶段的正向路径 F0、F1、F2 和 F3，然后是反向路径 B3、B2、B1 和 B0 的返回顺序。

PP 引入了一个新的超参数来进行调整，即 `chunks`，它定义了在相同的管道阶段中连续发送多少个数据块。例如，在下面的图表中，您可以看到 `chunks=4`。GPU0 在块 0、1、2 和 3（F0,0、F0,1、F0,2、F0,3）上执行相同的正向路径，然后它等待其他 GPU 完成它们的工作，只有当它们的工作开始完成时，GPU0 才开始工作，为块 3、2、1 和 0（B0,3、B0,2、B0,1、B0,0）执行反向路径。
需要注意的是，从概念上讲，这与梯度累积步骤（GAS）的概念是相同的。PyTorch 使用 `chunks`，而 DeepSpeed 将相同的超参数称为 GAS。

由于 `chunks`，PP 引入了微批次（MBS）的概念。DP 将全局数据批次大小分成小批次，因此如果您的 DP 等级为 4，全局批次大小为 1024，则会将其分成 4 个小批次，每个小批次为 256（1024/4）。如果 `chunks`（或 GAS）的数量是 32，那么我们最终得到的微批次大小为 8（256/32）。每个管道阶段一次只处理一个微批次。
为了计算 DP + PP 设置的全局批次大小，我们执行以下计算：`mbs*chunks*dp_degree`（`8*32*4=1024`）。
让我们回到图表上。

当 `chunks=1` 时，您将得到原始的 MP，这是非常低效的。当 `chunks` 值非常大时，您将得到非常小的微批次大小，这也可能不太高效。因此，人们必须进行实验，找到导致 GPU 利用率最高的值，从而最小化泡泡的大小。

虽然图表显示存在一个 "死" 时间的泡泡，因为最后的“正向”阶段必须等待“反向”完成管道，但找到 `chunks` 的最佳值的目的是实现所有参与的 GPU 的高并发 GPU 利用率，从而最小化泡泡的大小。

有两组解决方案-传统的管道 API 和更现代的解决方案，这些解决方案使最终用户的工作更加轻松。
传统的管道 API 解决方案：- PyTorch- FairScale- DeepSpeed- Megatron-LM
现代解决方案：- Varuna- Sagemaker
传统管道 API 解决方案存在的问题：- 必须对模型进行相当大的修改，因为管道要求将模块的正常流程重写为相同的 `nn.Sequential` 序列，这可能需要对模型的设计进行更改。
- 目前，管道 API 的功能非常受限。如果在管道的第一个阶段传递了一组 Python 变量，您将不得不找到解决方法。目前，管道接口要求将单个张量或张量元组作为唯一的输入和输出。这些张量的第一个维度必须是批次大小，因为管道将将小批次划分为微批次。

这里讨论了可能的改进 https://github.com/pytorch/pytorch/pull/50693- 不能在管道阶段的级别上进行条件控制流-例如，像 T5 这样的编码器-解码器模型需要特殊的解决方案来处理条件编码器阶段。- 必须安排每个层，以便一个模型的输出成为另一个模型的输入。

我们尚未尝试过 Varuna 和 SageMaker，但他们的论文报告称，他们已经解决了上述问题列表，并且对用户的模型所需的更改要小得多。

实现：
- [Pytorch](https://pytorch.org/docs/stable/pipeline.html)（pytorch-1.8 中的初始支持，1.9 逐渐改进，1.10 改进更多）。

一些 [示例](https://github.com/pytorch/pytorch/blob/master/benchmarks/distributed/pipeline/pipe.py)
- [FairScale](https://fairscale.readthedocs.io/en/latest/tutorials/pipe.html)
- [DeepSpeed](https://www.deepspeed.ai/tutorials/pipeline/)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 有一个内部实现-无 API。
- [Varuna](https://github.com/microsoft/varuna)- [SageMaker](https://arxiv.org/abs/2111.05972)-这是一种只能在 AWS 上使用的专有解决方案。
- [OSLO](https://github.com/tunib-ai/oslo)-这是基于 Hugging Face Transformers 实现的。

🤗 Transformers 状态：截至目前，没有任何模型支持完全 PP。GPT2 和 T5 模型支持原始 MP。主要障碍是无法将模型转换为 `nn.Sequential` 并使所有输入都成为张量。这是因为当前的模型包括许多复杂的特性，使得转换非常复杂，需要将这些特性删除才能完成转换。

其他方法：

DeepSpeed、Varuna 和 SageMaker 使用 [交织管道](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features.html) 的概念 ![交织管道执行](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-sagemaker-interleaved-pipeline.png)
在这里，通过优先处理反向传播来进一步减少泡泡（闲置时间）。

Varuna 通过使用模拟来发现最高效的调度来进一步改进计划。

OSLO 基于 Transformers 实现了管道并行性，而无需进行 `nn.Sequential` 转换。

## 张量并行

在张量并行中，每个 GPU 仅处理张量的一部分，并且仅在需要整个张量的操作中聚合完整的张量。

在本节中，我们使用 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 论文中的概念和图表：[在 GPU 集群上进行高效的大规模语言模型训练](https://arxiv.org/abs/2104.04473)。

任何 Transformer 的主要构建块都是一个完全连接的 `nn.Linear`，后面跟着非线性激活 `GeLU`。
按照 Megatron 论文的表示方法，我们可以将其点积部分写为 `Y = GeLU(XA)`，其中 `X` 和 `Y` 是输入和输出向量，`A` 是权重矩阵。

如果我们以矩阵形式查看计算，很容易看出矩阵乘法可以在多个 GPU 之间进行拆分：![并行 GEMM](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_gemm.png)
如果我们将权重矩阵 `A` 沿列在 `N` 个 GPU 上拆分，并并行执行矩阵乘法 `XA_1` 到 `XA_n`，那么我们将得到 `N` 个输出向量 `Y_1，Y_2，...，Y_n`，可以独立地输入到 `GeLU` 中：![独立的 GeLU](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-independent-gelu.png)
使用这个原理，我们可以更新任意深度的 MLP，而无需在 GPU 之间进行任何同步，直到最后需要从片段重建输出向量。Megatron-LM 论文的作者为此提供了一个有用的插图：![并行片段处理](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_shard_processing.png)
多头注意力层的并行化更加简单，因为它们本身就是并行的，具有多个独立的头部！![并行自注意力](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-tp-parallel_self_attention.png)
特殊注意事项：TP 需要非常快速的网络，因此不建议在超过一个节点上执行 TP。实际上，如果一个节点有 4 个 GPU，则最高的 TP 度数为 4。如果需要 8 个 TP 度数，则需要使用至少 8 个 GPU 的节点。

本节基于原始的更详细的 [TP 概述](https://github.com/huggingface/transformers/issues/10321#issuecomment-783543530)。由 [@anton-l](https://github.com/anton-l) 提供。

SageMaker 将 TP 与 DP 相结合，以实现更高效的处理。
替代名称：- DeepSpeed 将其称为 [tensor slicing](https://www.deepspeed.ai/features/#model-parallelism)
实现：
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 有一个内部实现，因为它非常特定于模型
- [parallelformers](https://github.com/tunib-ai/parallelformers)（目前仅支持推理）
- [SageMaker](https://arxiv.org/abs/2111.05972) - 这是一种专有解决方案，仅可在 AWS 上使用。
- [OSLO](https://github.com/tunib-ai/oslo) 基于 Transformers 的张量并行实现。

🤗 Transformers 状态：
- 核心：核心尚未实现

- 但如果您想要推理，[parallelformers](https://github.com/tunib-ai/parallelformers) 为我们的大多数模型提供此支持。因此，在核心实现之前，您可以使用它们的模型。希望训练模式也会得到支持。
- Deepspeed-Inference 还以其基于超快 CUDA 内核的推理模式支持我们的 BERT，GPT-2 和 GPT-Neo 模型，更多详情请参阅 [此处](https://www.deepspeed.ai/tutorials/inference-tutorial/)

## DP+PP

下图来自 DeepSpeed 的 [管道教程](https://www.deepspeed.ai/tutorials/pipeline/)，演示了如何将 DP 与 PP 结合使用。
![dp-pp-2d](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero-dp-pp.png)

在这里，重要的是要看到 DP 等级 0 看不到 GPU2，DP 等级 1 看不到 GPU3。对于 DP 来说，只有 GPU 0 和 1，它将数据作为只有 2 个 GPU 一样提供。GPU0 通过使用 PP 将其部分负载秘密地转移给 GPU2。GPU1 也通过请求 GPU3 的帮助来完成同样的操作。

由于每个维度至少需要 2 个 GPU，因此在这里您需要至少 4 个 GPU。

实现：
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [Varuna](https://github.com/microsoft/varuna)- [SageMaker](https://arxiv.org/abs/2111.05972)- [OSLO](https://github.com/tunib-ai/oslo)

🤗 Transformers 状态：尚未实现

## DP+PP+TP

为了实现更高效的训练，使用了三维并行，其中 PP 与 TP 和 DP 结合使用。下图展示了这种情况。
![dp-pp-tp-3d](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-deepspeed-3d.png)
此图来自博客文章 [3D parallelism: Scaling to trillion-parameter models](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)，这也是一篇很好的阅读材料。

由于每个维度至少需要 2 个 GPU，因此在这里您需要至少 8 个 GPU。

实现：- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - DeepSpeed 还包括更高效的 DP，称为 ZeRO-DP。- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)- [Varuna](https://github.com/microsoft/varuna)- [SageMaker](https://arxiv.org/abs/2111.05972)- [OSLO](https://github.com/tunib-ai/oslo)
🤗 Transformers 状态：尚未实现，因为我们没有 PP 和 TP。

## ZeRO DP+PP+TP

DeepSpeed 的主要特点之一是 ZeRO，它是 DP 的超可扩展扩展。它已经在 [ZeRO 数据并行](#zero-data-parallelism) 中讨论过。通常，它是一个独立的功能，不需要 PP 或 TP。但是它可以与 PP 和 TP 结合使用。

当 ZeRO-DP 与 PP（和可选的 TP）结合使用时，通常仅启用 ZeRO 阶段 1（优化器分片）。

虽然在 Pipeline Parallelism 中理论上可以使用 ZeRO 阶段 2（梯度分片），但它会对性能产生不良影响。每个微批次都需要有一个额外的 reduce-scatter 集合，以在分片之前聚合梯度，这会增加潜在的通信开销。由于 Pipeline Parallelism 使用小微批次，并且注重尽量平衡算术强度（微批次大小）和最小化 Pipeline 泡沫（微批次数），因此这些通信成本将会带来负面影响。
此外，由于 PP 已经减少了比普通情况下更少的层，因此内存节省并不是很大。PP 已经将梯度大小减少了 ``1/PP``，因此在此基础上进行梯度分片的节省效果不如纯 DP 显著。

ZeRO 阶段 3 也不是一个好选择，原因相同 - 需要更多的节点间通信。

由于我们拥有 ZeRO，另一个好处是 ZeRO-Offload。由于这是第 1 阶段的优化器状态，可以将其卸载到 CPU。

实现：
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed) 和 [BigScience 的 Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed)，这是前者的分支。
- [OSLO](https://github.com/tunib-ai/oslo)
重要论文：
- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/abs/2201.11990)

🤗 Transformers 状态：尚未实现，因为我们没有 PP 和 TP。

## FlexFlow

[FlexFlow](https://github.com/flexflow/FlexFlow) 还以稍微不同的方法解决了并行化问题。
论文：["Beyond Data and Model Parallelism for Deep Neural Networks" by Zhihao Jia, Matei Zaharia, Alex Aiken](https://arxiv.org/abs/1807.05358)
它在样本-运算符-属性-参数上执行了一种 4D 并行。

1. 样本 = 数据并行（按样本并行）
2. 运算符 = 将单个操作并行化为多个子操作 
3. 属性 = 数据并行（按长度并行）
4. 参数 = 模型并行（无论是水平还是垂直）

示例：
* 样本
假设有 10 个长度为 512 的批次。如果我们将它们按样本维度并行化为 2 个设备，那么我们将获得 10 x 512，变成 5 x 2 x 512。
* 运算符
如果我们执行层归一化，我们首先计算标准差，然后计算均值，然后可以对数据进行归一化。运算符并行化允许同时计算标准差和均值。因此，如果我们将它们按运算符维度并行化为 2 个设备（cuda: 0，cuda: 1），首先我们将输入数据复制到两个设备中，cuda: 0 同时计算标准差，cuda: 1 同时计算均值。
* 属性
我们有 10 个长度为 512 的批次。如果我们按属性维度将它们并行化到 2 个设备中，10 x 512 将变成 10 x 2 x 256。
* 参数
它与张量模型并行或天真的逐层模型并行类似。
![flex-flow-soap](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-flexflow.jpeg)
这个框架的重要性在于，它根据以下资源（1）GPU/TPU/CPU 与（2）RAM/DRAM 与（3）快速内部连接/慢速外部连接来自动优化算法，决定在哪里使用哪种并行化。

一个非常重要的方面是，FlexFlow 专为具有静态和固定工作负载的 DNN 并行化进行优化，因为具有动态行为的模型可能更喜欢在迭代过程中采用不同的并行化策略。
因此，这个框架的承诺非常吸引人 - 它在所选择的集群上运行 30 分钟的模拟，并提出最佳策略来利用这个特定环境。如果您添加/删除/替换任何部分，它将运行并重新优化该计划。然后您可以进行训练。不同的设置将有自己的定制优化。

🤗 Transformers 状态：尚未集成。我们已经通过 [transformers.utils.fx](https://github.com/huggingface/transformers/blob/master/src/transformers/utils/fx.py) 使我们的模型可以进行 FX 跟踪，这是 FlexFlow 的先决条件，因此有人需要弄清楚如何使 FlexFlow 与我们的模型一起工作。

## 使用哪种策略

以下是一个关于在什么情况下使用哪种并行化策略的粗略概述。每个列表中的第一项通常更快。

**⇨ 单个 GPU**

* 模型适合单个 GPU：
    1. 正常使用
* 模型不适合单个 GPU：
    1. ZeRO + 卸载 CPU 并可选地使用 NVMe    
    2. 如果最大的层无法适应单个 GPU，则使用 Memory Centric Tiling（有关详细信息，请参见下文）
* 最大层不适合单个 GPU：
1. ZeRO - 启用 [Memory Centric Tiling](https://deepspeed.readthedocs.io/en/latest/zero3.html#memory-centric-tiling)（MCT）。它允许您通过自动拆分和顺序执行来运行任意大的层。MCT 减少了在 GPU 上活动的参数数量，但不影响激活内存。由于此需求在本文撰写时非常罕见，用户需要手动覆盖 `torch.nn.Linear`。

**⇨ 单节点/多 GPU**

* 模型适合单个 GPU：
    1. DDP - 分布式 DP    
    2. ZeRO - 取决于情况和使用的配置，可能会更快
* 模型不适合单个 GPU：
    1. PP    
    2. ZeRO    
    3. TP
    在具有快速节点内连接（如 NVLINK 或 NVSwitch）的情况下，三种策略应该大致相当；如果没有这些，PP 将比 TP 或 ZeRO 更快。TP 的程度也可能有所不同。最好进行实验以找到特定设置的优胜者。

    TP 几乎总是在单个节点内使用。即 TP 大小 <= 每个节点的 GPU 数。

* 最大层不适合单个 GPU：
    1. 如果不使用 ZeRO - 必须使用 TP，因为仅使用 PP 无法适应。    
    2. 对于 ZeRO，请参阅上面的“单个 GPU”条目

**⇨ 多节点/多 GPU**

* 当您具有快速节点间连接时：
    1. ZeRO - 因为它对模型几乎没有修改要求    
    2. PP+TP+DP - 通信较少，但需要对模型进行大量更改
* 当您具有较慢的节点间连接并且 GPU 内存仍然较低时：
    1. DP+PP+TP+ZeRO-1