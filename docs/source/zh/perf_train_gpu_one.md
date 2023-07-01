<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证 2.0 版（“许可证”）进行许可；除非符合许可证，否则您不得使用此文件。您可以在许可证下获取许可证的副本。
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件按“按原样”基础分发，不附带任何形式的担保或条件，无论是明示的还是隐含的。请参阅许可证以获取
⚠️ 请注意，此文件是 Markdown 格式的，但包含我们文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确渲染。
-->

# 单 GPU 上的高效训练

本指南重点介绍如何在单个 GPU 上高效训练大型模型。如果您可以访问具有多个 GPU 的机器，这些方法仍然有效，但您还可以使用 [多 GPU 部分](perf_train_gpu_many) 中概述的其他方法。

在本节中，我们将介绍一些技巧，以减少内存占用并加速大型模型的训练，以及它们如何集成在 [`Trainer`] 和 [🤗 Accelerate](https://huggingface.co/docs/accelerate/) 中。每种方法都可以提高速度或内存使用情况，总结在下表中：
|Method|Speed|Memory|
|:-----|:----|:-----|
| Gradient accumulation | No | Yes |
| Gradient checkpointing | No| Yes |
| Mixed precision training | Yes | (No) |
| Batch size | Yes | Yes |
| Optimizer choice | Yes | Yes |
| DataLoader | Yes | No |
| DeepSpeed Zero | No | Yes |

方括号表示可能不严格是这样的，但通常要么不是主要问题，要么可以忽略不计。在开始之前，请确保已安装以下库：
```bash
pip install transformers datasets accelerate nvidia-ml-py3
```

`nvidia-ml-py3` 库允许我们从 Python 内部监视模型的内存使用情况。您可能熟悉终端中的 `nvidia-smi` 命令-此库允许直接在 Python 中访问相同的信息。

然后我们创建一些虚拟数据。我们创建介于 100 和 30000 之间的随机令牌 ID 和用于分类器的二进制标签。
总共，我们得到 512 个序列，每个序列长度为 512，并将它们存储在具有 PyTorch 格式的 [`~datasets.Dataset`] 中。


```py
import numpy as np
from datasets import Dataset


seq_len, dataset_size = 512, 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}
ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")
```

我们想要打印 GPU 利用率和训练运行的一些摘要统计信息，使用 [`Trainer`] 设置两个辅助函数来实现此目的：
```py
from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
```

让我们验证一下我们是否从空闲的 GPU 内存开始：
```py
>>> print_gpu_utilization()
GPU memory occupied: 0 MB.
```

看起来很好：在加载任何模型之前，GPU 内存没有被占用，这是我们预期的情况。如果在您的机器上不是这种情况，请确保停止使用 GPU 内存的所有进程。然而，并非所有的空闲 GPU 内存都可以被用户使用。当模型加载到 GPU 上时，内核也会被加载，这可能占用 1-2GB 的内存。为了查看具体占用了多少空间，我们将一个微小的张量加载到 GPU 中，这将触发内核的加载。
```py
>>> import torch


>>> torch.ones((1, 1)).to("cuda")
>>> print_gpu_utilization()
GPU memory occupied: 1343 MB.
```

我们看到内核独自占用了 1.3GB 的 GPU 内存。现在让我们看看模型使用了多少空间。

## 加载模型

首先，我们加载 `bert-large-uncased` 模型。我们直接将模型权重加载到 GPU 上，以便我们可以检查仅权重使用了多少空间。

```py
>>> from transformers import AutoModelForSequenceClassification


>>> model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
>>> print_gpu_utilization()
GPU memory occupied: 2631 MB.
```

我们可以看到模型权重独自占用了 1.3GB 的 GPU 内存。具体的数字取决于您使用的具体 GPU。请注意，在较新的 GPU 上，模型的占用空间有时可能更大，因为权重以优化的方式加载，从而加快了模型的使用。现在我们还可以快速检查是否与 `nvidia-smi` CLI 获得相同的结果：

```bash
nvidia-smi
```

```bash
Tue Jan 11 08:58:05 2022
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   37C    P0    39W / 300W |   2631MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      3721      C   ...nvs/codeparrot/bin/python     2629MiB |
+-----------------------------------------------------------------------------+
```

我们得到了与之前相同的数字，您还可以看到我们正在使用具有 16GB 内存的 V100 GPU。因此，现在我们可以开始训练模型并查看 GPU 内存消耗如何变化。首先，我们设置一些标准的训练参数，我们将在所有实验中使用这些参数：
```py
default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}
```

<Tip>
 注意：为了在实验之间正确清除内存，我们需要在实验之间重新启动 Python 内核。运行上面的所有步骤，然后只运行下面的一个实验。
</Tip>

## 原始训练

作为第一个实验，我们将使用 [`Trainer`] 训练模型，不进行任何其他修改，并使用批量大小为 4：
```py
from transformers import TrainingArguments, Trainer, logging

logging.set_verbosity_error()


training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 57.82
Samples/second: 8.86
GPU memory occupied: 14949 MB.
```

我们可以看到即使是相对较小的批量大小也几乎填满了 GPU 的全部内存。然而，较大的批量大小通常可以实现更快的模型收敛或更好的最终性能。因此，理想情况下，我们希望根据模型的需求而不是 GPU 限制来调整批量大小。有趣的是，我们使用的内存远远超过了模型的大小。为了更好地理解为什么会出现这种情况，让我们来看看模型的操作和内存需求。

## 模型操作解析

Transformers 架构包括以下 3 个主要的计算强度分组的操作。

1. **张量收缩**
    线性层和多头注意力的组成部分都进行批处理的 **矩阵-矩阵乘法**。这些操作是训练 transformer 的计算强度最高的部分。
2. **统计归一化**
    Softmax 和层归一化的计算强度比张量收缩要低，并涉及一个或多个 **规约操作**，其结果通过映射应用。
3. **逐元素运算**
    这些是剩余的操作：**偏置，dropout，激活和残差连接**。这些是计算强度最低的操作。

了解这些信息对于分析性能瓶颈很有帮助。
此摘要来自 [Data Movement Is All You Need: A Case Study on Optimizing Transformers 2020](https://arxiv.org/abs/2007.00072)


## 模型内存解析

我们已经看到训练模型使用的内存远远超过将模型放在 GPU 上的内存。这是因为在训练过程中有许多组件使用了 GPU 内存。GPU 内存上的组件包括：

1. 模型权重 
2. 优化器状态 
3. 梯度 
4. 用于梯度计算的正向激活
5. 临时缓冲区 
6. 特定功能的内存

使用 AdamW 进行混合精度训练的典型模型每个模型参数需要 18 个字节的内存，以及激活内存。对于推理，没有优化器状态和梯度，因此我们可以减去这些。因此，对于混合精度推理，每个模型参数需要 6 个字节的内存，以及激活内存。

让我们来看看具体细节。

**模型权重：**
- 4 个字节 *fp32 训练的参数数量- 6 个字节* 混合精度训练的参数数量（在内存中维护一个 fp32 模型和一个 fp16 模型）

**优化器状态：**
- 8 个字节 *正常 AdamW 的参数数量（维护 2 个状态）- 2 个字节* 8 位 AdamW 优化器（例如 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)）的参数数量- 4 个字节*带动量的 SGD 优化器（仅维护 1 个状态）的参数数量

**梯度**
- 每个 fp32 或混合精度训练的参数需要 4 个字节的内存（梯度始终以 fp32 保留）

**正向激活**
- 大小取决于许多因素，其中关键因素是序列长度，隐藏大小和批量大小。
通过前向和反向函数传入和返回的输入和输出以及为梯度计算保存的前向激活。

**临时内存**
此外，还有各种临时变量，在计算完成后会被释放，但在某些情况下可能需要额外的内存并可能导致 OOM。因此，在编码时，战略地考虑这些临时变量并有时在不再需要时明确释放它们是至关重要的。

**功能特定的内存**
然后，您的软件可能具有特殊的内存需求。例如，使用波束搜索生成文本时，软件需要维护多个输入和输出的副本。

**`forward` 与 `backward` 的执行速度**
对于卷积和线性层，反向传播中的计算量是前向传播的 2 倍，这通常导致速度慢大约 2 倍（有时更多，因为反向传播中的尺寸通常更尴尬）。激活通常受带宽限制，一个激活在反向传播中需要读取的数据通常比在前向传播中要多（例如，激活前向传递一次读取一次，写入一次；激活反向传递读取两次，gradOutput 和前向输出一次，写入一次，gradInput）。

因此，有几个地方可以节省 GPU 内存或加速操作。让我们从一个简单的优化开始：选择正确的批量大小。

## 批量大小
当批量大小和输入/输出神经元数量可以被某个数字整除时，可以获得最高效的性能，这个数字通常从 8 开始，但也可能更高。这个数字的大小取决于所使用的具体硬件和模型的 dtype。

例如，对于全连接层（对应于 GEMM），NVIDIA 提供了有关 [input/output 神经元数量的建议](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#input-features) 和 [批量大小](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#batch-size)。

[Tensor Core 要求](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc) 根据 dtype 和硬件定义了乘数。例如，对于 fp16，建议使用 8 的倍数，但对于 A100，则是 64！

对于小型参数，还需要考虑 [维度量化效应](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization)，这是平铺发生的地方，合适的乘数可以显著加速。

## 梯度累积

梯度累积的思想是，不是一次计算整个批量的梯度，而是分成较小的步骤进行。我们通过在模型中进行前向和后向传递并在此过程中累积梯度来计算梯度。当累积了足够的梯度时，我们运行模型的优化步骤。这样，我们可以轻松地将总批量大小增加到无法适应 GPU 内存的数字。然而，额外的前向和后向传递可能会稍微减慢训练速度。

我们可以通过在 [`TrainingArguments`] 中简单地添加 `gradient_accumulation_steps` 参数来在 [`Trainer`] 中使用梯度累积。
让我们看看它如何影响模型的内存占用：
```py
training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 66.03
Samples/second: 7.75
GPU memory occupied: 8681 MB.
```

我们可以看到，内存占用大大减少，而速度与普通运行相比略有下降。当然，随着累积步数的增加，情况会发生变化。通常，您希望尽可能充分利用 GPU 的使用率。

因此，在我们的例子中，批量大小为 4 已经非常接近 GPU 的限制。如果我们想使用批量大小为 64 进行训练，我们不应该使用 `per_device_train_batch_size=1` 和 `gradient_accumulation_steps=64`，而应该使用 `per_device_train_batch_size=4` 和 `gradient_accumulation_steps=16`，这样具有相同的有效批量大小，同时更好地利用可用的 GPU 资源。

有关 RTX-3090 的详细信息，请参阅基准测试结果：[RTX-3090](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004392537) 和 [A100](https://github.com/huggingface/transformers/issues/15026#issuecomment-1005033957)。
接下来，我们将介绍另一个小幅节省 GPU 内存的技巧，称为梯度检查点。

## 梯度检查点 Gradient Checkpointing

即使将批量大小设置为 1 并使用梯度累积，当处理大型模型时，仍然可能会用完内存。为了在反向传播期间计算梯度，通常会保存来自前向传播的所有激活值。这可能会导致内存开销很大。另一种方法是在前向传播期间忘记所有激活值，并在反向传播期间按需重新计算它们。然而，这将增加显著的计算开销并减慢训练速度。

梯度检查点在这两种方法之间取得了折衷，并在计算图中保存了策略性选择的激活值，因此只需重新计算部分激活值即可计算梯度。请参阅 [这篇很棒的文章](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)，详细解释了梯度检查点背后的思想。

要在 [`Trainer`] 中启用梯度检查点，我们只需要将其作为标志传递给 [`TrainingArguments`] 即可。其他所有工作都在幕后处理：

```py
training_args = TrainingArguments(
    per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True, **default_args
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 85.47
Samples/second: 5.99
GPU memory occupied: 6775 MB.
```

我们可以看到，这节省了更多的内存，但同时训练速度变慢了一些。一个经验法则是，梯度检查点会使训练速度减慢约 20%。
让我们看看另一种方法，可以提高一些速度：混合精度训练。

## 浮点数据类型 Floating Data Types

混合精度训练的思想是，并非所有变量都需要以完整的（32 位）浮点精度存储。如果我们可以降低精度，变量和计算速度就会更快。这是常用的浮点数据类型，选择哪种类型会影响内存使用和吞吐量：

- fp32（`float32`）
- fp16（`float16`）
- bf16（`bfloat16`）
- tf32（CUDA 内部数据类型）

下面的图表显示了这些数据类型彼此之间的关系。

![数据类型](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tf32-bf16-fp16-fp32.png)（来源：[NVIDIA Blog](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)）

虽然 fp16 和 fp32 已经存在一段时间了，但 bf16 和 tf32 仅适用于 Ampere 架构的 GPU，TPU 也支持 bf16。让我们从最常用的方法开始，即 FP16 训练/

### FP16 训练

混合精度训练的思想是并非所有变量都需要以完整（32 位）浮点精度存储。如果我们可以降低精度，则变量及其计算速度更快。主要优势来自以半（16 位）精度保存激活值。


虽然梯度也是以半精度计算的，但在优化步骤中会转换回完整精度，因此在这里不会节省内存。由于模型同时以 16 位和 32 位精度存在于 GPU 上，这可能会占用更多的 GPU 内存（GPU 上的原始模型的 1.5 倍），特别是对于小批量大小。由于部分计算使用全精度，部分使用半精度，因此此方法也被称为混合精度训练。

启用混合精度训练只需要将 `fp16` 标志设置为 `True`:

```py
training_args = TrainingArguments(per_device_train_batch_size=4, fp16=True, **default_args)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 27.46
Samples/second: 18.64
GPU memory occupied: 13939 MB.
```

我们可以看到，这几乎比常规训练快两倍。让我们将其添加到之前方法的组合中:

```py
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    **default_args,
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 50.76
Samples/second: 10.09
GPU memory occupied: 7275 MB.
```

我们可以看到，在这些调整后，与开始时相比，我们使用的 GPU 内存减少了约一半，同时速度稍微更快。

### BF16 

如果您可以访问 Ampere 或更新的硬件，可以在训练和评估中使用 bf16。虽然 bf16 的精度比 fp16 差，但其动态范围要大得多。因此，如果在过去的训练中经常遇到溢出问题，bf16 将在大多数情况下防止这种情况发生。请记住，在 fp16 中，您可以拥有的最大数字是 `65535`，超过该数字将导致溢出。bf16 数字最大可达到 `3.39e+38`（！），与 fp32 差不多 - 因为两者都使用了用于数值范围的 8 位。

您可以通过以下方式在 🤗 Trainer 中启用 BF16:

```python
TrainingArguments(bf16=True)
```

### TF32Ampere 

硬件使用了一种名为 tf32 的神奇数据类型。

它具有与 fp32 相同的数值范围（8 位），但是它只有 10 位精度（与 fp16 相同），总共只使用了 19 位。

它是神奇的，是因为您可以使用普通的 fp32 训练和/或推断代码，并通过启用 tf32 支持，您可以获得高达 3 倍的吞吐量改进。您只需要在代码中添加以下内容即可:
```
import torch
torch.backends.cuda.matmul.allow_tf32 = True
```

完成此操作后，CUDA 将自动切换到在可能的情况下使用 tf32 而不是 fp32。这当然假设所使用的 GPU 是 Ampere 系列的。

与所有降低精度的情况一样，这可能令您满意或不满意，因此您需要进行实验和观察。根据 [NVIDIA research](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/)，大多数机器学习训练不应受到影响，并且显示出与 fp32 训练相同的困惑度和收敛性。

如果您已经使用 fp16 或 bf16 混合精度，它也可能有助于吞吐量。

您可以通过以下方式在 🤗 Trainer 中启用此模式: 
```python
TrainingArguments(tf32=True)
```

默认情况下，使用的是 PyTorch 默认设置。

注意：tf32 模式是 CUDA 的内部功能，无法通过 `tensor.to(dtype=torch.tf32)` 直接访问，因为 `torch.tf32` 不存在。

注意：您需要 `torch>=1.7` 才能享受此功能。
您还可以查看关于 tf32 与其他精度的各种基准测试: [RTX-3090](https://github.com/huggingface/transformers/issues/14608#issuecomment-1004390803) 和 [A100](https://github.com/huggingface/transformers/issues/15026#issuecomment-1004543189)。

我们已经了解了如何更改浮点类型以提高吞吐量，但我们还没有完成！还有另一个领域可以节省 GPU 内存：优化器。

## 优化器

训练 transformer 模型最常用的优化器是 Adam 或 AdamW（带权重衰减的 Adam）。Adam 通过存储先前梯度的滚动平均值实现良好的收敛性，但这会增加大约模型参数数量的内存占用。解决此问题的方法之一是使用替代优化器，例如 Adafactor，它对某些模型效果很好，但通常存在不稳定性问题。

HF Trainer 集成了多种优化器，可以直接使用。要激活所需的优化器，只需将 `--optim` 标志传递给命令行。
要查看当前支持的优化器:

```bash
$ python examples/pytorch/translation/run_translation.py -h | grep "\-optim"
         [--optim {adamw_hf,adamw_torch,adamw_torch_xla,adamw_apex_fused,adafactor}]
```

例如，如果已安装了 [NVIDIA/apex](https://github.com/NVIDIA/apex)，`--optim adamw_apex_fused` 将为您提供所有支持的 AdamW 优化器中最快的训练体验。

另一方面，如果 8bit BNB 优化器被配置为量化所有优化器状态，它可以节省典型 AdamW 优化器使用的内存的 3/4，但在某些情况下只有某些优化器状态被量化，然后会使用更多内存。

让我们来了解一下这些数字，并以 3B 参数模型（例如 `t5-3b`）为例。请注意，由于千兆字节对应十亿字节，我们只需将参数（以十亿计）乘以每个参数所需的字节数，即可得到 GPU 内存使用量的千兆字节:

- 标准 AdamW 每个参数使用 8 字节，这里的优化器将需要（`8*3`）24GB 的 GPU 内存。
- Adafactor 使用略多于 4 字节，因此（`4*3`）12GB，然后还有一些额外的内存。
- 如果所有优化器状态都被量化，8bit BNB 量化优化器只会使用（`2*3`）6GB。

让我们先看看 Adafactor 的情况。

### Adafactor

Adafactor 不像 Adam 那样为权重矩阵中的每个元素保留滚动平均值，而是仅存储聚合信息（行和列的滚动平均值的总和），从而大大减少了内存占用。

 Adafactor 的一个缺点是，在某些情况下，收敛速度可能比 Adam 慢，因此建议在此处进行一些实验。我们可以通过设置 `optim="adafactor"` 来使用 Adafactor:

```py
training_args = TrainingArguments(per_device_train_batch_size=4, optim="adafactor", **default_args)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 64.31
Samples/second: 7.96
GPU memory occupied: 12295 MB.
```

我们可以看到，这进一步节省了几个 GB 的 GPU 内存。让我们看看将其添加到我们之前介绍的其他方法中的效果:

```py
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    optim="adafactor",
    **default_args,
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)
```

```
Time: 56.54
Samples/second: 9.06
GPU memory occupied: 4847 MB.
```

我们从使用 15 GB 内存降低到 5 GB - 同时保持吞吐量的 3 倍提升！但是，如前所述，Adafactor 的收敛性可能比 Adam 差。有一种名为 8-bit Adam 的替代方法。
### 8-bit Adam

8-bit Adam 不像 Adafactor 一样聚合优化器状态，而是保留完整状态并对其进行量化。量化意味着以较低精度存储状态，并仅在优化时进行去量化。这类似于 FP16 训练的思想，其中使用较低精度的变量可以节省内存。

与之前的方法不同，这个方法不是作为简单的标志集成到 [`Trainer`] 中。我们需要安装 8-bit 优化器，然后将其作为自定义优化器传递给 [`Trainer`]。

按照 Github [repo](https://github.com/TimDettmers/bitsandbytes) 中的安装指南安装实现 8-bit Adam 优化器的 `bitsandbytes` 库。

安装后，我们只需要初始化优化器。虽然看起来需要进行大量工作，但实际上只需两个步骤：首先，我们需要将模型的参数分为两组，对一组应用权重衰减，对另一组不应用。通常，不对偏置和层归一化参数应用权重衰减。然后，在第二步中，我们只需进行一些参数处理，以使用之前使用的 AdamW 优化器相同的参数。

<Tip>

 请注意，为了使用 8-bit 优化器与现有的预训练模型，需要对嵌入层进行更改。阅读更多信息，请查看 [此问题](https://github.com/huggingface/transformers/issues/14819)。
 </Tip>
```py
import bitsandbytes as bnb
from torch import nn
from transformers.trainer_pt_utils import get_parameter_names

training_args = TrainingArguments(per_device_train_batch_size=4, **default_args)

decay_parameters = get_parameter_names(model, [nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters],
        "weight_decay": training_args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
        "weight_decay": 0.0,
    },
]

optimizer_kwargs = {
    "betas": (training_args.adam_beta1, training_args.adam_beta2),
    "eps": training_args.adam_epsilon,
}
optimizer_kwargs["lr"] = training_args.learning_rate
adam_bnb_optim = bnb.optim.Adam8bit(
    optimizer_grouped_parameters,
    betas=(training_args.adam_beta1, training_args.adam_beta2),
    eps=training_args.adam_epsilon,
    lr=training_args.learning_rate,
)
```

我们现在可以将自定义优化器作为参数传递给 `Trainer`：

```py
trainer = Trainer(model=model, args=training_args, train_dataset=ds, optimizers=(adam_bnb_optim, None))
result = trainer.train()
print_summary(result)
```

```
Time: 55.95
Samples/second: 9.15
GPU memory occupied: 13085 MB.
```

我们可以看到，与 Adafactor 相比，我们获得了类似的内存改进。让我们使用完整的设置重复实验：
```py
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    **default_args,
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds, optimizers=(adam_bnb_optim, None))
result = trainer.train()
print_summary(result)
```

```
Time: 49.46
Samples/second: 10.35
GPU memory occupied: 5363 MB.
```

同样，我们获得了大约 3 倍的内存改进，甚至比使用 Adafactor 时吞吐量稍高。因此，我们已经看到了如何优化大型模型的内存占用。下图总结了我们所有的实验结果：
![png](https://huggingface.co/datasets/lvwerra/repo-images/raw/main/gpu-memory-savings.png)
### `_multi_tensor` 

pytorch-nightly 引入了 `torch.optim._multi_tensor`，它应该显著加速具有大量小特征张量的优化器。它最终应该成为默认选项，但如果您想尽早尝试并且不介意使用最新版本，请参阅：https://github.com/huggingface/transformers/issues/9965

## 使用 🤗 Accelerate

到目前为止，我们使用了 [`Trainer`] 来运行实验，但与该方法相比，更灵活的替代方案是使用🤗 Accelerate。使用🤗 Accelerate，您可以完全控制训练循环，并可以基本上以纯 PyTorch 编写循环，只需进行一些微小的修改。反过来，它允许您轻松跨不同的基础架构进行扩展，例如 CPU、GPU、TPU 或分布式多 GPU 设置，而无需更改任何代码。让我们看看如何在🤗 Accelerate 中实现所有上述调整。我们仍然可以使用 [`TrainingArguments`] 来包装训练设置：

```py
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    **default_args,
)
```

使用🤗 Accelerate 的完整示例训练循环只有几行代码长：

```py
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader

dataloader = DataLoader(ds, batch_size=training_args.per_device_train_batch_size)

if training_args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

accelerator = Accelerator(fp16=training_args.fp16)
model, optimizer, dataloader = accelerator.prepare(model, adam_bnb_optim, dataloader)

model.train()
for step, batch in enumerate(dataloader, start=1):
    loss = model(**batch).loss
    loss = loss / training_args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % training_args.gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

首先，我们将数据集包装在 [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 中。

然后，我们可以通过调用模型的 [`~PreTrainedModel.gradient_checkpointing_enable`] 方法来启用梯度检查点。当我们初始化 [`Accelerator`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator) 时，我们可以指定是否要使用混合精度训练，并且它会在 [`prepare`] 调用中为我们处理。在 [`prepare`](https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.prepare) 调用期间，如果我们使用多个 GPU，数据加载器也将被分布到工作器上。我们使用与先前实验中相同的 8 位优化器。
最后，我们可以编写主要的训练循环。

请注意，`backward` 调用由🤗 Accelerate 处理。我们还可以看到梯度累积的工作原理：我们对损失进行归一化，以便在累积结束时得到平均值，并且一旦步骤足够多，我们运行优化。现在的问题是：这是否使用与前面步骤相同的内存量？让我们来检查一下：

```py
>>> print_gpu_utilization()
GPU memory occupied: 5363 MB.
```

确实如此。
只需几行代码就可以使用🤗 Accelerate 实现这些优化技术，并且具有训练循环中更大的灵活性的好处。有关所有功能的完整文档，请参阅 [Accelerate 文档](https://huggingface.co/docs/accelerate/index)。
## DataLoader

达到出色的训练速度的一个重要要求是能够以 GPU 能够处理的最大速度提供数据。默认情况下，所有操作都发生在主进程中，可能无法快速地从磁盘读取数据，从而产生瓶颈，导致 GPU 利用率不高。

- `DataLoader(pin_memory=True, ...)` 可确保数据预加载到 CPU 上的固定内存中，并且通常导致从 CPU 到 GPU 内存的传输速度更快。
- `DataLoader(num_workers=4, ...)` 
- 启动多个工作器以更快地预加载数据 - 在训练过程中观察 GPU 利用率统计数据，如果远离 100%，尝试增加工作器的数量。当然，问题可能出在其他地方，因此非常多的工作器不一定会导致更好的性能。

## DeepSpeed ZeRO

有关如何使用 Deepspeed 的详细信息，请参阅 [此处](main_classes/deepspeed)。

首先，快速决策树：
1. 模型适合单个 GPU 并且您有足够的空间来容纳小批量大小 - 在这种情况下，您不需要使用 Deepspeed，因为它只会减慢速度。
2. 模型无法适应单个 GPU 或者您无法容纳小批量 - 使用 DeepSpeed ZeRO + CPU Offload，对于更大的模型使用 NVMe Offload。

如果决策树建议您首先使用 DeepSpeed，那么您首先需要 [安装它](main_classes/deepspeed#installation)，然后按照以下指南之一创建配置文件并启动 DeepSpeed。

激活：

- 基于 HF Trainer 的示例：参见此 [指南](main_classes/deepspeed#deployment-with-one-gpu)。- 基于自定义 HF Trainer 程序：与上述相同，但传递：
    ```python
    TrainingArguments(deepspeed="/path/to/ds_config.json")
    ```
- 在笔记本中部署：请参阅 [此指南](main_classes/deepspeed#deployment-in-notebooks)。

- 自定义训练循环：这有些复杂，但您可以研究如何在 [HF Trainer](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py) 中实现 - 只需在代码中搜索 `deepspeed`。

## GPU 的选择

有时，即使应用了上述的所有优化技巧，某个 GPU 的吞吐量可能仍然不够好。一个简单的解决方案是更换 GPU 的类型。例如，从 Google Colab 上通常使用的 K80 切换到更高级的 GPU，如 V100 或 A100。尽管它们的价格更高，但由于其更大的内存和更快的架构，通常比较便宜的 GPU 更具成本效益。

现在，让我们退后一步，讨论在扩展大型模型的训练时应该优化的内容。

## 如何进行扩展

当我们训练模型时，有两个方面需要同时优化：

- 数据吞吐量/训练时间
- 模型性能

我们已经看到每种方法都会改变内存使用和吞吐量。通常我们希望最大化吞吐量（样本/秒），以最小化训练成本。这通常通过尽可能充分利用 GPU 并将 GPU 内存填满来实现。例如，如前所述，我们仅在希望使用大于 GPU 内存大小的批次大小时才使用梯度累积。如果所需的批次大小适合内存，则没有理由应用梯度累积，因为这只会减慢训练速度。

第二个目标是模型性能。仅仅因为我们可以使用大批次大小并不意味着我们应该这样做。作为超参数调整的一部分，您应该确定哪个批次大小产生了最佳结果，然后相应地优化吞吐量。

## 高效的软件预构建

PyTorch 的 [pip 和 conda 构建](https://pytorch.org/get-started/locally/#start-locally) 预先构建了 cuda toolkit，足以运行 PyTorch，但如果需要构建 cuda 扩展，这是不够的。

有时候，可能需要额外的工作来预构建一些组件，例如，如果您使用的是不预编译的库（如 `apex`）。在其他情况下，如果您需要全系统范围内安装正确的 cuda toolkit 可能会很复杂。为了满足用户的需求，PyTorch 和 NVIDIA 发布了一个新版本的 NGC docker 容器，其中已经预先构建了所有内容，您只需将程序安装到其中，就可以立即运行。

如果您想调整 PyTorch 源代码和/或进行新的定制构建，这种方法也非常有用。

要找到您想要的 docker 映像版本，请从这里开始，选择最新的月度发布之一。进入所需版本的发布说明，检查环境的组件是否符合您的需求（包括 NVIDIA 驱动程序要求！），然后在该文档的顶部转到相应的 NGC 页面。如果不慎迷失方向，这是所有 PyTorch NGC 映像的索引。

接下来，请按照下载和部署 docker 映像的说明进行操作。
## 稀疏性

### 专家混合模型

最近的一些论文报告称，将专家混合模型（Mixture of Experts，MoE）集成到 Transformer 模型中可以加快训练速度 4-5 倍，并提高推断速度。

由于已经发现更多的参数可以带来更好的性能，这种技术可以使参数数量增加一个数量级，而不增加训练成本。

在这种方法中，每个 FFN（Feed-Forward Network）层都被 MoE 层替换，MoE 层由许多专家组成，通过门控函数以平衡的方式训练每个专家，具体取决于输入标记在序列中的位置。

![MoE Transformer 2x block](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/perf-moe-transformer.png)

(来源：[GLAM](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html))

您可以在本节末尾列出的论文中找到详尽的细节和比较表格。

这种方法的主要缺点是它需要大量的 GPU 内存，几乎比密集模型的内存要大一个数量级。为了解决更高的内存需求，提出了各种蒸馏和方法。

然而，存在一种直接的权衡，您可以使用少量的专家和 2-3 倍较小的基本模型，而不是数十个或数百个专家，这样可以得到一个 5 倍较小的模型，从而在适度增加内存需求的同时适度提高训练速度。

大多数相关的论文和实现都是基于 TensorFlow/TPU 的：

- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- [GLaM: Generalist Language Model (GLaM)](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html)

对于 PyTorch，DeepSpeed 也构建了一个专家混合模型：[DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596)，[Mixture of Experts](https://www.deepspeed.ai/tutorials/mixture-of-experts/) - 博文：[1](https://www.microsoft.com/en-us/research/blog/deepspeed-powers-8x-larger-moe-model-training-with-high-performance/)，[2](https://www.microsoft.com/en-us/research/publication/scalable-and-efficient-moe-training-for-multitask-multilingual-models/)，以及用于大型基于 Transformer 的自然语言生成模型的具体部署：[博文](https://www.deepspeed.ai/news/2021/12/09/deepspeed-m

oe-nlg.html)，[Megatron-Deepspeed 分支](Thttps://github.com/microsoft/Megatron-DeepSpeed/tree/moe-training)。

## 超越单个 GPU 的扩展

对于一些应用，例如预训练大型语言模型，应用上述所有方法可能仍然不够快。在这种情况下，您可以将实验扩展到多个 GPU 上。

在需要在多个 GPU 上进行训练的另一个用例是，如果模型无法在单个 GPU 上使用所有提到的技巧进行训练。尽管此时的方法更加复杂，但通常涉及某种形式的流水线或张量并行，其中模型本身在多个 GPU 上进行分布。您还可以利用 DeepSpeed，它实现了一些并行策略以及一些更多的优化，以减少内存占用，例如对优化器状态进行分区。您可以在 ["多 GPU 训练" 部分](perf_train_gpu_many) 中阅读更多相关信息。

## 使用 PyTorch 原生注意力机制

PyTorch 2.0 发布了原生的 [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html) (SDPA)，它可以使用融合的 GPU 内核进行 [内存高效的注意力计算](https://arxiv.org/abs/2112.05682) 和 [快闪注意力](https://arxiv.org/abs/2205.14135)。

在安装了 [`optimum`](https://github.com/huggingface/optimum) 包之后，可以使用以下代码将相关内部模块替换为使用 PyTorch 的原生注意力机制：

```python
model = model.to_bettertransformer()
```

然后可以像往常一样进行训练。

## 使用 torch.compile

PyTorch 2.0 引入了一个新的编译函数，您可以在他们的 [文档](https://pytorch.org/get-started/pytorch-2.0/) 中了解更多信息。它使用 Python 的帧评估 API 从现有的 PyTorch 程序自动创建图形。在捕获图形之后，可以部署不同的后端来将图形降低到优化的引擎。您可以从下面的选项中选择一个来提高性能。

`torch.compile` 拥有越来越多的后端，可以在 [backends.py](https://github.com/pytorch/pytorch/blob/master/torch/_dynamo/optimizations/backends.py) 中找到，或者使用 `torchdynamo.list_backends()`，每个后端都有其可选的依赖项。

其中一些最常用的后端是：

**调试后端**：
* `dynamo.optimize("eager")` - 使用 PyTorch 运行提取的 GraphModule。这在调试 TorchDynamo 问题时非常有用。
* `dynamo.optimize("aot_eager")` - 仅使用 PyTorch 的 eager 模式对 AotAutograd 提取的前向和反向图进行运行。这对调试很有用，但不太可能提供

速度优势。

**训练和推断后端**：
* `dynamo.optimize("inductor")` - 使用 TorchInductor 后端，结合 AotAutograd 和 cudagraphs，利用代码生成的 Triton 内核。[了解更多](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
* `dynamo.optimize("nvfuser")` - 使用 TorchScript 的 nvFuser。[了解更多](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `dynamo.optimize("aot_nvfuser")` - 使用 AotAutograd 的 nvFuser。[了解更多](https://dev-discuss.pytorch.org/t/tracing-with-primitives-update-1-nvfuser-and-its-primitives/593)
* `dynamo.optimize("aot_cudagraphs")` - 使用 AotAutograd 的 cudagraphs。[了解更多](https://github.com/pytorch/torchdynamo/pull/757)

**仅推断后端**：
* `dynamo.optimize("ofi")` - 使用 Torchscript 的 optimize_for_inference。[了解更多](https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html)
* `dynamo.optimize("fx2trt")` - 使用 Nvidia TensorRT 进行推断优化。[了解更多](https://github.com/pytorch/TensorRT/blob/master/docsrc/tutorials/getting_started_with_fx_path.rst)
* `dynamo.optimize("onnxrt")` - 使用 ONNXRT 进行 CPU/GPU 上的推断。[了解更多](https://onnxruntime.ai/)
* `dynamo.optimize("ipex")` - 使用 IPEX 在 CPU 上进行推断。[了解更多](https://github.com/intel/intel-extension-for-pytorch)
