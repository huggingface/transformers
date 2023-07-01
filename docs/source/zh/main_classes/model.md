<!--版权所有 2020 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非符合许可证的规定，否则您不得使用本文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按原样分发的，不附带任何明示或暗示的担保或条件。请参阅许可证中的特定语言的权限和限制。⚠️请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->

# 模型 Models

基类 [`PreTrainedModel`]，[`TFPreTrainedModel`] 和 [`FlaxPreTrainedModel`] 实现了从本地文件或目录加载/保存模型的常用方法，或从库提供的预训练模型配置（从 HuggingFace 的 AWS S3 存储库下载）。

[`PreTrainedModel`] 和 [`TFPreTrainedModel`] 还实现了一些所有模型中常见的方法，包括：

- 当新的标记添加到词汇表中时，调整输入标记嵌入的大小- 对模型的注意头进行修剪。

其他适用于每个模型的方法在 [`~modeling_utils.ModuleUtilsMixin`]（用于 PyTorch 模型）和 [`~modeling_tf_utils.TFModuleUtilsMixin`]（用于 TensorFlow 模型）中定义，或者对于文本生成，在 PyTorch 模型中使用 [`~generation.GenerationMixin`]，在 TensorFlow 模型中使用 [`~generation.TFGenerationMixin`]，在 Flax/JAX 模型中使用
[`~generation.FlaxGenerationMixin`] 定义。

## PreTrainedModel

[[autodoc]] PreTrainedModel
    - push_to_hub
    - all

<a id='from_pretrained-torch-dtype'> </a>




### Large model loading

在 Transformers 4.20.0 版本中，[`~PreTrainedModel.from_pretrained`] 方法已经进行了重构，以适应使用 [Accelerate](https://huggingface.co/docs/accelerate/big_modeling) 进行大型模型训练。这需要使用 Accelerate >= 0.9.0 和 PyTorch >= 1.9.0。不再是创建完整的模型，然后在其中加载预训练权重（这会在内存中占用模型大小的两倍，一个用于随机初始化的模型，一个用于权重），而是可以选择将模型创建为空壳，然后在加载预训练权重时才实例化其参数。

可以通过设置 `low_cpu_mem_usage=True` 来激活此选项。模型首先在元设备上创建（权重为空），然后将状态字典加载到其中（对于分片检查点，逐个分片加载）。这样，最大使用的内存仅为模型的完整大小。

```py
from transformers import AutoModelForSeq2SeqLM

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", low_cpu_mem_usage=True)
```

此外，如果模型无法完全放入内存中，您可以直接将模型放置在不同的设备上（目前仅适用于推理）。通过设置 `device_map="auto"`，Accelerate 会确定将每个层放置在哪个设备上，以最大限度地利用最快的设备（GPU），并将剩余部分卸载到 CPU 甚至硬盘上（如果您的 GPU 内存或 CPU 内存不足）。即使模型分布在多个设备上，它也将按照您通常的预期运行。

当传递 `device_map` 时，`low_cpu_mem_usage` 会自动设置为 `True`，因此您不需要指定它：

```py
from transformers import AutoModelForSeq2SeqLM

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto")
```

您可以通过查看模型的 `hf_device_map` 属性来检查模型在设备上的划分方式：

```py
t0pp.hf_device_map
```

```python out
{'shared': 0,
 'decoder.embed_tokens': 0,
 'encoder': 0,
 'decoder.block.0': 0,
 'decoder.block.1': 1,
 'decoder.block.2': 1,
 'decoder.block.3': 1,
 'decoder.block.4': 1,
 'decoder.block.5': 1,
 'decoder.block.6': 1,
 'decoder.block.7': 1,
 'decoder.block.8': 1,
 'decoder.block.9': 1,
 'decoder.block.10': 1,
 'decoder.block.11': 1,
 'decoder.block.12': 1,
 'decoder.block.13': 1,
 'decoder.block.14': 1,
 'decoder.block.15': 1,
 'decoder.block.16': 1,
 'decoder.block.17': 1,
 'decoder.block.18': 1,
 'decoder.block.19': 1,
 'decoder.block.20': 1,
 'decoder.block.21': 1,
 'decoder.block.22': 'cpu',
 'decoder.block.23': 'cpu',
 'decoder.final_layer_norm': 'cpu',
 'decoder.dropout': 'cpu',
 'lm_head': 'cpu'}
```

您还可以按照相同的格式（将层名称映射到设备）编写自己的设备映射。它应该将模型的所有参数映射到给定的设备，但如果某个层的所有子模块完全在同一设备上，则无需详细说明它们的具体位置。例如，以下设备映射对于 T0pp 将正常工作（只要您具有足够的 GPU 内存）：
```python
device_map = {"shared": 0, "encoder": 0, "decoder": 1, "lm_head": 1}
```
为了最小化模型对内存的影响，另一种方法是将其实例化为较低精度的 dtype（例如 `torch.float16`），或使用下面描述的直接量化技术。

### 模型实例化的 dtype

在 PyTorch 中，模型通常使用 `torch.float32` 格式进行实例化。如果尝试加载权重为 fp16 的模型，则可能会遇到问题，因为这将需要两倍的内存。为了克服这个限制，您可以通过使用 `torch_dtype` 参数显式传递所需的 `dtype` 来实现：

```python
model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype=torch.float16)
```

如果您希望模型始终以最优的内存模式加载，可以使用特殊值 `"auto"`，然后 `dtype` 将根据模型的权重自动推导出来：

```python
model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype="auto")
```

从头开始实例化的模型也可以通过以下方式告知使用哪种 `dtype`：

```python
config = T5Config.from_pretrained("t5")
model = AutoModel.from_config(config)
```

由于 PyTorch 的设计，此功能仅适用于浮点 dtype。


## ModuleUtilsMixin

[[autodoc]] modeling_utils.ModuleUtilsMixin

## TFPreTrainedModel

[[autodoc]] TFPreTrainedModel
    - push_to_hub
    - all

## TFModelUtilsMixin

[[autodoc]] modeling_tf_utils.TFModelUtilsMixin

## FlaxPreTrainedModel

[[autodoc]] FlaxPreTrainedModel
    - push_to_hub
    - all

## Pushing to the Hub

[[autodoc]] utils.PushToHubMixin

## Sharded checkpoints

[[autodoc]] modeling_utils.load_sharded_checkpoint
