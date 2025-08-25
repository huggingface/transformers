<!--版权所有 2020 年 HuggingFace 团队。保留所有权利。

根据 Apache 许可证 2.0 版本许可，除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则依照许可证分发的软件是基于“原样”提供的，不附带任何明示或暗示的担保或条件。有关特定语言下权限的限制和限制，请参阅许可证。-->

# 模型

基类 [`PreTrainedModel`]、[`TFPreTrainedModel`] 和 [`FlaxPreTrainedModel`] 实现了从本地文件或目录加载/保存模型的常用方法，或者从库上提供的预训练模型配置（从 HuggingFace 的 AWS S3 存储库下载）加载模型。

[`PreTrainedModel`] 和 [`TFPreTrainedModel`] 还实现了一些所有模型共有的方法：

- 在向量词嵌入增加新词汇时调整输入标记（token）的大小
- 对模型的注意力头进行修剪。

其他的通用方法在 [`~modeling_utils.ModuleUtilsMixin`]（用于 PyTorch 模型）和 [`~modeling_tf_utils.TFModuleUtilsMixin`]（用于 TensorFlow 模型）中定义；文本生成方面的方法则定义在 [`~generation.GenerationMixin`]（用于 PyTorch 模型）、[`~generation.TFGenerationMixin`]（用于 TensorFlow 模型）和 [`~generation.FlaxGenerationMixin`]（用于 Flax/JAX 模型）中。

## PreTrainedModel

[[autodoc]] PreTrainedModel
    - push_to_hub
    - all

<a id='from_pretrained-torch-dtype'></a>

### 大模型加载

在 Transformers 4.20.0 中，[`~PreTrainedModel.from_pretrained`] 方法已重新设计，以适应使用 [Accelerate](https://huggingface.co/docs/accelerate/big_modeling) 加载大型模型的场景。这需要您使用的 Accelerate 和 PyTorch 版本满足： Accelerate >= 0.9.0， PyTorch >= 1.9.0。除了创建完整模型，然后在其中加载预训练权重（这会占用两倍于模型大小的内存空间，一个用于随机初始化模型，一个用于预训练权重），我们提供了一种选项，将模型创建为空壳，然后只有在加载预训练权重时才实例化其参数。

此外，如果内存不足以放下加载整个模型（目前仅适用于推理），您可以直接将模型放置在不同的设备上。使用 `device_map="auto"`，Accelerate 将确定将每一层放置在哪个设备上，以最大化使用最快的设备（GPU），并将其余部分卸载到 CPU，甚至硬盘上（如果您没有足够的 GPU 内存 或 CPU 内存）。即使模型分布在几个设备上，它也将像您通常期望的那样运行。

```python
from transformers import AutoModelForSeq2SeqLM

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto")
```

您可以通过 `hf_device_map` 属性来查看模型是如何在设备上分割的：

```python
t0pp.hf_device_map
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

您还可以按照相同的格式（一个层名称到设备的映射关系的字典）编写自己的设备映射规则。它应该将模型的所有参数映射到给定的设备上，如果该层的所有子模块都在同一设备上，您不必详细说明其中所有子模块的位置。例如，以下设备映射对于 T0pp 将正常工作（只要您有 GPU 内存）：

```python
device_map = {"shared": 0, "encoder": 0, "decoder": 1, "lm_head": 1}
```

另一种减少模型内存影响的方法是以较低精度的 dtype（例如 `torch.float16`）实例化它，或者使用下面介绍的直接量化技术。

### 模型实例化 dtype

在 PyTorch 下，模型通常以 `torch.float32` 格式实例化。如果尝试加载权重为 fp16 的模型，这可能会导致问题，因为它将需要两倍的内存。为了克服此限制，您可以使用 `dtype` 参数显式传递所需的 `dtype`：

```python
model = T5ForConditionalGeneration.from_pretrained("t5", dtype=torch.float16)
```
或者，如果您希望模型始终以最优的内存模式加载，则可以使用特殊值 `"auto"`，然后 `dtype` 将自动从模型的权重中推导出：
```python
model = T5ForConditionalGeneration.from_pretrained("t5", dtype="auto")
```

也可以通过以下方式告知从头开始实例化的模型要使用哪种 `dtype`：

```python
config = T5Config.from_pretrained("t5")
model = AutoModel.from_config(config)
```

由于 PyTorch 的设计，此功能仅适用于浮点类型。


## ModuleUtilsMixin

[[autodoc]] modeling_utils.ModuleUtilsMixin

TFPreTrainedModel
[[autodoc]] TFPreTrainedModel
    - push_to_hub
    - all

## TFModelUtilsMixin
[[autodoc]] modeling_tf_utils.TFModelUtilsMixin

FlaxPreTrainedModel
[[autodoc]] FlaxPreTrainedModel
    - push_to_hub
    - all

## 推送到 Hub
[[autodoc]] utils.PushToHubMixin

## 分片检查点
[[autodoc]] modeling_utils.load_sharded_checkpoint
