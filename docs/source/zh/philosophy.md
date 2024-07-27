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



# Transformers 的设计理念

🤗 Transformers 是一个专为以下用户群体构建的库：

- 寻求使用、研究或扩展大规模 Transformers 模型的机器学习研究人员和教育者。
- 希望微调这些模型或在生产环境中使用它们（或两者兼而有之）的实际操作者。
- 只想下载预训练模型并将其用于解决给定机器学习任务的工程师。

Transformers 设计时有两个主要目标：

1. 尽可能简单快速地使用：

   - 我们尽可能地限制用户能接触的抽象层，实际上几乎没有抽象。用户只需学习三个标准类即可使用每个模型：[configuration](main_classes/configuration)、[models](main_classes/model) 和一个预处理类（用于 NLP 的 [tokenizer](main_classes/tokenizer)，用于视觉的 [image processor](main_classes/image_processor)，用于音频的 [feature extractor](main_classes/feature_extractor)，以及用于多模态输入的 [processor](main_classes/processors)）。
   - 所有这些类都可以通过一个通用的 `from_pretrained()` 方法从预训练实例中简单统一地初始化，该方法会从提供在 [Hugging Face Hub](https://huggingface.co/models) 上的预训练检查点（如果需要的话）下载、缓存和加载相关类实例及相关数据（配置的超参数、分词器的词汇表和模型的权重）。
   - 在这三个基本类之上，该库提供了两种 API：[`pipeline`] 用于快速在给定任务上使用模型进行推断，以及 [`Trainer`] 用于快速训练或微调 PyTorch 模型（所有 TensorFlow 模型与 `Keras.fit` 兼容）。
   - 因此，Transformers 不是神经网络的模块化工具箱。如果要基于 Transformers 扩展或搭建新项目，请使用常规的 Python、PyTorch、TensorFlow、Keras 模块，并从 Transformers 的基类继承以重用模型加载和保存等功能。如果想了解更多有关我们的模型代码的设计理念，请查看我们的[重复自己](https://huggingface.co/blog/transformers-design-philosophy)博文。

2. 提供与原始模型性能尽可能接近的最新模型：

   - 我们为每种架构提供至少一个示例，复现了该架构官方作者提供的结果。
   - 代码通常尽可能接近原始代码库，这意味着某些 PyTorch 代码可能不够*pytorchic*，因为它是转换后的 TensorFlow 代码，反之亦然。

其他几个目标：

- 尽可能一致地公开模型的内部：

   - 我们使用单一 API 提供对完整隐藏状态和注意力权重的访问。
   - 预处理类和基本模型 API 标准化，便于在不同模型之间轻松切换。

- 结合主观选择的有前途的工具进行模型微调和调查：

   - 简单一致的方法来向词汇表和嵌入中添加新标记以进行微调。
   - 简单的方法来屏蔽和修剪 Transformer 头部。

- 轻松在 PyTorch、TensorFlow 2.0 和 Flax 之间切换，允许使用一个框架进行训练并使用另一个进行推断。

## 主要概念

该库围绕每个模型的三类类构建：

- **模型类** 可以是 PyTorch 模型（[torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)）、Keras 模型（[tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)）或 JAX/Flax 模型（[flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html)），这些模型可以使用库中提供的预训练权重。
- **配置类** 存储构建模型所需的超参数（如层数和隐藏大小）。通常情况下，如果您使用不进行任何修改的预训练模型，则创建模型将自动处理配置的实例化（配置是模型的一部分）。
- **预处理类** 将原始数据转换为模型可接受的格式。一个 [tokenizer](main_classes/tokenizer) 存储每个模型的词汇表，并提供编码和解码字符串为要馈送到模型的令牌嵌入索引列表的方法。[Image processors](main_classes/image_processor) 预处理视觉输入，[feature extractors](main_classes/feature_extractor) 预处理音频输入，而 [processor](main_classes/processors) 则处理多模态输入。

所有这些类都可以从预训练实例中实例化、本地保存，并通过以下三种方法与 Hub 共享：

- `from_pretrained()` 允许您从库自身提供的预训练版本（支持的模型可在 [Model Hub](https://huggingface.co/models) 上找到）或用户本地（或服务器上）存储的版本实例化模型、配置和预处理类。
- `save_pretrained()` 允许您本地保存模型、配置和预处理类，以便可以使用 `from_pretrained()` 重新加载。
- `push_to_hub()` 允许您将模型、配置和预处理类共享到 Hub，以便所有人都可以轻松访问。
