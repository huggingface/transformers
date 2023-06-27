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

# 🤗 Transformers 哲学
🤗 Transformers 是一个为以下人群构建的倾向性库：

- 寻求使用、研究或扩展大规模 Transformers 模型的机器学习研究人员和教育工作者。
- 希望对这些模型进行微调或在生产环境中使用它们（或二者兼有）的实践者。- 只想下载预训练模型并将其用于解决特定机器学习任务的工程师。

该库的设计有两个强烈的目标：

1. 尽可能易于使用和快速使用：

  - 我们极大地限制了需要学习的用户界面抽象，实际上几乎没有抽象，只需学习三个标准类别以使用每个模型：[配置](main_classes/configuration)，[模型](main_classes/model) 和一个预处理类（用于 NLP 的 [分词器 (Tokenizer)](main_classes/tokenizer)、用于视觉的 [图像处理器 (Image Processor)](main_classes/image_processor)、用于音频的 [特征提取器](main_classes/feature_extractor) 和用于多模态输入的 [处理器](main_classes/processors)）。 
  - 所有这些类别都可以通过使用 `from_pretrained()` 方法以简单、统一的方式从预训练实例进行初始化，该方法会从 [Hugging Face Hub](https://huggingface.co/models) 或您自己的保存的检查点下载（如果需要），缓存和加载相关类别实例和关联数据（配置的超参数，分词器 (Tokenizer)的词汇表，模型的权重）。
  - 在这三个基本类别之上，该库提供了两个 API：[`pipeline`] 用于快速    在给定任务上使用模型进行推断，以及 [`Trainer`] 用于快速训练或微调 PyTorch 模型（所有 TensorFlow 模型都与 `Keras.fit` 兼容）。

  - 因此，该库不是神经网络的模块化工具箱。如果您想扩展或构建该库，只需使用常规的 Python、PyTorch、TensorFlow、Keras 模块，并从库的基本类继承，以重用模型加载和保存等功能。




  
  如果您想了解有关我们的模型编码哲学的更多信息，请查看我们的 [重复自己](https://huggingface.co/blog/transformers-design-philosophy) 博文。

2. 提供与原始模型尽可能接近的性能的最新模型：
  - 我们为每种架构提供至少一个示例，该示例重现了官方作者提供的结果。  
  - 代码通常与原始代码库尽可能接近，这意味着由于转换为 TensorFlow 代码和反之亦然，一些 PyTorch 代码可能不够符合 PyTorch 的编码习惯。 

      "*pytorchic*"可以描述那些被转换为PyTorch代码的TensorFlow代码，反之亦然。

其他几个目标：
- 尽可能一致地公开模型的内部：
  - 我们使用单个 API 提供对完整隐藏状态和注意力权重的访问。  - 预处理类和基本模型 API 标准化，以便在不同模型之间轻松切换。
- 结合一些有前途的工具，用于微调和研究这些模型：
  - 一种简单而一致的方式来向词汇表和嵌入中添加新的标记以进行微调。  - 屏蔽和修剪 Transformer 头的简单方法。
- 轻松在 PyTorch、TensorFlow 2.0 和 Flax 之间切换，允许使用一个框架进行训练和使用另一个框架进行推断。

## 主要概念

该库围绕每个模型构建了三种类型的类别：

- **模型类** 可以是 PyTorch 模型（[torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)）、Keras 模型（[tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)）或 JAX/Flax 模型（[flax.linen.Module](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html)），它们与库中提供的预训练权重一起使用。
- **配置类** 存储构建模型所需的超参数（例如层数和隐藏大小）。您不必始终自己实例化这些类别。特别是，如果您使用的是预训练模型且没有进行任何修改，创建模型将自动处理实例化配置（配置是模型的一部分）。
- **预处理类** 将原始数据转换为模型接受的格式。[分词器 (Tokenizer)](main_classes/tokenizer) 存储每个模型的词汇表，并提供将字符串编码和解码为要提供给模型的标记嵌入索引列表的方法。[图像处理器 (Image Processor)](main_classes/image_processor) 预处理视觉输入，[特征提取器](main_classes/feature_extractor) 预处理音频输入，而 [处理器](main_classes/processors) 处理多模态输入。

所有这些类别都可以从预训练实例进行实例化、本地保存，并使用以下三种方法在 Hub 上共享：

- `from_pretrained()` 允许您从库本身提供的预训练版本（支持的模型可在 [Model Hub](https://huggingface.co/models) 上找到）或  用户本地（或服务器上）存储的版本实例化模型、配置和预处理类。  stored locally (or on a server) by the user.
- `save_pretrained()` 允许您本地保存模型、配置和预处理类，以便可以使用 `from_pretrained()` 重新加载。  `from_pretrained()`.
- `push_to_hub()` 允许您将模型、配置和预处理类共享到 Hub 上，以便所有人都可以轻松访问。
