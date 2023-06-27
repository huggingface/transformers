<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；您除非符合许可证，否则不得使用此文件。您可以在以下位置获取许可证副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“按原样”分发的基础上，不提供任何明示或暗示的保证或条件。请参阅许可证特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->
# RegNet

## 概述

RegNet 模型是由 Ilija Radosavovic、Raj Prateek Kosaraju、Ross Girshick、Kaiming He、Piotr Doll á r 在《设计网络设计空间》（https://arxiv.org/abs/2003.13678）中提出的。作者设计了搜索空间以执行神经架构搜索（NAS）。他们首先从一个高维搜索空间开始，然后通过根据当前搜索空间采样的最佳模型经验性地应用约束来迭代地减少搜索空间。

论文的摘要如下：

*在这项工作中，我们提出了一种新的网络设计范式。我们的目标是推进网络设计的理解，并发现适用于各种设置的设计原则。我们不是专注于设计单个网络实例，而是设计参数化网络的网络设计空间。整个过程类似于经典手动设计网络，但提升到设计空间级别。使用我们的方法，我们探索了网络设计的结构方面，并得出了一个由简单、规则网络组成的低维设计空间，我们称之为 RegNet。RegNet 参数化的核心洞见非常简单：良好网络的宽度和深度可以通过量化的线性函数来解释。我们分析了 RegNet 设计空间，并得出了与当前网络设计实践不符的有趣发现。在可比的训练设置和 flops 下，RegNet 模型在 GPU 上的性能优于流行的 EfficientNet 模型，同时速度更快，最高达到 5 倍。*

提示：

- 可以使用 [`AutoImageProcessor`] 来为模型准备图像。
- 来自 [Self-supervised Pretraining of Visual Features in the Wild](https://arxiv.org/abs/2103.01988) 的巨大 10B 模型，在十亿个 Instagram 图像上进行训练，在 [hub](https://huggingface.co/facebook/regnet-y-10b-seer) 上可用
此模型由 [Francesco](https://huggingface.co/Francesco) 贡献。

模型的 TensorFlow 版本由 [sayakpaul](https://huggingface.com/sayakpaul) 和 [ariG23498](https://huggingface.com/ariG23498) 贡献。

原始代码可以在 [这里](https://github.com/facebookresearch/pycls) 找到。

## 资源

以下是官方 Hugging Face 和社区（由🌎表示）提供的资源列表，以帮助您开始使用 RegNet。
<PipelineTag pipeline="image-classification"/>
- [`RegNetForImageClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 支持。

- 另请参阅：[图像分类任务指南](../tasks/image_classification)
如果您有兴趣提交资源以包含在此处，请随时提出拉取请求，我们将对其进行审核！该资源应该展示出新的内容，而不是重复现有资源。

## RegNetConfig

[[autodoc]] RegNetConfig


## RegNetModel

[[autodoc]] RegNetModel
    - forward


## RegNetForImageClassification

[[autodoc]] RegNetForImageClassification
    - forward

## TFRegNetModel

[[autodoc]] TFRegNetModel
    - call


## TFRegNetForImageClassification

[[autodoc]] TFRegNetForImageClassification
    - call


## FlaxRegNetModel

[[autodoc]] FlaxRegNetModel
    - __call__


## FlaxRegNetForImageClassification

[[autodoc]] FlaxRegNetForImageClassification
    - __call__