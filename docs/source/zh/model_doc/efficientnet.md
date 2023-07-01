<!--版权所有 2023 年 HuggingFace 团队。保留所有权利。

根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获得许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不提供任何明示或暗示的担保或条件。请查看许可证以了解许可证下的特定语言权限和限制。
⚠️ 请注意，此文件是 Markdown 格式的，但包含我们的文档构建器（类似于 MDX）的特定语法，您的 Markdown 查看器可能无法正确渲染。
具体的语言权限和限制，请参阅许可证。-->


# EfficientNet

## 概述

EfficientNet 模型是由 Mingxing Tan 和 Quoc V. Le 在 [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) 中提出的。EfficientNets 是一系列图像分类模型，其在准确率方面取得了最先进的成果，同时比之前的模型小一个数量级且速度更快。

以下是论文中的摘要：



*卷积神经网络（ConvNets）通常在固定的资源预算下开发，如果有更多资源可用，则通过扩大规模以提高准确性。在本文中，我们系统地研究了模型扩展并确定了平衡网络深度、宽度和分辨率的关键性，并通过使用一个简单但非常有效的复合系数均匀地扩展深度/宽度/分辨率的所有维度来提出一种新的扩展方法。我们通过扩大 MobileNets 和 ResNet 的规模来证明了此方法的有效性。
为了进一步提高性能，我们使用神经架构搜索设计了一个新的基准网络并将其扩展为一个模型系列，称为 EfficientNets，这些模型在准确性和效率方面比之前的 ConvNets 要好得多。特别是，我们的 EfficientNet-B7 在 ImageNet 上实现了 84.3 ％的前 1 位准确率，同时比现有最佳 ConvNet 小 8.4 倍且推断速度快 6.1 倍。我们的 EfficientNets 还具有良好的迁移能力，并在 CIFAR-100（91.7 ％），Flowers（98.8 ％）和其他 3 个迁移学习数据集上实现了最先进的准确性，参数数量减少了一个数量级。*


此模型由 [adirik](https://huggingface.co/adirik) 贡献。

原始代码可在 [此处](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) 找到。


## EfficientNetConfig

[[autodoc]] EfficientNetConfig

## EfficientNetImageProcessor

[[autodoc]] EfficientNetImageProcessor
    - preprocess

## EfficientNetModel

[[autodoc]] EfficientNetModel
    - forward

## EfficientNetForImageClassification

[[autodoc]] EfficientNetForImageClassification
    - forward