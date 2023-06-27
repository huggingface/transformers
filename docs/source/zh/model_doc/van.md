<!--版权所有 2022 年的 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在许可证的副本中获取许可证。
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”基础分发，不附带任何形式的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->

# VAN 模型

## 概述

VAN 模型在 [Visual Attention Network](https://arxiv.org/abs/2202.09741) 一文中由 Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu, Ming-Ming Cheng, Shi-Min Hu 提出。

本文介绍了一种基于卷积操作的新型注意力层，能够捕捉局部和远距离关系。这是通过结合普通卷积层和大卷积核卷积层实现的。后者使用扩张卷积来捕捉远距离相关性。

论文摘要如下:

*虽然自注意机制最初是为自然语言处理任务设计的，但它最近在各个计算机视觉领域引起了轰动。然而，图像的二维性质对于应用自注意力在计算机视觉中带来了三个挑战。(1)将图像视为 1D 序列忽略了它们的 2D 结构。(2)二次复杂度对于高分辨率图像来说太昂贵了。(3)它只捕捉了空间适应性，但忽略了通道适应性。在本文中，我们提出了一种新颖的大卷积核注意力（LKA）模块，以在避免上述问题的同时实现自适应和长程相关性。我们进一步介绍了一种基于 LKA 的新型神经网络，即 Visual Attention Network（VAN）。尽管非常简单，但 VAN 在包括图像分类、目标检测、语义分割、实例分割等广泛实验中，以极大的优势超越了最先进的视觉变换器和卷积神经网络。代码可在 [此 https 网址](https://github.com/Visual-Attention-Network/VAN-Classification) 上找到.*

提示:

- VAN 没有嵌入层，因此 `hidden_states` 的长度将等于阶段数。

下图显示了 Visual Attention Layer 的体系结构。摘自 [原始论文](https://arxiv.org/abs/2202.09741)。
<img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/van_architecture.png"/>
此模型由 [Francesco](https://huggingface.co/Francesco) 贡献。原始代码可在 [此处](https://github.com/Visual-Attention-Network/VAN-Classification) 找到。

## 资源

以下是官方 Hugging Face 和社区（由🌎表示）资源列表，可帮助您开始使用 VAN。

<PipelineTag pipeline="image-classification"/>
- [`VanForImageClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 支持。

- 另请参阅：[图像分类任务指南](../tasks/image_classification)
如果您有兴趣提交资源以被包含在此处，请随时提出拉取请求，我们将对其进行审核！该资源应理想地展示出新的东西，而不是重复现有的资源。


## VanConfig

[[autodoc]] VanConfig


## VanModel

[[autodoc]] VanModel
    - forward


## VanForImageClassification

[[autodoc]] VanForImageClassification
    - forward

