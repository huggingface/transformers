<!-- 版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；您除非符合许可证的要求，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的明示或默示的保证或条件。请参阅许可证中的特定语言的权限和限制。【注意】此文件是 Markdown 格式，但包含特定语法，用于我们的文档生成器（类似于 MDX），在您的 Markdown 查看器中可能无法正确呈现。
⚠️请注意，此文件是 Markdown 格式，但包含特定语法，用于我们的文档生成器（类似于 MDX），在您的 Markdown 查看器中可能无法正确呈现。
-->

# Swin Transformer

## 概述

Swin Transformer 是由 Ze Liu、Yutong Lin、Yue Cao、Han Hu、Yixuan Wei、Zheng Zhang、Stephen Lin 和 Baining Guo 提出的 [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
中提出的。以下是论文的摘要内容：

*本文提出了一种新的视觉 Transformer，称为 Swin Transformer，它可以作为通用的计算机视觉主干。从语言到视觉的 Transformer 的适应挑战源于两个领域之间的差异，例如视觉实体的规模差异较大，图像中的像素分辨率与文本中的单词相比较高。为了解决这些差异，我们提出了一种使用移位窗口计算表示的分层 Transformer。移位窗口方案通过将自注意计算限制在不重叠的局部窗口上，从而提高了效率，并允许窗口间的连接。这种分层结构具有在不同尺度上建模的灵活性，并且相对于图像大小具有线性计算复杂度。Swin Transformer 的这些特点使其适用于各种视觉任务，包括图像分类（ImageNet-1K 上的 87.3 的 top-1 准确率）以及目标检测（COCO test-dev 上的 58.7 的 box AP 和 51.1 的 mask AP）和语义分割（ADE20K val 上的 53.5 的 mIoU）。它的性能大大超过了之前的最新技术，COCO 上的+2.7 的 box AP 和+2.6 的 mask AP，以及 ADE20K 上的+3.2 的 mIoU，展示了基于 Transformer 的模型作为视觉主干的潜力。分层设计和移位窗口方法也对所有 MLP 架构有益。*

提示：
- 您可以使用 [`AutoImageProcessor`] API 为模型准备图像。
- Swin 填充输入以支持任何输入高度和宽度（如果可被 `32` 整除）。
- Swin 可以用作 *主干*。当 `output_hidden_states = True` 时，它将同时输出 `hidden_states` 和 `reshaped_hidden_states`。`reshaped_hidden_states` 的形状为
`(batch, num_channels, height, width)`，而不是 `(batch_size, sequence_length, num_channels)`。
< img src =" https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/swin_transformer_architecture.png "
alt = "drawing" width = "600"/>

<small> Swin Transformer 架构。来自 <a href="https://arxiv.org/abs/2102.03334"> 原始论文 </a>。</small>

该模型由 [novice03](https://huggingface.co/novice03) 贡献。此模型的 Tensorflow 版本由 [amyeroberts](https://huggingface.co/amyeroberts) 贡献。原始代码可在 [此处](https://github.com/microsoft/Swin-Transformer) 找到。

## 资源

以下是官方 Hugging Face 和社区（由🌎表示）资源列表，可帮助您入门使用 Swin Transformer。

- [`SwinForImageClassification`] 支持此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

除此之外：
- [`SwinForMaskedImageModeling`] 支持此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining)。

如果您有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将进行审核！该资源应该展示出一些新的东西，而不是重复现有的资源。

## SwinConfig

[[autodoc]] SwinConfig


## SwinModel

[[autodoc]] SwinModel
    - forward

## SwinForMaskedImageModeling

[[autodoc]] SwinForMaskedImageModeling
    - forward

## SwinForImageClassification

[[autodoc]] transformers.SwinForImageClassification
    - forward

## TFSwinModel

[[autodoc]] TFSwinModel
    - call

## TFSwinForMaskedImageModeling

[[autodoc]] TFSwinForMaskedImageModeling
    - call

## TFSwinForImageClassification

[[autodoc]] transformers.TFSwinForImageClassification
    - call
