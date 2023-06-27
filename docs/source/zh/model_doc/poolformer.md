<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按 "原样" 分发的，不附带任何形式的保证或条件。请参阅许可证以获取特定语言下的权限和限制。注意：此文件是 Markdown 格式，但包含我们的文档构建器的特定语法（类似于 MDX），可能无法在 Markdown 查看器中正确显示。
⚠️ 请注意，此文件是 Markdown 格式的，但包含了我们的文档构建器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确呈现。渲染。
-->
# PoolFormer

## 概览（Overview）

PoolFormer 模型是由 Sea AI Labs 在 [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418) 中提出的。

与设计复杂的令牌混合器以实现 SOTA 性能不同，这项工作的目标是展示变压器模型的竞争力主要源于通用架构 MetaFormer。

论文中的摘要如下所示：

*变压器已经在计算机视觉任务中展现出巨大潜力。人们普遍认为，变压器中基于注意力的令牌混合器模块对其竞争力做出了最大贡献。然而，最近的研究表明，变压器中基于注意力的模块可以被空间 MLP 替代，替代模型仍然表现出很好的性能。基于这一观察，我们假设变压器的通用架构，而不是特定的令牌混合器模块，对模型的性能更为重要。为了验证这一点，我们故意将变压器中的注意力模块替换为一个非常简单的空间汇聚运算符，只进行最基本的令牌混合。令人惊讶的是，我们观察到，得到的模型（称为 PoolFormer）在多个计算机视觉任务上都取得了竞争性的性能。例如，在 ImageNet-1K 上，PoolFormer 以 82.1%的 top-1 准确率超过了调优的视觉变压器/MLP-like 基线 DeiT-B/ResMLP-B24 的 0.3%/1.1%准确率，而参数减少了 35%/52%，MAC 减少了 48%/60%。PoolFormer 的有效性验证了我们的假设，并促使我们提出“MetaFormer”这一概念，它是从变压器中抽象出来的通用架构，而不指定令牌混合器。基于广泛的实验，我们认为 MetaFormer 是实现最近变压器和 MLP-like 模型在视觉任务上取得优越结果的关键因素。这项工作呼吁未来的研究更专注于改进 MetaFormer，而不是仅关注令牌混合器模块。此外，我们提出的 PoolFormer 可以作为未来 MetaFormer 架构设计的起点基线。*

下图展示了 PoolFormer 的架构，摘自 [原始论文](https://arxiv.org/abs/2111.11418)。

<img width="600" src="https://user-images.githubusercontent.com/15921929/142746124-1ab7635d-2536-4a0e-ad43-b4fe2c5a525d.png"/>

提示：

- PoolFormer 具有分层架构，其中使用简单的平均池化层代替了注意力。模型的所有检查点都可以在 [hub](https://huggingface.co/models?other=poolformer) 上找到。
- 您可以使用 [`PoolFormerImageProcessor`] 来为模型准备图像。- 与大多数模型一样，PoolFormer 有不同的规格，具体详情请参见下表。

| **模型变体** | **深度**    | **隐藏大小**    | **参数（M）** | **ImageNet-1k top 1** || :---------------: | ------------- | ------------------- | :------------: | :-------------------: || s12               | [2, 2, 6, 2]  | [64, 128, 320, 512] | 12             | 77.2                  || s24               | [4, 4, 12, 4] | [64, 128, 320, 512] | 21             | 80.3                  || s36               | [6, 6, 18, 6] | [64, 128, 320, 512] | 31             | 81.4                  || m36               | [6, 6, 18, 6] | [96, 192, 384, 768] | 56             | 82.1                  || m48               | [8, 8, 24, 8] | [96, 192, 384, 768] | 73             | 82.5                  |

此模型由 [heytanay](https://huggingface.co/heytanay) 贡献。原始代码可在 [此处](https://github.com/sail-sg/poolformer) 找到。

## 资源（Resources）

以下是一些官方 Hugging Face 和社区（由🌎表示）资源，可帮助您入门 PoolFormer。

<PipelineTag pipeline="image-classification"/>
- [`PoolFormerForImageClassification`] 支持此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。

- 另请参阅：[图像分类任务指南](../tasks/image_classification)
如果您有兴趣提交要包含在此处的资源，请随时提出拉取请求，我们将进行审核！该资源理想情况下应该展示出与现有资源不同的新内容，而不是重复已有资源。

## PoolFormerConfig

[[autodoc]] PoolFormerConfig

## PoolFormerFeatureExtractor

[[autodoc]] PoolFormerFeatureExtractor
    - __call__

## PoolFormerImageProcessor

[[autodoc]] PoolFormerImageProcessor
    - preprocess

## PoolFormerModel

[[autodoc]] PoolFormerModel
    - forward

## PoolFormerForImageClassification

[[autodoc]] PoolFormerForImageClassification
    - forward