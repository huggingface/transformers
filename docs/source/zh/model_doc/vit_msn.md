<!--版权 2022 年由 HuggingFace 团队保留。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；您除非符合许可证的要求，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按原样” BASIS，无论是明示还是暗示的，都没有任何保证或条件。请参阅许可证具体语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式的，但包含我们的文档构建器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确显示。
-->
# ViTMSN

## 概述

ViTMSN 模型是由 Mahmoud Assran，Mathilde Caron，Ishan Misra，Piotr Bojanowski，Florian Bordes，Pascal Vincent，Armand Joulin，Michael Rabbat 和 Nicolas Ballas 在“Masked Siamese Networks for Label-Efficient Learning”（https://arxiv.org/abs/2204.07141）中提出的。

该论文提出了一种联合嵌入架构，用于匹配带有掩码补丁的原型和未掩码补丁的原型。通过这种设置，他们的方法在低样本和极低样本范围中表现出色。这篇论文的摘要如下：*我们提出了 Masked Siamese Networks (MSN)，这是一种用于学习图像表示的自监督学习框架。我们的方法将包含随机掩码补丁的图像视图的表示与原始未掩码图像的表示进行匹配。当应用于 Vision Transformers 时，这种自监督预训练策略特别可扩展，因为网络只处理未掩码的补丁。因此，MSN 改善了联合嵌入架构的可扩展性，同时产生了高语义级别的表示，在低样本图像分类上表现出竞争力。例如，在 ImageNet-1K 上，只有 5,000 个注释图像，我们的基本 MSN 模型达到了 72.4%的 Top-1 准确率，而在 ImageNet-1K 标签的 1%的情况下，我们达到了 75.7%的 Top-1 准确率，为该基准上的自监督学习设定了新的最佳效果。* 

注意：

- MSN（Masked Siamese Networks）是一种用于 Vision Transformers（ViTs）的自监督预训练方法。预训练的目标是将未掩码视图的原型与相同图像的掩码视图的原型进行匹配。- 作者仅发布了骨干部分（ImageNet-1k 预训练）的预训练权重。

因此，要在自己的图像分类数据集上使用它，请使用 [`ViTMSNForImageClassification`] 类，该类从 [`ViTMSNModel`] 初始化。请参阅 [此笔记本](https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 以获取有关微调的详细教程。

- MSN 在低样本和极低样本范围中特别有用。值得注意的是，在微调时，它在仅使用 1%的 ImageNet-1K 标签时可以达到 75.7%的 Top-1 准确率。*

提示：

- MSN（Masked Siamese Networks）是一种用于 Vision Transformers（ViTs）的自监督预训练方法。预训练的目标是将未掩码的图像视图分配的原型与同一图像的掩码视图相匹配。- 作者仅发布了骨干部分（ImageNet-1k 预训练）的预训练权重。

因此，要在自己的图像分类数据集上使用它，请使用 [`ViTMSNForImageClassification`] 类，该类从 [`ViTMSNModel`] 初始化。请参阅 [此笔记本](https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 以获取有关微调的详细教程。

- MSN 在低样本和极低样本范围中特别有用。值得注意的是，在微调时，它在仅使用 1%的 ImageNet-1K 标签时可以达到 75.7%的 Top-1 准确率。

<img src="https://i.ibb.co/W6PQMdC/Screenshot-2022-09-13-at-9-08-40-AM.png" alt="drawing" width="600"/> 

<small> MSN 架构。来自 <a href="https://arxiv.org/abs/2204.07141"> 原始论文。</a> </small>
此模型由 [sayakpaul](https://huggingface.co/sayakpaul) 贡献。原始代码可以在 [此处](https://github.com/facebookresearch/msn) 找到。

## 资源

以下是官方 Hugging Face 和社区（使用🌎标识）资源列表，可帮助您开始使用 ViT MSN。

<PipelineTag pipeline="image-classification"/>
- [`ViTMSNForImageClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 支持。
- 参见：[图像分类任务指南](../tasks/image_classification)

如果您有兴趣提交资源以包含在此处，请随时提出拉取请求，我们将进行审查！资源应该展示出一些新东西，而不是重复现有的资源。

## ViTMSNConfig

[[autodoc]] ViTMSNConfig


## ViTMSNModel

[[autodoc]] ViTMSNModel
    - forward


## ViTMSNForImageClassification

[[autodoc]] ViTMSNForImageClassification
    - forward