<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；在符合条件下，您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”基础分发的，不附带任何形式的保证或条件。请参阅许可证以获取特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含了特定的语法，用于我们的文档生成器（类似于 MDX），可能无法在您的 Markdown 查看器中正确呈现。
-->

# 卷积视觉变换器（CvT）

## 概述

CvT 模型在《CvT：引入卷积到视觉变换器》（[CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808)）中由 Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan 和 Lei Zhang 提出。卷积视觉变换器（CvT）通过将卷积引入 ViT 以提高性能和效率，将 ViT 的设计优势发挥到极致。

论文摘要如下所示：
*我们在本文中提出了一种新的架构，称为卷积视觉变换器（CvT），通过将卷积引入 ViT 以提高性能和效率，以实现两者设计的最佳效果。这是通过两个主要修改实现的：包含新的卷积标记嵌入的 Transformer 层次结构，以及利用卷积投影的卷积 Transformer 块。这些改变将卷积神经网络（CNN）的有益性质（如位移、缩放和失真不变性）引入了 ViT 架构，同时保持了 Transformer 的优点（如动态注意力、全局上下文和更好的泛化能力）。我们通过进行大量实验验证了 CvT 的有效性，并展示了该方法在 ImageNet-1k 上相对于其他视觉变换器和 ResNet 的最新性能，参数更少，FLOPs 更低。此外，当在更大的数据集（如 ImageNet-22k）上预训练并微调到下游任务时，性能增益仍然存在。在 ImageNet-22k 上预训练的 CvT-W24 在 ImageNet-1k 验证集上获得了 87.7\% 的 top-1 准确率。最后，我们的结果表明，位置编码（现有视觉变换器中的重要组成部分）可以在我们的模型中安全地删除，从而简化更高分辨率视觉任务的设计。* 

提示：

- CvT 模型是常规的视觉变换器，但是使用卷积进行训练。当在 ImageNet-1K 和 CIFAR-100 上微调时，它们胜过 [原始模型（ViT）](vit)。- 您可以在 [此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer) 查看有关推理以及在自定义数据上进行微调的演示笔记本（您只需将 [`ViTFeatureExtractor`] 替换为 [`AutoImageProcessor`]，将 [`ViTForImageClassification`] 替换为 [`CvtForImageClassification`]）。- 可用的检查点是：（1）仅在 [ImageNet-22k](http://www.image-net.org/)（包含 1400 万张图像和 22k 类别）上进行预训练，（2）在 ImageNet-22k 上进行微调，或（3）在 [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/)（也称为 ILSVRC 2012，包含 130 万张图像和 1000 类别）上进行微调。

该模型由 [anugunj](https://huggingface.co/anugunj) 贡献。可以在 [此处](https://github.com/microsoft/CvT) 找到原始代码。

## 资源

以下是官方 Hugging Face 和社区（通过 🌎 表示）资源列表，可帮助您开始使用 CvT。

<PipelineTag pipeline="image-classification"/>
- [`CvtForImageClassification`] 可以使用此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。

- 另请参阅：[图像分类任务指南](../tasks/image_classification)
如果您有兴趣提交要包括在此处的资源，请随时提出拉取请求，我们将进行审核！该资源应该展示出与现有资源不同的新东西，而不是重复现有资源。

## CvtConfig

[[autodoc]] CvtConfig

## CvtModel

[[autodoc]] CvtModel
    - forward

## CvtForImageClassification

[[autodoc]] CvtForImageClassification
    - forward

## TFCvtModel

[[autodoc]] TFCvtModel
    - call

## TFCvtForImageClassification

[[autodoc]] TFCvtForImageClassification
    - call

