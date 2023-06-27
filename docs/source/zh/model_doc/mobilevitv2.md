<!--版权所有2023年HuggingFace团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）的规定，您除非符合许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请参阅许可证中的特定语言规定权限和限制。⚠️请注意，此文件是 Markdown 文件，但包含我们的文档生成器的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确显示。
-->

# MobileViTV2

## 概述
MobileViTV2 模型是由 Sachin Mehta 和 Mohammad Rastegari 在 [可分离自注意力用于移动视觉变换](https://arxiv.org/abs/2206.02680) 中提出的。


MobileViTV2 是 MobileViT 的第二个版本，通过将 MobileViT 中的多头自注意力替换为可分离自注意力构建而成。

论文中的摘要如下：

*移动视觉变换器（MobileViT）可以在多个移动视觉任务（包括分类和检测）中实现最先进的性能。尽管这些模型的参数较少，但与基于卷积神经网络的模型相比，它们的延迟很高。MobileViT 的主要效率瓶颈是变换器中的多头自注意力（MHA），其与标记（或补丁）数 k 的时间复杂度为 O(k ²)。此外，MHA 需要昂贵的操作（例如，批次矩阵乘法）来计算自注意力，影响资源受限设备上的延迟。本文介绍了一种具有线性复杂度的可分离自注意力方法，即 O(k)。所提出的方法的一个简单但有效的特点是使用逐元素操作来计算自注意力，这使其成为资源受限设备的不错选择。改进的 MobileViTV2 模型在多个移动视觉任务（包括 ImageNet 对象分类和 MS-COCO 对象检测）中达到了最先进的性能。MobileViTV2 具有约 300 万个参数，在 ImageNet 数据集上实现了 75.6%的 top-1 准确率，相比 MobileViT 提高了约 1%，同时在移动设备上运行速度提高了 3.2 倍。*

提示：

- MobileViTV2 更像是 CNN 而不是 Transformer 模型。它不适用于序列数据，而是适用于图像批次。与 ViT 不同，没有嵌入。骨干模型输出一个特征图。- 您可以使用 [`MobileViTImageProcessor`] 来为模型准备图像。请注意，如果您自己进行预处理，则预训练检查点需要图像以 BGR 像素顺序（而不是 RGB）为顺序。- 可用的图像分类检查点是在 [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)（也称为 ILSVRC 2012，包含 130 万张图像和 1,000 个类别的集合）上进行预训练的。- 分割模型使用 [DeepLabV3](https://arxiv.org/abs/1706.05587) 头部。可用的语义分割检查点是在 [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 上进行预训练的。

此模型由 [shehan97](https://huggingface.co/shehan97) 贡献。原始代码可以在 [这里](https://github.com/apple/ml-cvnets) 找到。

## MobileViTV2Config

[[autodoc]] MobileViTV2Config

## MobileViTV2Model

[[autodoc]] MobileViTV2Model
    - forward

## MobileViTV2ForImageClassification

[[autodoc]] MobileViTV2ForImageClassification
    - forward

## MobileViTV2ForSemanticSegmentation

[[autodoc]] MobileViTV2ForSemanticSegmentation
    - forward