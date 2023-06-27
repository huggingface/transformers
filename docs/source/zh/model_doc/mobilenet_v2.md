<!--版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；您除非遵守许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”分发的基础上，不附带任何形式的保证或条件，无论是明示还是暗示。请参阅许可证中的特定语言授权权限和限制。
⚠️ 请注意，此文件是 Markdown 格式的，但包含特定于我们的 doc-builder（类似于 MDX）的语法，可能不会在您的 Markdown 查看器中正确显示。渲染。
-->
# MobileNet V2

## 概述

MobileNet 模型是由 Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen 在 [MobileNetV2：反向残差和线性瓶颈](https://arxiv.org/abs/1801.04381) 中提出的。
论文中的摘要如下所示：

*在本文中，我们描述了一种新的移动架构 MobileNetV2，它在多个任务和基准测试中提高了移动模型的性能，并在不同模型尺寸的光谱上进行了改进。我们还描述了一种有效的方法，将这些移动模型应用于我们称之为 SSDLite 的新颖框架中的目标检测。此外，我们还演示了如何通过我们称之为 Mobile DeepLabV3 的简化形式来构建移动语义分割模型。*
*MobileNetV2 架构基于反向残差结构，其中残差块的输入和输出是细的瓶颈层，而不是传统残差模型中在输入和 MobileNetV2 使用轻量级的深度卷积来过滤中间扩展层的特征。此外，我们发现，在窄层中去除非线性是保持表示能力的重要因素。我们证明了这可以提高性能，并提供了导致此设计的直觉。最后，我们的方法允许将输入/输出域与转换的表现力解耦，从而为进一步的分析提供了一个方便的框架。我们通过 Imagenet 分类，COCO 目标检测，VOC 图像分割来衡量我们的性能。我们评估了准确性和操作次数（乘加次数）以及参数数量之间的权衡。*

提示：

- 检查点的命名方式为**mobilenet\_v2\_ *depth*\_*size***，例如**mobilenet\_v2\_1.0\_224**，其中**1.0**是深度乘数（有时也称为 "alpha" 或宽度乘数），**224**是模型训练的输入图像的分辨率。

- 即使检查点是在特定大小的图像上训练的，模型也可以处理任意大小的图像。支持的最小图像尺寸为 32x32。
- 可以使用 [`MobileNetV2ImageProcessor`] 来准备图像以供模型使用。

- 可用的图像分类检查点是在 [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)（也称为 ILSVRC 2012，包含 130 万张图像和 1000 个类别）上进行预训练的。但是，该模型预测的类别有 1001 个：来自 ImageNet 的 1000 个类别以及额外的“背景”类别（索引为 0）。

- 分割模型使用 [DeepLabV3+](https://arxiv.org/abs/1802.02611) 头部。可用的语义分割检查点在 [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) 上进行了预训练。

- 原始的 TensorFlow 检查点使用与 PyTorch 不同的填充规则，这要求模型在推理时确定填充量，因为这取决于输入图像的大小。要使用本机的 PyTorch 填充行为，请创建一个带有 `tf_padding = False` 的 [`MobileNetV2Config`]。

不支持的功能：

- [`MobileNetV2Model`] 输出最后隐藏状态的全局池化版本。在原始模型中，可以使用具有固定 7x7 窗口和步幅 1 的平均池化层，而不是全局池化。对于大于推荐图像尺寸的输入，这将给出大于 1x1 的池化输出。Hugging Face 的实现不支持此功能。

- 原始的 TensorFlow 检查点包括量化模型。我们不支持这些模型，因为它们包括额外的“FakeQuantization”操作来解量化权重。

- 通常会提取来自扩展层的输出，例如索引 10 和 13 的输出，以及来自最后的 1x1 卷积层的输出，以进行下游处理。使用 `output_hidden_states=True` 将返回所有中间层的输出。目前无法将其限制为特定的层。

- DeepLabV3+分割头部不使用骨干网络的最后一层卷积层，但此层仍会计算。目前无法告知 [`MobileNetV2Model`] 应运行到哪一层。

该模型由 [matthijs](https://huggingface.co/Matthijs) 贡献。可以在 [此处找到主模型的原始代码和权重](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet) 以及 [此处的 DeepLabV3+](https://github.com/tensorflow/models/tree/master/research/deeplab)。

## 资源

官方 Hugging Face 和社区（通过🌎表示）的资源列表，可帮助您开始使用 MobileNetV2。

<PipelineTag pipeline="image-classification"/>
- [`MobileNetV2ForImageClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 支持。

- 另请参阅：[图像分类任务指南](../tasks/image_classification)

**语义分割**

- [语义分割任务指南](../tasks/semantic_segmentation)
如果您有兴趣提交资源以纳入此处，请随时提出拉取请求，我们将进行审查！资源应该展示出与现有资源不同的新东西，而不是重复现有资源。

## MobileNetV2Config

[[autodoc]] MobileNetV2Config

## MobileNetV2FeatureExtractor

[[autodoc]] MobileNetV2FeatureExtractor
    - preprocess
    - post_process_semantic_segmentation

## MobileNetV2ImageProcessor

[[autodoc]] MobileNetV2ImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## MobileNetV2Model

[[autodoc]] MobileNetV2Model
    - forward

## MobileNetV2ForImageClassification

[[autodoc]] MobileNetV2ForImageClassification
    - forward

## MobileNetV2ForSemanticSegmentation

[[autodoc]] MobileNetV2ForSemanticSegmentation
    - forward
