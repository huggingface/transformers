<!--版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的规定，否则您不能使用此文件。您可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本。
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证的具体语言以了解权限和限制。
特别注意：此文件是 Markdown 格式，但包含特定于我们的文档构建器（类似于 MDX）的语法，可能在您的 Markdown 查看器中无法正确渲染。
⚠️ 请注意，此文件是 Markdown 格式，但包含特定于我们的文档构建器（类似于 MDX）的语法，可能在您的 Markdown 查看器中无法正确渲染。渲染示例：
-->

# MobileNet V1（移动网络 V1）

## 概述

MobileNet 模型是由 Andrew G. Howard、Menglong Zhu、Bo Chen、Dmitry Kalenichenko、Weijun Wang、Tobias Weyand、Marco Andreetto、Hartwig Adam 在《MobileNets：Efficient Convolutional Neural Networks for Mobile Vision Applications》（[https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)）中提出的。

论文中的摘要如下：

*我们提出了一类称为 MobileNets 的高效模型，用于移动和嵌入式视觉应用。MobileNets 基于一种简化的架构，使用深度可分离卷积构建轻量级深度神经网络。我们引入了两个简单的全局超参数，可以在延迟和准确性之间高效地权衡。这些超参数允许模型构建者根据问题的约束条件选择适合其应用的合适大小的模型。我们进行了大量的资源和准确性权衡实验，并在 ImageNet 分类任务上与其他流行模型相比表现出色。然后，我们展示了 MobileNets 在各种应用和用例中的有效性，包括目标检测、细粒度分类、人脸属性和大规模地理定位。*

提示：

- 检查点的命名方式为**mobilenet\_v1\_ *depth*\_*size***，例如**mobilenet\_v1\_1.0\_224**，其中**1.0**是深度乘数（有时也称为“alpha”或宽度乘数），**224**是模型训练时输入图像的分辨率。

- 即使检查点是在特定大小的图像上训练的，该模型也可以处理任意大小的图像。支持的最小图像大小为 32x32。
- 您可以使用 [`MobileNetV1ImageProcessor`] 来准备图像以供模型使用。
- 可用的图像分类检查点是在 [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)（也称为 ILSVRC 2012，包含 130 万张图像和 1000 个类别的集合）上进行预训练的。然而，该模型会预测 1001 个类别：ImageNet 的 1000 个类别加上额外的“background”类别（索引为 0）。
- 原始的 TensorFlow 检查点使用与 PyTorch 不同的填充规则，这要求模型在推理时确定填充量，因为这取决于输入图像的大小。要使用本机 PyTorch 的填充行为，请创建一个具有 `tf_padding = False` 的 [`MobileNetV1Config`]。

不支持的功能：

- [`MobileNetV1Model`] 输出最后一个隐藏状态的全局池化版本。在原始模型中，可以使用 7x7 的平均池化层（步幅为 2）代替全局池化。对于较大的输入，这会给出大于 1x1 像素的汇合输出。HuggingFace 的实现不支持这一点。
- 目前无法指定 `output_stride`。对于较小的输出步幅，原始模型会调用扩张卷积以防止空间分辨率进一步降低。HuggingFace 模型的输出步幅始终为 32。
- 原始的 TensorFlow 检查点包括量化模型。我们不支持这些模型，因为它们包含额外的“FakeQuantization”操作以取消量化权重。
- 通常会从点乘层的索引 5、11、12、13 提取输出，以供后续使用。使用 `output_hidden_states=True` 返回所有中间层的输出。目前无法限制为特定层。

此模型由 [matthijs](https://huggingface.co/Matthijs) 贡献。原始代码和权重可在 [此处](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) 找到。

## 资源

以下是官方 Hugging Face 资源和社区资源（由🌎表示），可帮助您开始使用 MobileNetV1。
<PipelineTag pipeline="image-classification"/>
- [`MobileNetV1ForImageClassification`] 由此 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 支持。
- 参见：[图像分类任务指南](../tasks/image_classification)
如果您有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将进行审核！资源应该展示出一些新的内容，而不是重复现有的资源。
## MobileNetV1Config

[[autodoc]] MobileNetV1Config

## MobileNetV1FeatureExtractor

[[autodoc]] MobileNetV1FeatureExtractor
    - preprocess

## MobileNetV1ImageProcessor

[[autodoc]] MobileNetV1ImageProcessor
    - preprocess

## MobileNetV1Model

[[autodoc]] MobileNetV1Model
    - forward

## MobileNetV1ForImageClassification

[[autodoc]] MobileNetV1ForImageClassification
    - forward
