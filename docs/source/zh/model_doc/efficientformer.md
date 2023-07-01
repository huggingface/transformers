<!-- 版权所有2022年HuggingFace团队保留所有权利。
根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证进行的软件分发基于“按原样”基础，没有任何明示或暗示的保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️请注意，此文件是Markdown格式的，但包含有关我们的doc-builder（类似于MDX）的特定语法，这些语法可能无法在您的Markdown查看器中正确显示。
-->
# EfficientFormer
## 概述
EfficientFormer模型是由Yanyu Li，Geng Yuan，Yang Wen，Eric Hu，Georgios Evangelidis，Sergey Tulyakov，Yanzhi Wang，Jian Ren在[EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)中提出的。EfficientFormer提出了一种在移动设备上运行的维度一致的纯Transformer，用于密集预测任务，如图像分类，物体检测和语义分割。

从论文中摘录的摘要如下所示：

*Vision Transformers（ViT）在计算机视觉任务中取得了快速进展，在各种基准测试中取得了有希望的结果。但是，由于参数和模型设计的大规模数量，例如注意机制，基于ViT的模型通常比轻量级卷积网络慢得多。因此，将ViT部署到实时应用程序中尤为具有挑战性，特别是在资源受限的硬件上，如移动设备。最近的努力尝试通过网络架构搜索或与MobileNet块的混合设计来减少ViT的计算复杂性，但推理速度仍然不尽人意。这引出了一个重要的问题：Transformer能否与MobileNet一样快速，同时实现高性能？为了回答这个问题，我们首先重新审视在基于ViT的模型中使用的网络架构和运算符，并确定了低效的设计。然后，我们引入了一个维度一致的纯Transformer（不包含MobileNet块）作为设计范例。最后，我们进行基于延迟的精简，得到了一系列称为EfficientFormer的最终模型。广泛的实验证明了EfficientFormer在移动设备上的性能和速度的优势。我们最快的模型EfficientFormer-L1，在ImageNet-1K上实现了79.2%的top-1准确率，仅需1.6毫秒的推理延迟在iPhone 12上（使用CoreML编译），与MobileNetV2×1.4（1.6毫秒，74.7% top-1）一样快，我们最大的模型EfficientFormer-L7在仅需7.0毫秒的延迟下实现了83.3%的准确率。我们的工作证明了适当设计的Transformer可以在移动设备上实现极低的延迟，同时保持高性能。*

该模型由[novice03](https://huggingface.co/novice03)和[Bearnardd](https://huggingface.co/Bearnardd)贡献。可以在[此处](https://github.com/snap-research/EfficientFormer)找到原始代码。此模型的TensorFlow版本由[D-Roberts](https://huggingface.co/D-Roberts)添加。

## 文档资源
- [图像分类任务指南](../tasks/image_classification)
## Documentation resources

- [Image classification task guide](../tasks/image_classification)

## EfficientFormerConfig

[[autodoc]] EfficientFormerConfig

## EfficientFormerImageProcessor

[[autodoc]] EfficientFormerImageProcessor
    - preprocess

## EfficientFormerModel

[[autodoc]] EfficientFormerModel
    - forward

## EfficientFormerForImageClassification

[[autodoc]] EfficientFormerForImageClassification
    - forward

## EfficientFormerForImageClassificationWithTeacher

[[autodoc]] EfficientFormerForImageClassificationWithTeacher
    - forward

## TFEfficientFormerModel

[[autodoc]] TFEfficientFormerModel
    - call

## TFEfficientFormerForImageClassification

[[autodoc]] TFEfficientFormerForImageClassification
    - call

## TFEfficientFormerForImageClassificationWithTeacher

[[autodoc]] TFEfficientFormerForImageClassificationWithTeacher
    - call
