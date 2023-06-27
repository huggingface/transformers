<!--版权所有2022年携抱团队。保留所有权利。-->
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。您可以在
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按照“按原样”基础分发的，不提供任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们 doc-builder 的特定语法（类似于 MDX），可能无法在您的 Markdown 查看器中正确呈现。
-->
# ResNet

## 概述

ResNet 模型是由 Kaiming He、Xiangyu Zhang、Shaoqing Ren 和 Jian Sun 在 [深度残差学习用于图像识别](https://arxiv.org/abs/1512.03385) 中提出的。我们的实现遵循 [Nvidia](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch) 做出的小改动，我们在瓶颈的 `3x3` 卷积中应用 `stride=2` 进行下采样，而不是在第一个 `1x1` 卷积中。这通常被称为“ResNet v1.5”。

ResNet 引入了残差连接，可以训练具有未知层数的网络（高达 1000 层）。ResNet 赢得了 2015 年 ILSVRC 和 COCO 竞赛，这是深度计算机视觉的一个重要里程碑。
以下是来自论文的摘要：

*更深的神经网络更难训练。我们提出了一个残差学习框架，以便训练比以前使用的网络更深的网络变得更容易。我们明确地将层重新表述为相对于层输入的学习残差函数，而不是学习无参考函数。我们提供了全面的经验证据，显示这些残差网络更容易优化，并且可以通过显著增加的深度获得准确性。在 ImageNet 数据集上，我们评估了具有多达 152 层深度的残差网络，比 VGG 网络深 8 倍，但复杂度较低。这个结果在 ILSVRC 2015 分类任务中获得了 3.57%的错误率。我们还在带有 100 层和 1000 层的 CIFAR-10 上进行了分析。表示的深度对于许多视觉识别任务非常重要。仅仅由于我们极深的表示，我们在 COCO 目标检测数据集上获得了 28%的相对改进。深度残差网络是我们提交给 ILSVRC 和 COCO 2015 竞赛的基础，我们在 ImageNet 检测、ImageNet 定位、COCO 检测和 COCO 分割任务中也获得了第一名。*

提示：

- 您可以使用 [`AutoImageProcessor`] 来为模型准备图像。

下图显示了 ResNet 的体系结构。取自 [原始论文](https://arxiv.org/abs/1512.03385)。
<img width="600" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/resnet_architecture.png"/>
该模型由 [Francesco](https://huggingface.co/Francesco) 贡献。该模型的 TensorFlow 版本由 [amyeroberts](https://huggingface.co/amyeroberts) 添加。原始代码可以在 [此处](https://github.com/KaimingHe/deep-residual-networks) 找到。

## 资源

以下是官方 Hugging Face 和社区（由🌎表示）资源列表，可帮助您开始使用 ResNet。
<PipelineTag pipeline="image-classification"/>
- [`ResNetForImageClassification`] 在这个 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) 中得到支持。

- 另请参阅：[图像分类任务指南](../tasks/image_classification)
如果您有兴趣提交要包含在此处的资源，请随时发起拉取请求，我们将进行审核！资源应该展示出一些新的东西，而不是重复现有的资源。

## ResNetConfig

[[autodoc]] ResNetConfig


## ResNetModel

[[autodoc]] ResNetModel
    - forward


## ResNetForImageClassification

[[autodoc]] ResNetForImageClassification
    - forward


## TFResNetModel

[[autodoc]] TFResNetModel
    - call


## TFResNetForImageClassification

[[autodoc]] TFResNetForImageClassification
    - call

## FlaxResNetModel

[[autodoc]] FlaxResNetModel
    - __call__

## FlaxResNetForImageClassification

[[autodoc]] FlaxResNetForImageClassification
    - __call__
