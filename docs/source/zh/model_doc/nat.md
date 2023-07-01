<!--版权所有 2022 年 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证的要求，否则您不得使用此文件。您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是按原样提供的，不附带任何形式的明示或暗示的担保或条件。请参阅许可证以获取特定语言下的权限和限制。⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法
在您的 Markdown 阅读器中正确显示渲染。
-->

# 邻域注意力变换器 (Neighborhood Attention Transformer)

## 概述

NAT 是由 Ali Hassani、Steven Walton、Jiachen Li、Shen Li 和 Humphrey Shi 在 [邻域注意力变换器](https://arxiv.org/abs/2204.07143) 提出的。

它是一种基于邻域注意力的分层视觉变换器，是一种滑动窗口自注意力模式。

论文中的摘要如下：

*我们提出了邻域注意力（NA），这是一种用于视觉的高效且可扩展的滑动窗口注意力机制。NA 是一个逐像素的操作，将自注意力（SA）局限于最近的邻居像素，因此具有与 SA 相比的线性时间和空间复杂度。滑动窗口模式使 NA 的感受野可以增长，而无需额外的像素平移，并且与 Swin 变换器的窗口自注意力（WSA）不同，它保持了平移等变性。我们开发了 NATTEN（邻域注意力扩展），这是一个带有高效 C++ 和 CUDA 核心的 Python 包，使 NA 的运行速度比 Swin 的 WSA 快 40%，内存使用量减少 25%。我们进一步提出了基于 NA 的新分层变换器设计 NAT（邻域注意力变换器），它提高了图像分类和下游视觉性能。NAT 在 NAT 上的实验结果具有竞争力；NAT-Tiny 在 ImageNet 上达到了 83.2% 的 top-1 准确率，在 MS-COCO 上达到了 51.4% 的 mAP，在 ADE20K 上达到了 48.4% 的 mIoU，比具有相似大小的 Swin 模型提高了 1.9% 的 ImageNet 准确率，1.0% 的 COCO mAP 和 2.6% 的 ADE20K mIoU。*

提示：
- 可以使用 [`AutoImageProcessor`] API 来为模型准备图像。
- NAT 可以作为 *骨干* 使用。当 `output_hidden_states = True` 时，它将同时输出 `hidden_states` 和 `reshaped_hidden_states`。`reshaped_hidden_states` 的形状为 `(batch, num_channels, height, width)`，而不是 `(batch_size, height, width, num_channels)`。

注意：

- NAT 依赖于 [NATTEN](https://github.com/SHI-Labs/NATTEN/) 对邻域注意力的实现。您可以通过参考 [shi-labs.com/natten](https://shi-labs.com/natten) 在 Linux 上安装预构建的轮子，或者通过运行 `pip install natten` 在您的系统上进行构建。请注意，后者可能需要一些时间进行编译。NATTEN 尚不支持 Windows 设备。
- 目前仅支持 4 的补丁大小。

<imgsrc="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/neighborhood-attention-pattern.jpg"alt="drawing" width="600"/>

<small> 邻域注意力与其他注意力模式的比较。来自 <a href="https://arxiv.org/abs/2204.07143"> 原始论文 </a>。</small>

此模型由 [Ali Hassani](https://huggingface.co/alihassanijr) 贡献。原始代码可以在 [此处](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer) 找到。

## 资源

这是一些官方 Hugging Face 和社区（🌎）资源的列表，可帮助您开始使用 NAT。
<PipelineTag pipeline="image-classification"/>

- [`NatForImageClassification`] 支持这个 [示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) 和 [笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb)。
- 另请参阅：[图像分类任务指南](../tasks/image_classification)

如果您有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将对其进行审查！该资源应该展示出一些新的东西，而不是重复现有的资源。

## NatConfig

[[autodoc]] NatConfig


## NatModel

[[autodoc]] NatModel
    - forward

## NatForImageClassification

[[autodoc]] NatForImageClassification
    - forward
