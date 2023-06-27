<!--版权所有 2022 年的 HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权; 您不得使用此文件，除非符合许可证。您可以在下面获得许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“按原样”分发，不附带任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件采用 Markdown 格式，但包含我们 doc-builder 的特定语法（类似于 MDX），在您的 Markdown 查看器中可能无法正确渲染。
-->
# FocalNet

## 概述

FocalNet 模型在 Jianwei Yang、Chunyuan Li、Xiyang Dai、Lu Yuan、Jianfeng Gao 的 [Focal Modulation Networks](https://arxiv.org/abs/2203.11926) 一文中提出。FocalNets 通过焦点调制机制完全替代了自注意力（在 [ViT](vit) 和 [Swin](swin) 等模型中使用）来建模视觉中的标记交互。作者声称，在图像分类、目标检测和分割等任务上，FocalNets 在类似的计算成本下优于基于自注意力的模型。

来自论文的摘要如下：

*我们提出了焦点调制网络（简称 FocalNets），其中自注意力（SA）完全被焦点调制机制替代，用于建模视觉中的标记交互。焦点调制包括三个组成部分：（i）层次化上下文编码，使用一系列深度卷积层实现，以从短到长的范围编码视觉上下文，（ii）门控聚合，根据查询标记的内容选择性地收集上下文，并且（iii）逐元素调制或仿射变换，将聚合的上下文注入查询中。大量实验证明，FocalNets 在图像分类、目标检测和分割等任务上优于最先进的自注意力对应模型（例如 Swin 和 Focal Transformers），且计算成本相近。具体而言，FocalNets 在小型和基础模型尺寸上在 ImageNet-1K 上实现了 82.3%和 83.9%的 top-1 准确率。在 224 分辨率下在 ImageNet-22K 上进行预训练后，它在分辨率为 224 和 384 时分别达到了 86.5%和 87.3%的 top-1 准确率。在转移到下游任务时，FocalNets 表现出明显的优势。对于使用 Mask R-CNN 进行目标检测，1\times 训练的 FocalNet 基础模型优于 Swin 对应模型 2.1 个点，并且已经超过了以 3\times 计划训练的 Swin 模型（49.0 对 48.5）。对于使用 UPerNet 进行语义分割，单尺度下的 FocalNet 基础模型优于 Swin 2.4 个点，并且在多尺度上也超过了 Swin（50.5 对 49.7）。使用大型 FocalNet 和 Mask2former，我们在 ADE20K 语义分割上实现了 58.5 的 mIoU，并在 COCO Panoptic Segmentation 上实现了 57.9 的 PQ。使用巨型 FocalNet 和 DINO，在 COCO minival 和 test-dev 上分别实现了 64.3 和 64.4 的 mAP，建立在像 Swinv2-G 和 BEIT-3 这样更大的基于注意力的模型之上。* 

提示：

- 可以使用 [`AutoImageProcessor`] 类来为模型准备图像。
此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码可以在 [这里](https://github.com/microsoft/FocalNet) 找到。

## FocalNetConfig

[[autodoc]] FocalNetConfig

## FocalNetModel

[[autodoc]] FocalNetModel
    - forward

## FocalNetForMaskedImageModeling

[[autodoc]] FocalNetForMaskedImageModeling
    - forward

## FocalNetForImageClassification

[[autodoc]] FocalNetForImageClassification
    - forward
