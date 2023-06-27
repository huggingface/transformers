<!--版权所有2022年HuggingFace团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权；除非遵守许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件以“原样”分发，不附带任何形式的保证或条件。请参阅许可证中的具体语言，了解权限和限制。⚠️请注意，此文件为 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确显示。特定语言的权限和限制。
⚠️请注意，此文件为 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在 Markdown 查看器中正确显示。特定语言的权限和限制。
-->
# Swin2SR

## 概述

Swin2SR 模型是由 Marcos V. Conde、Ui-Jin Choi、Maxime Burchi 和 Radu Timofte 在 [Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration](https://arxiv.org/abs/2209.11345) 一文中提出的。

Swin2SR 通过引入 [Swin Transformer v2](swinv2) 层来改进 [SwinIR](https://github.com/JingyunLiang/SwinIR/) 模型，从而解决了训练不稳定、预训练和微调之间的分辨率差距以及数据饥饿等问题。

该论文的摘要如下：

*压缩在通过流媒体服务、虚拟现实或视频游戏等受带宽限制的系统进行高效传输和存储图像和视频方面起着重要作用。然而，压缩不可避免地会导致伪影和原始信息的丢失，从而严重降低视觉质量。因此，提高压缩图像的质量已成为一个热门研究主题。尽管大多数最先进的图像恢复方法都基于卷积神经网络，但其他基于 Transformer 的方法，如 SwinIR，在这些任务上表现出色。
在本文中，我们探索了新颖的 Swin Transformer V2，以改进 SwinIR 用于图像超分辨率，特别是在压缩输入场景下。使用这种方法，我们可以解决 Transformer 视觉模型训练中的主要问题，例如训练不稳定、预训练和微调之间的分辨率差距以及数据饥饿。我们在三个典型任务上进行了实验：JPEG 压缩伪影去除、图像超分辨率（经典和轻量级）以及压缩图像超分辨率。实验结果表明，我们的方法 Swin2SR 可以改善 SwinIR 的训练收敛性和性能，并且是“AIM 2022 超分辨率压缩图像和视频挑战赛”的前 5 名解决方案。*  

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/swin2sr_architecture.png"alt="drawing" width="600"/>
<small> Swin2SR 架构。

摘自 <a href="https://arxiv.org/abs/2209.11345"> 原始论文。</a> </small>
该模型由 [nielsr](https://huggingface.co/nielsr) 贡献。

原始代码可在 [此处](https://github.com/mv-lab/swin2sr) 找到。

## 资源

Swin2SR 的演示笔记本可在 [此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Swin2SR) 找到。
使用 SwinSR 进行图像超分辨率的演示空间可在 [此处](https://huggingface.co/spaces/jjourney1125/swin2sr) 找到。


## Swin2SRImageProcessor

[[autodoc]] Swin2SRImageProcessor
    - preprocess

## Swin2SRConfig

[[autodoc]] Swin2SRConfig

## Swin2SRModel

[[autodoc]] Swin2SRModel
    - forward

## Swin2SRForImageSuperResolution

[[autodoc]] Swin2SRForImageSuperResolution
    - forward