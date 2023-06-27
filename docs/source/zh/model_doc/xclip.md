<!--版权所有2022年The HuggingFace团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本：
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发，不附带任何形式的任何保证或条件。有关许可证的详细信息以及许可证下的特定语言权限和限制，请参阅许可证。
⚠️请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确显示。
-->
# X-CLIP

## 概述

X-CLIP 模型是由 Bolin Ni、Houwen Peng、Minghao Chen、Songyang Zhang、Gaofeng Meng、Jianlong Fu、Shiming Xiang、Haibin Ling 在《扩展语言-图像预训练模型用于通用视频识别》中提出的（https://arxiv.org/abs/2208.02816）。

X-CLIP 是 [CLIP](clip) 在视频领域的最简扩展。该模型由文本编码器、跨帧视觉编码器、多帧集成 Transformer 和视频特定提示生成器组成。

该论文的摘要如下：

*对比语言-图像预训练已经在从 Web 规模数据中学习视觉-文本联合表示方面取得了巨大成功，展示了在各种图像任务中的显著“零样本”泛化能力。然而，如何有效地将这些新的语言-图像预训练方法扩展到视频领域仍然是一个未解决的问题。在这项工作中，我们提出了一种简单而有效的方法，直接将预训练的语言-图像模型转换为视频识别，而不是从头开始预训练新模型。更具体地说，为了捕捉帧在时间维度上的长程依赖关系，我们提出了一个跨帧注意力机制，该机制明确地在帧之间交换信息。这种模块轻量级且可以无缝地插入预训练的语言-图像模型。此外，我们提出了一种视频特定的提示方案，利用视频内容信息生成有区别的文本提示。大量实验证明我们的方法是有效的，并且可以推广到不同的视频识别场景。特别是在完全监督的设置下，我们的方法在 Kinectics-400 上达到了 87.1%的 top-1 准确率，与 Swin-L 和 ViViT-H 相比，FLOPs 减少了 12 倍。在零样本实验中，我们的方法在两个流行协议下的 top-1 准确率分别超过当前最先进的方法+7.6%和+14.9%。在少样本场景中，我们的方法在标记数据极为有限时超过了以前最佳方法+32.1%和+23.1%。*

提示：

- 使用 X-CLIP 与 [CLIP](clip) 的用法完全相同。
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/xclip_architecture.png"alt="drawing" width="600"/> 
<small> X-CLIP 架构。摘自 <a href="https://arxiv.org/abs/2208.02816"> 原始论文。</a> </small>

此模型由 [nielsr](https://huggingface.co/nielsr) 贡献。原始代码可在 [此处](https://github.com/microsoft/VideoX/tree/master/X-CLIP) 找到。

## 资源

以下是官方 Hugging Face 和社区（由🌎表示）的资源列表，可帮助您开始使用 X-CLIP。

- X-CLIP 的演示笔记本可以在 [此处](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/X-CLIP) 找到。

如果您有兴趣提交资源以包含在此处，请随时打开拉取请求，我们将进行审查！资源应该展示出新的东西，而不是重复现有的资源。
## XCLIPProcessor

[[autodoc]] XCLIPProcessor

## XCLIPConfig

[[autodoc]] XCLIPConfig
    - from_text_vision_configs

## XCLIPTextConfig

[[autodoc]] XCLIPTextConfig

## XCLIPVisionConfig

[[autodoc]] XCLIPVisionConfig

## XCLIPModel

[[autodoc]] XCLIPModel
    - forward
    - get_text_features
    - get_video_features

## XCLIPTextModel

[[autodoc]] XCLIPTextModel
    - forward

## XCLIPVisionModel

[[autodoc]] XCLIPVisionModel
    - forward
