<!--版权所有 2023 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件被分发在“按原样”基础上，不提供任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️ 请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确呈现。
-->
# BLIP

## 概述

BLIP 模型是由 Junnan Li，Dongxu Li，Caiming Xiong 和 Steven Hoi 在 [《BLIP：引导语言-图像预训练实现统一的视觉-语言理解与生成》](https://arxiv.org/abs/2201.12086) 提出的。

BLIP 是一种能够执行各种多模态任务的模型，包括- 视觉问答- 图像-文本检索（图像-文本匹配）- 图像字幕生成
论文中的摘要如下：

*视觉-语言预训练（VLP）已经提高了许多视觉-语言任务的性能。然而，大多数现有的预训练模型只在理解类任务或生成类任务中表现出色。此外，通过扩大数据集规模并使用从网络收集的带有噪声的图像-文本配对来实现性能改进，而这是次优的监督来源。在本文中，我们提出了 BLIP，一种新的 VLP 框架，可以灵活地转移到视觉-语言理解和生成任务。BLIP 通过引导标题有效地利用了嘈杂的网络数据，其中一个标题生成器生成合成标题，然后通过一个过滤器删除噪声标题。我们在各种视觉-语言任务中取得了最先进的成果，例如图像-文本检索（平均召回率@1 提高 2.7%）、图像字幕生成（CIDEr 提高 2.8%）和视觉问答（VQA 得分提高 1.6%）。BLIP 在直接转移到零样本视觉-语言任务方面也表现出了强大的泛化能力。代码、模型和数据集已经发布。*
![BLIP.gif](https://s3.amazonaws.com/moonup/production/uploads/1670928184033-62441d1d9fdefb55a0b7d12c.gif)
此模型由 [ybelkada](https://huggingface.co/ybelkada) 贡献。原始代码可以在 [这里](https://github.com/salesforce/BLIP) 找到。

## 资源

- [Jupyter 笔记本](https://github.com/huggingface/notebooks/blob/main/examples/image_captioning_blip.ipynb)，介绍如何在自定义数据集上对 BLIP 进行微调

## BlipConfig

[[autodoc]] BlipConfig
    - from_text_vision_configs
## BlipTextConfig
[[autodoc]] BlipTextConfig
## BlipVisionConfig
[[autodoc]] BlipVisionConfig
## BlipProcessor
[[autodoc]] BlipProcessor

## BlipImageProcessor

[[autodoc]] BlipImageProcessor
    - preprocess

## BlipModel

[[autodoc]] BlipModel
    - forward
    - get_text_features
    - get_image_features

## BlipTextModel

[[autodoc]] BlipTextModel
    - forward


## BlipVisionModel

[[autodoc]] BlipVisionModel
    - forward


## BlipForConditionalGeneration

[[autodoc]] BlipForConditionalGeneration
    - forward


## BlipForImageTextRetrieval

[[autodoc]] BlipForImageTextRetrieval
    - forward


## BlipForQuestionAnswering

[[autodoc]] BlipForQuestionAnswering
    - forward

## TFBlipModel

[[autodoc]] TFBlipModel
    - call
    - get_text_features
    - get_image_features

## TFBlipTextModel

[[autodoc]] TFBlipTextModel
    - call


## TFBlipVisionModel

[[autodoc]] TFBlipVisionModel
    - call


## TFBlipForConditionalGeneration

[[autodoc]] TFBlipForConditionalGeneration
    - call


## TFBlipForImageTextRetrieval

[[autodoc]] TFBlipForImageTextRetrieval
    - call


## TFBlipForQuestionAnswering

[[autodoc]] TFBlipForQuestionAnswering
    - call