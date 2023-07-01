<!-- 版权所有2022年HuggingFace团队保留所有权利。
根据Apache许可证第2.0版（“许可证”）授权；除非符合许可证的规定，否则您不得使用此文件您可以在以下网址获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”分发的，不附带任何形式的保证或条件。请查看许可证以获取特定语言下的权限和限制。具体语言的规定。
⚠️请注意，此文件是Markdown格式，但包含特定于我们的文档生成器（类似于MDX）的语法，可能无法在您的Markdown查看器中正确显示。
-->
# FLAVA

## 概览

FLAVA模型是由Amanpreet Singh，Ronghang Hu，Vedanuj Goswami，Guillaume Couairon，Wojciech Galuba，Marcus Rohrbach和Douwe Kiela在CVPR 2022上提出的[FLAVA：基础语言与视觉对齐模型](https://arxiv.org/abs/2112.04482)。

该论文旨在创建一个统一的基础模型，可以在视觉、语言以及视觉-语言多模态任务中工作。论文摘要如下所示：

*当前最先进的视觉和视觉-语言模型依赖于大规模的视觉-语言预训练，以获得在各种下游任务中的良好性能。通常，这些模型通常是跨模态（对比）或多模态（早期融合），但两者都不是；它们通常只针对特定的模态或任务。一个有希望的方向是使用一个单一的综合的通用模型，作为一个“基础”，它同时针对所有模态(with earlier fusion) -一个真正的视觉和语言基础模型应在视觉任务、语言任务以及跨模态和多模态的视觉和语言任务方面表现出色。我们将FLAVA引入作为这样一个模型，并展示在涵盖这些目标模态的广泛任务范围上的出色性能。*

此模型由[aps](https://huggingface.co/aps)贡献。原始代码可在[此处](https://github.com/facebookresearch/multimodal/tree/main/examples/flava)找到。

## FlavaConfig

[[autodoc]] FlavaConfig

## FlavaTextConfig

[[autodoc]] FlavaTextConfig

## FlavaImageConfig

[[autodoc]] FlavaImageConfig

## FlavaMultimodalConfig

[[autodoc]] FlavaMultimodalConfig

## FlavaImageCodebookConfig

[[autodoc]] FlavaImageCodebookConfig

## FlavaProcessor

[[autodoc]] FlavaProcessor

## FlavaFeatureExtractor

[[autodoc]] FlavaFeatureExtractor

## FlavaImageProcessor

[[autodoc]] FlavaImageProcessor
    - preprocess

## FlavaForPreTraining

[[autodoc]] FlavaForPreTraining
    - forward

## FlavaModel

[[autodoc]] FlavaModel
    - forward
    - get_text_features
    - get_image_features

## FlavaImageCodebook

[[autodoc]] FlavaImageCodebook
    - forward
    - get_codebook_indices
    - get_codebook_probs

## FlavaTextModel

[[autodoc]] FlavaTextModel
    - forward

## FlavaImageModel

[[autodoc]] FlavaImageModel
    - forward

## FlavaMultimodalModel

[[autodoc]] FlavaMultimodalModel
    - forward
