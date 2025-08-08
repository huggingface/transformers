<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# FLAVA

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The FLAVA model was proposed in [FLAVA: A Foundational Language And Vision Alignment Model](https://huggingface.co/papers/2112.04482) by Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela and is accepted at CVPR 2022.

The paper aims at creating a single unified foundation model which can work across vision, language
as well as vision-and-language multimodal tasks.

The abstract from the paper is the following:

*State-of-the-art vision and vision-and-language models rely on large-scale visio-linguistic pretraining for obtaining good performance on a variety
of downstream tasks. Generally, such models are often either cross-modal (contrastive) or multi-modal
(with earlier fusion) but not both; and they often only target specific modalities or tasks. A promising
direction would be to use a single holistic universal model, as a "foundation", that targets all modalities
at once -- a true vision and language foundation model should be good at vision tasks, language tasks, and
cross- and multi-modal vision and language tasks. We introduce FLAVA as such a model and demonstrate
impressive performance on a wide range of 35 tasks spanning these target modalities.*

This model was contributed by [aps](https://huggingface.co/aps). The original code can be found [here](https://github.com/facebookresearch/multimodal/tree/main/examples/flava).

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

## FlavaImageProcessorFast

[[autodoc]] FlavaImageProcessorFast
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
