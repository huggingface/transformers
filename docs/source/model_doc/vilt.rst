<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ViLT

## Overview

The VilT model was proposed in [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)
by Wonjae Kim, Bokyung Son, Ildoo Kim. ViLT incorporates text embeddings into a Vision Transformer (ViT), allowing it to have a minimal design
for Vision-and-Language Pre-training (VLP).

The abstract from the paper is the following:

*Vision-and-Language Pre-training (VLP) has improved performance on various joint vision-and-language downstream tasks.
Current approaches to VLP heavily rely on image feature extraction processes, most of which involve region supervision
(e.g., object detection) and the convolutional architecture (e.g., ResNet). Although disregarded in the literature, we
find it problematic in terms of both (1) efficiency/speed, that simply extracting input features requires much more
computation than the multimodal interaction steps; and (2) expressive power, as it is upper bounded to the expressive
power of the visual embedder and its predefined visual vocabulary. In this paper, we present a minimal VLP model,
Vision-and-Language Transformer (ViLT), monolithic in the sense that the processing of visual inputs is drastically
simplified to just the same convolution-free manner that we process textual inputs. We show that ViLT is up to tens of
times faster than previous VLP models, yet with competitive or better downstream task performance.*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/dandelin/ViLT).

## ViltConfig

[[autodoc]] ViltConfig
    - all

## ViltFeatureExtractor

[[autodoc]] ViltFeatureExtractor
    - __call__

## ViltProcessor

[[autodoc]] ViltProcessor
    - __call__

## ViltModel

[[autodoc]] ViltModel
    - forward

## ViltForPreTraining

[[autodoc]] ViltForPreTraining
    - forward

## ViltForMaskedLM

[[autodoc]] ViltForMaskedLM
    - forward

## ViltForVisualQuestionAnswering

[[autodoc]] ViltForVisualQuestionAnswering
    - forward

## ViltForNaturalLanguageVisualReasoning

[[autodoc]] ViltForNaturalLanguageVisualReasoning
    - forward

## ViltForImageRetrievalTextRetrieval

[[autodoc]] ViltForImageRetrievalTextRetrieval
    - forward
