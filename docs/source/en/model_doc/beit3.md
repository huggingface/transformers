<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# BEiT-3

## Overview

The BEiT-3 model was proposed in [Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language
Tasks](https://arxiv.org/abs/2208.10442) by Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu,
Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, Furu Wei.

The abstract from the paper is the following:

*A big convergence of language, vision, and multimodal pretraining is emerging. In this work, we introduce a
general-purpose multimodal foundation model BEiT-3, which achieves state-of-the-art transfer performance on both vision
and vision-language tasks. Specifically, we advance the big convergence from three aspects: backbone architecture,
pretraining task, and model scaling up. We introduce Multiway Transformers for general-purpose modeling, where the
modular architecture enables both deep fusion and modality-specific encoding. Based on the shared backbone, we perform
masked "language" modeling on images (Imglish), texts (English), and image-text pairs ("parallel sentences") in a
unified manner. Experimental results show that BEiT-3 obtains state-of-the-art performance on object detection (COCO),
image classification (ImageNet), visual reasoning (NLVR2), visual question answering
(VQAv2), image captioning (COCO), and cross-modal retrieval (Flickr30K, COCO).*

This model was contributed by [Raghavan](https://huggingface.co/Raghavan).
The original code can be found [here](https://github.com/microsoft/unilm/tree/master/beit3).

## BEiT3 specific outputs

[[autodoc]] models.beit3.modeling_beit3.Biet3ImageTextMatchingModelOutput

## Beit3Config

[[autodoc]] Beit3Config

## Beit3Processor

[[autodoc]] Beit3Processor

## Beit3ImageProcessor

[[autodoc]] Beit3ImageProcessor
    - preprocess

## Beit3Model

[[autodoc]] Beit3Model
    - forward

## Beit3ForCaptioning

[[autodoc]] Beit3ForCaptioning
    - forward

## Beit3ForImageClassification

[[autodoc]] Beit3ForImageClassification
    - forward

## Beit3ForImageTextRetrieval

[[autodoc]] Beit3ForImageTextRetrieval
    - forward

## Beit3ForVisualQuestionAnswering

[[autodoc]] Beit3ForVisualQuestionAnswering
    - forward

## Beit3ForVisualReasoning

[[autodoc]] Beit3ForVisualReasoning
    - forward



