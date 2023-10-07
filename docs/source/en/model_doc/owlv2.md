<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# OWLv2

## Overview

OWLv2 was proposed in [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683)
by Matthias Minderer, Alexey Gritsenko, Neil Houlsby. OWLv2 scales up [OWL-ViT](owlvit) using self-training, which uses an existing detector to generate pseudo-box annotations on image-text pairs. This results in large gains over the previous state-of-the-art for zero-shot object detection.

The abstract from the paper is the following:

*Open-vocabulary object detection has benefited greatly from pretrained vision-language models, but is still limited by the amount of available detection training data. While detection training data can be expanded by using Web image-text pairs as weak supervision, this has not been done at scales comparable to image-level pretraining. Here, we scale up detection data with self-training, which uses an existing detector to generate pseudo-box annotations on image-text pairs. Major challenges in scaling self-training are the choice of label space, pseudo-annotation filtering, and training efficiency. We present the OWLv2 model and OWL-ST self-training recipe, which address these challenges. OWLv2 surpasses the performance of previous state-of-the-art open-vocabulary detectors already at comparable training scales (~10M examples). However, with OWL-ST, we can scale to over 1B examples, yielding further large improvement: With an L/14 architecture, OWL-ST improves AP on LVIS rare classes, for which the model has seen no human box annotations, from 31.2% to 44.6% (43% relative improvement). OWL-ST unlocks Web-scale training for open-world localization, similar to what has been seen for image classification and language modelling.*

Tips:

- Usage of OWLv2 is identical to [OWL-ViT](owlvit) with a new, updated image processor.

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit).


## Owlv2Config

[[autodoc]] Owlv2Config
    - from_text_vision_configs

## Owlv2TextConfig

[[autodoc]] Owlv2TextConfig

## Owlv2VisionConfig

[[autodoc]] Owlv2VisionConfig

## Owlv2ImageProcessor

[[autodoc]] Owlv2ImageProcessor
    - preprocess
    - post_process_object_detection
    - post_process_image_guided_detection

## Owlv2Processor

[[autodoc]] Owlv2Processor

## Owlv2Model

[[autodoc]] Owlv2Model
    - forward
    - get_text_features
    - get_image_features

## Owlv2TextModel

[[autodoc]] Owlv2TextModel
    - forward

## Owlv2VisionModel

[[autodoc]] Owlv2VisionModel
    - forward

## Owlv2ForObjectDetection

[[autodoc]] Owlv2ForObjectDetection
    - forward
    - image_guided_detection
