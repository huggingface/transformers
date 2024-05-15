<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# FalconVlm

## Overview

The FalconVlm model was proposed from [Technology Innovation Institute](https://www.tii.ae/).


The abstract from the paper is the following:

The Falcon2-11B VLM is a vision-language model (VLM) for additionally handling image inputs and answering the queries corresponding to the images. To achieve this, we integrate the pretrained CLIP ViT-L/14 vision encoder with our Falcon2-11B chat-finetuned model and train with image-text data. For enhancing the VLM's perception of fine-grained details w.r.t small objects in images, we employ a dynamic encoding mechanism at high-resolution for image inputs, similar to LLaVA-Next.

## FalconVlmConfig

[[autodoc]] FalconVlmConfig

## FalconImageProcessor

[[autodoc]] FalconImageProcessor
    - preprocess

## FalconVLProcessor

[[autodoc]] FalconVLProcessor

## FalconVlmForConditionalGeneration

[[autodoc]] FalconVlmForConditionalGeneration
    - forward
