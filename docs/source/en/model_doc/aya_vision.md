<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# AyaVision

## Overview

The Aya Vision 8B model is a state-of-the-art multilingual multimodal model developed by Cohere For AI. It builds on the Aya Expanse 8B recipe to handle both visual and textual information without compromising on the strong multilingual textual performance of the original model.

Aya Vision combines the `Siglip2-so400-384-14` vision encoder with the Cohere CommandR-7B language model further post-trained with the Aya Expanse recipe, creating a powerful vision-language model capable of understanding images and generating text across 23 languages.

Key features of Aya Vision include:
- Multimodal capabilities in 23 languages
- Strong text-only multilingual capabilities inherited from CommandR-7B post-trained with the Aya Expanse recipe
- High-quality visual understanding using the Siglip2-so400-384-14 vision encoder
- Seamless integration of visual and textual information

<!-- <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/aya_vision_architecture.webp"
alt="drawing" width="600"/>

<small> Aya Vision architecture. </small> -->

Tips:

- Aya Vision is a multimodal model that takes images and text as input and produces text as output.
- Images are represented using the `<image>` tag in the templated input.
- For best results, use the `apply_chat_template` method of the processor to format your inputs correctly.
- The model can process multiple images in a single conversation.
- Aya Vision can understand and generate text in 23 languages, making it suitable for multilingual multimodal applications.

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [saurabhdash](https://huggingface.co/saurabhdash) and [yonigozlan](https://huggingface.co/yonigozlan).


## Usage

Here's how to use Aya Vision for inference:

<!-- todo: add usage examples, see tests for examples with apply_chat_template -->

## How to use

<INSERT Usage examples here>


## AyaVisionProcessor

[[autodoc]] AyaVisionProcessor

## AyaVisionConfig

[[autodoc]] AyaVisionConfig

## AyaVisionForConditionalGeneration

[[autodoc]] AyaVisionForConditionalGeneration
    - forward
