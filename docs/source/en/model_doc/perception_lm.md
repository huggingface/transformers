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
*This model was released on 2025-04-17 and added to Hugging Face Transformers on 2025-07-11.*

# PerceptionLM

## Overview

The [PerceptionLM](https://huggingface.co/papers/2504.13180) model was proposed in [PerceptionLM: Open-Access Data and Models for Detailed Visual Understanding](https://ai.meta.com/research/publications/perceptionlm-open-access-data-and-models-for-detailed-visual-understanding/) by Jang Hyun Cho et al. It's a fully open, reproducible model for transparent research in image and video understanding. PLM consists of
a vision encoder with a small scale (<8B parameters) LLM decoder.

The abstract from the paper is the following:

*Vision-language models are integral to computer vision research, yet many high-performing models
remain closed-source, obscuring their data, design and training recipe. The research community
has responded by using distillation from black-box models to label training data, achieving strong
benchmark results, at the cost of measurable scientific progress. However, without knowing the details
of the teacher model and its data sources, scientific progress remains difficult to measure. In this
paper, we study building a Perception Language Model (PLM) in a fully open and reproducible
framework for transparent research in image and video understanding. We analyze standard training
pipelines without distillation from proprietary models and explore large-scale synthetic data to identify
critical data gaps, particularly in detailed video understanding. To bridge these gaps, we release 2.8M
human-labeled instances of fine-grained video question-answer pairs and spatio-temporally grounded
video captions. Additionally, we introduce PLM–VideoBench, a suite for evaluating challenging video
understanding tasks focusing on the ability to reason about “what”, “where”, “when”, and “how” of a
video. We make our work fully reproducible by providing data, training recipes, code & models.*


This model was contributed by [shumingh](https://huggingface.co/shumingh).
The original code can be found [here](https://github.com/facebookresearch/perception_models).


## PerceptionLMConfig

[[autodoc]] PerceptionLMConfig

## PerceptionLMProcessor

[[autodoc]] PerceptionLMProcessor

## PerceptionLMImageProcessorFast

[[autodoc]] PerceptionLMImageProcessorFast

## PerceptionLMVideoProcessor

[[autodoc]] PerceptionLMVideoProcessor

## PerceptionLMModel

[[autodoc]] PerceptionLMModel

## PerceptionLMForConditionalGeneration

[[autodoc]] PerceptionLMForConditionalGeneration
    - forward
