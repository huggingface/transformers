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

# chameleon

## Overview

The Chameleon model was proposed in [Chameleon: Mixed-Modal Early-Fusion Foundation Models
](https://arxiv.org/abs/2405.09818v1) by META AI Chameleon Team. Chameleon is a Vision-Language Model that use vector quantization to tokenize images which enanles the model to generate multimodal output. Additionally the model can generate from an interleaved vision-text input. 


The abstract from the paper is the following:

We present Chameleon, a family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence. We outline a stable training
approach from inception, an alignment recipe, and an architectural parameterization tailored for the
early-fusion, token-based, mixed-modal setting. The models are evaluated on a comprehensive range
of tasks, including visual question answering, image captioning, text generation, image generation, and
long-form mixed modal generation. Chameleon demonstrates broad and general capabilities, including
state-of-the-art performance in image captioning tasks, outperforms Llama-2 in text-only tasks while
being competitive with models such as Mixtral 8x7B and Gemini-Pro, and performs non-trivial image
generation, all in a single model. It also matches or exceeds the performance of much larger models,
including Gemini Pro and GPT-4V, according to human judgments on a new long-form mixed-modal
generation evaluation, where either the prompt or outputs contain mixed sequences of both images and
text. Chameleon marks a significant step forward in a unified modeling of full multimodal documents


This model was contributed by [joaogante](https://huggingface.co/joaogante) and [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).
The original code can be found [here](https://github.com/facebookresearch/chameleon).


## ChameleonConfig

[[autodoc]] ChameleonConfig

## ChameleonVQConfig

[[autodoc]] ChameleonVQConfig

## ChameleonProcessor

[[autodoc]] ChameleonProcessor

## ChameleonImageProcessor

[[autodoc]] ChameleonImageProcessor
    - preprocess

## ChameleonModel

[[autodoc]] ChameleonModel
    - forward

## ChameleonForCausalLM

[[autodoc]] ChameleonForCausalLM
    - forward

## ChameleonForSequenceClassification

[[autodoc]] ChameleonForSequenceClassification
    - forward

## ChameleonForQuestionAnswering

[[autodoc]] ChameleonForQuestionAnswering
    - forward
