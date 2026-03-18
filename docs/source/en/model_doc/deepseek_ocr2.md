<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DeepSeek-OCR-2


## Overview

The DeepSeek-OCR-2 model was proposed in [Visual Causal Flow: A Novel Approach to OCR-Specialized Vision-Language Models](https://arxiv.org/abs/2601.20552) by the DeepSeek team.

DeepSeek-OCR-2 is an OCR-specialized vision-language model built on a distinctive architecture: a SAM ViT-B vision encoder feeds into a Qwen2 hybrid attention encoder, which is connected through an MLP projector to a DeepSeek-V2 Mixture-of-Experts (MoE) language model. A key feature of the model is its hybrid attention mechanism, which applies bidirectional attention over image tokens and causal attention over query tokens, enabling efficient and accurate document understanding.

## Usage example

```python

```

## DeepseekOcr2Config

[[autodoc]] DeepseekOcr2Config

## DeepseekOcr2ImageProcessor

[[autodoc]] DeepseekOcr2ImageProcessor

## DeepseekOcr2ImageProcessorFast

[[autodoc]] DeepseekOcr2ImageProcessorFast

## DeepseekOcr2Processor

[[autodoc]] DeepseekOcr2Processor

## DeepseekOcr2Model

[[autodoc]] DeepseekOcr2Model

## DeepseekOcr2ForConditionalGeneration

[[autodoc]] DeepseekOcr2ForConditionalGeneration
