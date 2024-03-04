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

# Gemma

## Overview

The Gemma model was proposed in [Gemma: Open Models Based on Gemini Technology and Research](https://blog.google/technology/developers/gemma-open-models/) by Gemma Team, Google.
Gemma models are trained on 6T tokens, and released with 2 versions, 2b and 7b.

The abstract from the paper is the following:

*This work introduces Gemma, a new family of open language models demonstrating strong performance across academic benchmarks for language understanding, reasoning, and safety. We release two sizes of models (2 billion and 7 billion parameters), and provide both pretrained and fine-tuned checkpoints. Gemma outperforms similarly sized open models on 11 out of 18 text-based tasks, and we present comprehensive evaluations of safety and responsibility aspects of the models, alongside a detailed description of our model development. We believe the responsible release of LLMs is critical for improving the safety of frontier models, and for enabling the next wave of LLM innovations*

Tips:

- The original checkpoints can be converted using the conversion script `src/transformers/models/gemma/convert_gemma_weights_to_hf.py` 

This model was contributed by [Arthur Zucker](https://huggingface.co/ArthurZ), [Younes Belkada](https://huggingface.co/ybelkada), [Sanchit Gandhi](https://huggingface.co/sanchit-gandhi), [Pedro Cuenca](https://huggingface.co/pcuenq).


## GemmaConfig

[[autodoc]] GemmaConfig

## GemmaTokenizer

[[autodoc]] GemmaTokenizer


## GemmaTokenizerFast

[[autodoc]] GemmaTokenizerFast

## GemmaModel

[[autodoc]] GemmaModel
    - forward

## GemmaForCausalLM

[[autodoc]] GemmaForCausalLM
    - forward

## GemmaForSequenceClassification

[[autodoc]] GemmaForSequenceClassification
    - forward

## FlaxGemmaModel

[[autodoc]] FlaxGemmaModel
    - __call__

## FlaxGemmaForCausalLM

[[autodoc]] FlaxGemmaForCausalLM
    - __call__
