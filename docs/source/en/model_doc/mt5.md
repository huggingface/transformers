<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# mT5

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=mt5">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-mt5-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/mt5-small-finetuned-arxiv-cs-finetuned-arxiv-cs-full">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The mT5 model was presented in [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934) by Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya
Siddhant, Aditya Barua, Colin Raffel.

The abstract from the paper is the following:

*The recent "Text-to-Text Transfer Transformer" (T5) leveraged a unified text-to-text format and scale to attain
state-of-the-art results on a wide variety of English-language NLP tasks. In this paper, we introduce mT5, a
multilingual variant of T5 that was pre-trained on a new Common Crawl-based dataset covering 101 languages. We detail
the design and modified training of mT5 and demonstrate its state-of-the-art performance on many multilingual
benchmarks. We also describe a simple technique to prevent "accidental translation" in the zero-shot setting, where a
generative model chooses to (partially) translate its prediction into the wrong language. All of the code and model
checkpoints used in this work are publicly available.*

Note: mT5 was only pre-trained on [mC4](https://huggingface.co/datasets/mc4) excluding any supervised training.
Therefore, this model has to be fine-tuned before it is usable on a downstream task, unlike the original T5 model.
Since mT5 was pre-trained unsupervisedly, there's no real advantage to using a task prefix during single-task
fine-tuning. If you are doing multi-task fine-tuning, you should use a prefix.

Google has released the following variants:

- [google/mt5-small](https://huggingface.co/google/mt5-small)

- [google/mt5-base](https://huggingface.co/google/mt5-base)

- [google/mt5-large](https://huggingface.co/google/mt5-large)

- [google/mt5-xl](https://huggingface.co/google/mt5-xl)

- [google/mt5-xxl](https://huggingface.co/google/mt5-xxl).

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The original code can be
found [here](https://github.com/google-research/multilingual-t5).

## Resources

- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## MT5Config

[[autodoc]] MT5Config

## MT5Tokenizer

[[autodoc]] MT5Tokenizer

See [`T5Tokenizer`] for all details.


## MT5TokenizerFast

[[autodoc]] MT5TokenizerFast

See [`T5TokenizerFast`] for all details.

<frameworkcontent>
<pt>

## MT5Model

[[autodoc]] MT5Model

## MT5ForConditionalGeneration

[[autodoc]] MT5ForConditionalGeneration

## MT5EncoderModel

[[autodoc]] MT5EncoderModel

## MT5ForSequenceClassification

[[autodoc]] MT5ForSequenceClassification

## MT5ForTokenClassification

[[autodoc]] MT5ForTokenClassification

## MT5ForQuestionAnswering

[[autodoc]] MT5ForQuestionAnswering

</pt>
<tf>

## TFMT5Model

[[autodoc]] TFMT5Model

## TFMT5ForConditionalGeneration

[[autodoc]] TFMT5ForConditionalGeneration

## TFMT5EncoderModel

[[autodoc]] TFMT5EncoderModel

</tf>
<jax>

## FlaxMT5Model

[[autodoc]] FlaxMT5Model

## FlaxMT5ForConditionalGeneration

[[autodoc]] FlaxMT5ForConditionalGeneration

## FlaxMT5EncoderModel

[[autodoc]] FlaxMT5EncoderModel

</jax>
</frameworkcontent>
