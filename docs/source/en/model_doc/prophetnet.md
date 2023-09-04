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

# ProphetNet

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=prophetnet">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-prophetnet-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/prophetnet-large-uncased">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>


**DISCLAIMER:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) and assign
@patrickvonplaten

## Overview

The ProphetNet model was proposed in [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training,](https://arxiv.org/abs/2001.04063) by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei
Zhang, Ming Zhou on 13 Jan, 2020.

ProphetNet is an encoder-decoder model and can predict n-future tokens for "ngram" language modeling instead of just
the next token.

The abstract from the paper is the following:

*In this paper, we present a new sequence-to-sequence pretraining model called ProphetNet, which introduces a novel
self-supervised objective named future n-gram prediction and the proposed n-stream self-attention mechanism. Instead of
the optimization of one-step ahead prediction in traditional sequence-to-sequence model, the ProphetNet is optimized by
n-step ahead prediction which predicts the next n tokens simultaneously based on previous context tokens at each time
step. The future n-gram prediction explicitly encourages the model to plan for the future tokens and prevent
overfitting on strong local correlations. We pre-train ProphetNet using a base scale dataset (16GB) and a large scale
dataset (160GB) respectively. Then we conduct experiments on CNN/DailyMail, Gigaword, and SQuAD 1.1 benchmarks for
abstractive summarization and question generation tasks. Experimental results show that ProphetNet achieves new
state-of-the-art results on all these datasets compared to the models using the same scale pretraining corpus.*

Tips:

- ProphetNet is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.
- The model architecture is based on the original Transformer, but replaces the “standard” self-attention mechanism in the decoder by a a main self-attention mechanism and a self and n-stream (predict) self-attention mechanism.

The Authors' code can be found [here](https://github.com/microsoft/ProphetNet).

## Documentation resources

- [Causal language modeling task guide](../tasks/language_modeling)
- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## ProphetNetConfig

[[autodoc]] ProphetNetConfig

## ProphetNetTokenizer

[[autodoc]] ProphetNetTokenizer

## ProphetNet specific outputs

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput

[[autodoc]] models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput

## ProphetNetModel

[[autodoc]] ProphetNetModel
    - forward

## ProphetNetEncoder

[[autodoc]] ProphetNetEncoder
    - forward

## ProphetNetDecoder

[[autodoc]] ProphetNetDecoder
    - forward

## ProphetNetForConditionalGeneration

[[autodoc]] ProphetNetForConditionalGeneration
    - forward

## ProphetNetForCausalLM

[[autodoc]] ProphetNetForCausalLM
    - forward
