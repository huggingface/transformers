<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Transformer XL

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=transfo-xl">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-transfo--xl-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/transfo-xl-wt103">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The Transformer-XL model was proposed in [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) by Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan
Salakhutdinov. It's a causal (uni-directional) transformer with relative positioning (sinusoïdal) embeddings which can
reuse previously computed hidden-states to attend to longer context (memory). This model also uses adaptive softmax
inputs and outputs (tied).

The abstract from the paper is the following:

*Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the
setting of language modeling. We propose a novel neural architecture Transformer-XL that enables learning dependency
beyond a fixed length without disrupting temporal coherence. It consists of a segment-level recurrence mechanism and a
novel positional encoding scheme. Our method not only enables capturing longer-term dependency, but also resolves the
context fragmentation problem. As a result, Transformer-XL learns dependency that is 80% longer than RNNs and 450%
longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1,800+
times faster than vanilla Transformers during evaluation. Notably, we improve the state-of-the-art results of
bpc/perplexity to 0.99 on enwiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5 on Penn
Treebank (without finetuning). When trained only on WikiText-103, Transformer-XL manages to generate reasonably
coherent, novel text articles with thousands of tokens.*

Tips:

- Transformer-XL uses relative sinusoidal positional embeddings. Padding can be done on the left or on the right. The
  original implementation trains on SQuAD with padding on the left, therefore the padding defaults are set to left.
- Transformer-XL is one of the few models that has no sequence length limit.
- Same as a regular GPT model, but introduces a recurrence mechanism for two consecutive segments (similar to a regular RNNs with two consecutive inputs). In this context, a segment is a number of consecutive tokens (for instance 512) that may span across multiple documents, and segments are fed in order to the model.
- Basically, the hidden states of the previous segment are concatenated to the current input to compute the attention scores. This allows the model to pay attention to information that was in the previous segment as well as the current one. By stacking multiple attention layers, the receptive field can be increased to multiple previous segments.
- This changes the positional embeddings to positional relative embeddings (as the regular positional embeddings would give the same results in the current input and the current hidden state at a given position) and needs to make some adjustments in the way attention scores are computed.

This model was contributed by [thomwolf](https://huggingface.co/thomwolf). The original code can be found [here](https://github.com/kimiyoung/transformer-xl).

<Tip warning={true}>

TransformerXL does **not** work with *torch.nn.DataParallel* due to a bug in PyTorch, see [issue #36035](https://github.com/pytorch/pytorch/issues/36035)

</Tip>

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Causal language modeling task guide](../tasks/language_modeling)

## TransfoXLConfig

[[autodoc]] TransfoXLConfig

## TransfoXLTokenizer

[[autodoc]] TransfoXLTokenizer
    - save_vocabulary

## TransfoXL specific outputs

[[autodoc]] models.transfo_xl.modeling_transfo_xl.TransfoXLModelOutput

[[autodoc]] models.transfo_xl.modeling_transfo_xl.TransfoXLLMHeadModelOutput

[[autodoc]] models.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLModelOutput

[[autodoc]] models.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLLMHeadModelOutput

## TransfoXLModel

[[autodoc]] TransfoXLModel
    - forward

## TransfoXLLMHeadModel

[[autodoc]] TransfoXLLMHeadModel
    - forward

## TransfoXLForSequenceClassification

[[autodoc]] TransfoXLForSequenceClassification
    - forward

## TFTransfoXLModel

[[autodoc]] TFTransfoXLModel
    - call

## TFTransfoXLLMHeadModel

[[autodoc]] TFTransfoXLLMHeadModel
    - call

## TFTransfoXLForSequenceClassification

[[autodoc]] TFTransfoXLForSequenceClassification
    - call

## Internal Layers

[[autodoc]] AdaptiveEmbedding

[[autodoc]] TFAdaptiveEmbedding
