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

# MEGA

## Overview

The MEGA model was proposed in [Mega: Moving Average Equipped Gated Attention](https://arxiv.org/abs/2209.10655) by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, and Luke Zettlemoyer.
MEGA proposes a new approach to self-attention with each encoder layer having a multi-headed exponential moving average in addition to a single head of standard dot-product attention, giving the attention mechanism 
stronger positional biases. This allows MEGA to perform competitively to Transformers on standard benchmarks including LRA 
while also having significantly fewer parameters. MEGA's compute efficiency allows it to scale to very long sequences, making it an 
attractive option for long-document NLP tasks.

The abstract from the paper is the following:

 *The design choices in the Transformer attention mechanism, including weak inductive bias and quadratic computational complexity, have limited its application for modeling long sequences. In this paper, we introduce Mega, a simple, theoretically grounded, single-head gated attention mechanism equipped with (exponential) moving average to incorporate inductive bias of position-aware local dependencies into the position-agnostic attention mechanism. We further propose a variant of Mega that offers linear time and space complexity yet yields only minimal quality loss, by efficiently splitting the whole sequence into multiple chunks with fixed length. Extensive experiments on a wide range of sequence modeling benchmarks, including the Long Range Arena, neural machine translation, auto-regressive language modeling, and image and speech classification, show that Mega achieves significant improvements over other sequence models, including variants of Transformers and recent state space models. *

Tips:

- MEGA can perform quite well with relatively few parameters. See Appendix D in the MEGA paper for examples of architectural specs which perform well in various settings. If using MEGA as a decoder, be sure to set `bidirectional=False` to avoid errors with default bidirectional. 
- Mega-chunk is a variant of mega that reduces time and spaces complexity from quadratic to linear. Utilize chunking with MegaConfig.use_chunking and control chunk size with MegaConfig.chunk_size 

This model was contributed by [mnaylor](https://huggingface.co/mnaylor).
The original code can be found [here](https://github.com/facebookresearch/mega).

Implementation Notes:

- The original implementation of MEGA had an inconsistent expectation of attention masks for padding and causal self-attention between the softmax attention and Laplace/squared ReLU method. This implementation addresses that inconsistency.
- The original implementation did not include token type embeddings; this implementation adds support for these, with the option controlled by MegaConfig.add_token_type_embeddings


## MegaConfig

[[autodoc]] MegaConfig

## MegaModel

[[autodoc]] MegaModel
    - forward

## MegaForCausalLM

[[autodoc]] MegaForCausalLM
    - forward

## MegaForMaskedLM

[[autodoc]] MegaForMaskedLM
    - forward

## MegaForSequenceClassification

[[autodoc]] MegaForSequenceClassification
    - forward

## MegaForMultipleChoice

[[autodoc]] MegaForMultipleChoice
    - forward

## MegaForTokenClassification

[[autodoc]] MegaForTokenClassification
    - forward

## MegaForQuestionAnswering

[[autodoc]] MegaForQuestionAnswering
    - forward
