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

# Mamba

## Overview

The Mamba model was proposed in [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) by Albert Gu and Tri Dao.

This model is a new paradigm architecture based on `state-space-models`. You can read more about the intuition behind these [here](https://srush.github.io/annotated-s4/).

The abstract from the paper is the following:

*Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba). Mamba enjoys fast inference (5× higher throughput than Transformers) and linear scaling in sequence length, and its performance improves on real data up to million-length sequences. As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics. On language modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation.*

Tips:

- Mamba is a new `state space model` architecture that rivals the classic Transformers. It is based on the line of progress on structured state space models, with an efficient hardware-aware design and implementation in the spirit of [FlashAttention](https://github.com/Dao-AILab/flash-attention).
- Mamba stacks `mixer` layers, which are the equivalent of `Attention` layers. The core logic of `mamba` is held in the `MambaMixer` class.
- Two implementation cohabit: one is optimized and uses fast cuda kernels, while the other one is naive but can run on any device!
- The current implementation leverages the original cuda kernels: the equivalent of flash attention for Mamba are hosted in the [`mamba-ssm`](https://github.com/state-spaces/mamba) and the [`causal_conv1d`](https://github.com/Dao-AILab/causal-conv1d) repositories. Make sure to install them if your hardware supports them!
- Contributions to make the naive path faster are welcome 🤗

This model was contributed by [ArthurZ](https://huggingface.co/ArthurZ).
The original code can be found [here](https://github.com/state-spaces/mamba).


## MambaConfig

[[autodoc]] MambaConfig

## MambaModel

[[autodoc]] MambaModel
    - forward

## MambaLMHeadModel

[[autodoc]] MambaForCausalLM
    - forward
