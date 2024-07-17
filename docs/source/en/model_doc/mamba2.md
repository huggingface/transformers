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

# Mamba2

## Overview

The Mamba2 model was proposed in [Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) by Tri Dao and Albert Gu.

This model is a successor to [Mamba](https://arxiv.org/abs/2312.00752) which further improves on `state-space models (SSMs)` by creating a bridge between 
masked attention and state-space-machines under the so-called `Structured State Space Duality`. You can read more about the intuition behind 
these [here](https://tridao.me/blog/2024/mamba2-part1-model/).

The abstract from the paper is the following:

*While Transformers have been the main architecture behind deep learning's success in language modeling, state-space models (SSMs) such as Mamba have recently been shown to match or outperform Transformers at small to medium scale. We show that these families of models are actually quite closely related, and develop a rich framework of theoretical connections between SSMs and variants of attention, connected through various decompositions of a well-studied class of structured semiseparable matrices. Our state space duality (SSD) framework allows us to design a new architecture (Mamba-2) whose core layer is an a refinement of Mamba's selective SSM that is 2-8X faster, while continuing to be competitive with Transformers on language modeling.*

Tips:

- Mamba2 is a new `state-space model` architecture that rivals the classic Transformers.
- Performance is further improved by reformulating the Mamba2 SSM layer with mostly matrix multiplications.
- Mamba2 stacks `Mixer` layers with the improved SSM formulation, which are the equivalent of `Attention` layers. The core logic of `mamba2` is held in the `Mamba2Mixer` class.
- Mamba2 offers additional flexibility by allowing to incorporate optional `Attention` and `MLP` layers.
- Two implementations cohabit: one is optimized and uses fast cuda/triton kernels, while the other one is naive but can run on any device!
- The current implementation leverages the original cuda/triton kernels: the equivalent of flash attention for Mamba2 are hosted in the [`mamba-ssm`](https://github.com/state-spaces/mamba) and the [`causal_conv1d`](https://github.com/Dao-AILab/causal-conv1d) repositories. Make sure to install them if your hardware supports them!

This model was contributed by TODO.
The original code can be found [here](https://github.com/state-spaces/mamba).

## Usage

### Prerequisites
In order to run optimized Mamba2 implementations, you first need to install `mamba-ssm` and `causal-conv1d` with the following versions:
```bash
pip install mamba-ssm>=2.0.4 causal-conv1d>=1.2.0
```

### A simple generation example
```python 
from transformers import Mamba2Config, Mamba2ForCausalLM, AutoTokenizer

model_name = '<path-to-new-ckpts>'
model = Mamba2ForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```

## Mamba2Config

[[autodoc]] Mamba2Config

## Mamba2Model

[[autodoc]] Mamba2Model
- forward

## Mamba2LMHeadModel

[[autodoc]] Mamba2ForCausalLM
- forward

## Mamba2ForSequenceClassification

[[autodoc]] Mamba2ForSequenceClassification
- forward