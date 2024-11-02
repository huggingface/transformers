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

# Doge

## Overview

The Doge model was proposed in [Wonderful Matrices: More Efficient and Effective Architecture for Language Modeling Tasks](https://arxiv.org/abs/2407.16958) by Jingze Shi.

This model comes from the algorithm proposed in `Wonderful Matrices`, which enhances the state representation and noise filtering capabilities of self-attention through `Inner Function Attention with Dynamic Mask`, and uses a `Cross Domain Mixture of Experts` to avoid parameter redundancy in the feedforward network by mixing dense and sparse activations. However, it removes part of the `State Space Duality` to ensure adaptability to different environments.


The abstract from the paper is the following:

*We prove the availability of inner product form position encoding in the state space duality algorithm and study the effectiveness of different position embeddings in the hybrid quadratic causal self-attention and state space duality algorithms. We propose inner function attention with dynamic mask, which can improve the expressiveness of the attention algorithm and avoid the sequence noise significantly affecting the accuracy of the attention score. We also design cross domain mixture of experts, which can improve the granularity of the sparse activation feedforward network while maintaining the efficiency of parameter utilization and retrieval. The combination of these methods constitutes our foundation model architecture: Wonderful Matrices. We conduct experiments on the language modeling task and find that Wonderful Matrices are more efficient and effective in handling complex language tasks.*

Tips:

- The Doge model is an optimized model for both sequence transformation and state transformation under the Transformers framework.
- By modifying the value part of self-attention to a heuristic `Inner Function`, the model improves the expressiveness of sequence transformation while keeping the number of parameters almost the same.
- An additional learnable parameter `Dynamic Mask` is defined on the basis of the original masking logic, which enhances the noise filtering capability of the sequence.
- An efficient retrieval expert `Efficient Retrieval Experts` is added to the dense activation `MLP` to improve the parameter utilization of the mixture of experts structure.
- The core logic of sequence transformation and state transformation of the Doge model is respectively held in the `DogeInnerFuncAttn` class and the `DogeCDMoE` class.
- Since the fast cuda/triton kernels in `mamba-ssm` have not been updated for a while, and the speed of the naive implementation is slow, the Doge model temporarily removes part of the `State Space Duality`.

## Usage

### Prerequisites

In order to run the `CDMoE` implementation, you first need to install `einx`:
```bash
pip install einx
```

### A simple generation example
```python 
from transformers import DogeConfig, DogeForCausalLM, AutoTokenizer

model_name = '<path-to-new-ckpts>'
model = DogeForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))
```

## DogeConfig

[[autodoc]] DogeConfig

## DogeModel

[[autodoc]] DogeModel
- forward

## DogeForCausalLM

[[autodoc]] DogeForCausalLM
- forward

## DogeForSequenceClassification

[[autodoc]] DogeForSequenceClassification
- forward