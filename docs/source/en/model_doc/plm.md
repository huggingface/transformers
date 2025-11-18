<!--Copyright 2025 The PLM Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# PLM
<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The PLM model was proposed in [PLM: Efficient Peripheral Language Models Hardware-Co-Designed for Ubiquitous Computing](https://arxiv.org/abs/2503.12167) by PLM-Team.

### Summary

The PLM (Peripheral Language Model) series introduces a novel model architecture to peripheral computing by delivering powerful language capabilities within the constraints of resource-limited devices. Through modeling and system co-design strategy, PLM optimizes model performance and fits edge system requirements, PLM employs Multi-head Latent Attention and squared ReLU activation to achieve sparsity, significantly reducing memory footprint and computational demands. Coupled with a meticulously crafted training regimen using curated datasets and a Warmup-Stable-Decay-Constant learning rate scheduler, PLM demonstrates superior performance compared to existing small language models, all while maintaining the lowest activated parameters, making it ideally suited for deployment on diverse peripheral platforms like mobile phones and Raspberry Pis.


## Usage tips

Ensure your Transformers library version is up-to-date. PLM requires Transformers>=4.51.3 for full support.


`PLM-1.8B-Instruct` can be found on the [Huggingface Hub](https://huggingface.co/PLM-Team/PLM-1.8B-Instruct)


In the following, we demonstrate how to use it for inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("PLM-Team/PLM-1.8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("PLM-Team/PLM-1.8B-Instruct", torch_dtype=torch.bfloat16)

# Input text
input_text = "Tell me something about reinforcement learning."
inputs = tokenizer(input_text, return_tensors="pt")

# Completion
output = model.generate(inputs["input_ids"], max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```


## PLMConfig

[[autodoc]] PLMConfig

## PLMModel

[[autodoc]] PLMModel
    - forward

## PLMForCausalLM

[[autodoc]] PLMForCausalLM
    - forward

## PLMForSequenceClassification

[[autodoc]] PLMForSequenceClassification
    - forward

## PLMForTokenClassification

[[autodoc]] PLMForTokenClassification
    - forward