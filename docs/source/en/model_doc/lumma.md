<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was contributed to Hugging Face Transformers on 2026-08-26.*

# Lumma

## Overview

Lumma is a multilingual decoder-only language model trained from scratch on 1 trillion tokens, designed for 
efficient deployment and English–Indic language understanding. It features Shared KV attention for ~50% 
reduced KV-cache memory, Grouped Query Attention (GQA), RMSNorm with QK Normalization, SwiGLU FFN, 
and factorized tied embeddings.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("FrontiersMind/Lumma-0.6B-Base")
model = AutoModelForCausalLM.from_pretrained(
    "FrontiersMind/Lumma-0.6B-Base",
    torch_dtype=torch.bfloat16
).eval()

inputs = tokenizer("The world is a strange place", return_tensors="pt")
outputs = model.generate(**inputs,  max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## LummaConfig

[[autodoc]] LummaConfig

## LummaModel

[[autodoc]] LummaModel
    - forward

## LummaForCausalLM

[[autodoc]] LummaForCausalLM
    - forward

## LummaTokenizer

[[autodoc]] LummaTokenizer
