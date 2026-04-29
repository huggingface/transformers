<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-09-05 and added to Hugging Face Transformers on 2026-04-29.*

# MiniCPM3

## Overview

MiniCPM3 is the third-generation MiniCPM dense language model from OpenBMB. The 4B variant
([`openbmb/MiniCPM3-4B`](https://huggingface.co/openbmb/MiniCPM3-4B)) outperforms many 7B–9B open
models on standard benchmarks while remaining lightweight enough for on-device usage.

MiniCPM3 combines several architectural ideas:

- **Multi-head Latent Attention (MLA)** from DeepSeek-V2, which compresses the key/value cache
  into a low-rank latent representation while still using rotary embeddings on a portion of the
  query/key heads.
- A standard SwiGLU MLP (no MoE).
- Three scalar scaling factors that govern signal flow:
  - `scale_emb` — scales input embeddings.
  - `scale_depth / sqrt(num_hidden_layers)` — scales residual connections.
  - `hidden_size / dim_model_base` — scales hidden states before the language model head.

## Usage tips

```python
from transformers import AutoTokenizer, MiniCPM3ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM3-4B")
model = MiniCPM3ForCausalLM.from_pretrained("openbmb/MiniCPM3-4B", device_map="auto")

inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## MiniCPM3Config

[[autodoc]] MiniCPM3Config

## MiniCPM3Model

[[autodoc]] MiniCPM3Model
    - forward

## MiniCPM3ForCausalLM

[[autodoc]] MiniCPM3ForCausalLM
    - forward

## MiniCPM3ForSequenceClassification

[[autodoc]] MiniCPM3ForSequenceClassification
    - forward
