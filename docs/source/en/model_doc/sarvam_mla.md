<!--Copyright 2026 Sarvam AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

-->

# SarvamMLA

## Overview

SarvamMLA is a 105B parameter Mixture of Experts (MoE) language model developed by [Sarvam AI](https://www.sarvam.ai/). It uses Multi-head Latent Attention (MLA) combined with sparse MoE routing, architecturally identical to DeepSeek-V3.

Key architectural features:

- **Multi-head Latent Attention (MLA)**: Low-rank KV compression with decoupled RoPE, reducing KV cache memory while maintaining performance.
- **Sparse Mixture of Experts**: 128 routed experts with 8 active per token, plus 1 shared expert. The first layer uses a dense MLP.
- **DeepSeek YaRN RoPE**: Extended context support up to 131K tokens via YaRN rotary position embeddings.
- **Sigmoid routing with group-based top-k**: Token-choice routing using sigmoid scores with expert bias correction and group-aware selection.

This model uses the DeepSeek-V3 architecture with a custom configuration. See the [DeepSeek-V3 documentation](deepseek_v3) for model and forward reference.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "sarvamai/sarvam-105b",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-105b")

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## SarvamMLAConfig

[[autodoc]] SarvamMLAConfig
