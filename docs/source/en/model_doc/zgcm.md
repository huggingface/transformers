<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-09-15.*

# ZGCM

[ZGCM-1](https://github.com/zgcagi/ZGCM-1) is a dense decoder-only language model developed by Zhongguancun Academy and Zhongguancun Institute of Artificial Intelligence for mathematical reasoning and tool-assisted search. The [ZGCM-1-7B checkpoint](https://huggingface.co/zgcagi/ZGCM-1-7B) contains approximately 7.39 billion parameters and supports a configured context length of 262,144 tokens.

The model combines 27 gated sliding-window attention layers with five full-attention layers. It uses grouped-query attention with 32 query heads and eight key/value heads, query/key RMS normalization, and partial rotary position embeddings. The local attention window contains 128 tokens, including the current token.

## Usage

Load the checkpoint with the native Transformers implementation and use its chat template to format messages.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "zgcagi/ZGCM-1-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=False,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
messages = [{"role": "user", "content": "Explain why the sum of two odd numbers is even."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Implementation notes

- SDPA is the default attention backend. Set `attn_implementation="flash_attention_2"` to use FlashAttention 2 with a compatible installation, GPU, and dtype. Eager attention is also supported.
- KV caching is enabled by default. Sliding layers retain the recent keys and values needed for their local window, while full-attention layers retain the full history.
- Partial RoPE rounds the configured rotary dimension down to an even number. For the published checkpoint, `head_dim=128` and `partial_rotary_factor=0.334` rotate the first 42 dimensions of each head.
- RMS normalization and rotary arithmetic preserve the reference implementation's float32 computation before casting back to the activation dtype. Attention gates apply an elementwise sigmoid before the attention output projection.
- Existing checkpoint parameter names are preserved, so no weight conversion is required. Despite their names, `post_attention_layernorm` and `post_feedforward_layernorm` normalize the inputs to the attention and feedforward sublayers, respectively.
- Different attention backends may produce different floating-point results and generated continuations, especially in BF16.

## ZgcmConfig

[[autodoc]] ZgcmConfig

## ZgcmModel

[[autodoc]] ZgcmModel
    - forward

## ZgcmForCausalLM

[[autodoc]] ZgcmForCausalLM
    - forward
