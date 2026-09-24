<!--Copyright 2026 The RWB AI Assist team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-09-16.*

## Overview

BerryLM-OS is a hybrid mixture-of-experts decoder for Russian and English with a thinking mode. Its architecture
combines three ideas:

- **Hybrid attention**: three linear-attention layers per full-attention layer. The linear-attention layers run a
  gated delta rule with a **per-channel forget gate** (a low-rank projection of the layer input sets a separate
  log-decay for every key channel of every head, the Kimi Delta Attention recurrence), the full-attention layers use
  grouped-query attention with a sigmoid output gate and partial rotary embeddings.
- **Gated Block AttnRes**: every decoder layer reads a softmax mixture of the residual streams committed at block
  boundaries (the embeddings count as block 0) instead of the plain residual stream, gated back toward the identity by
  a per-layer scalar (`y = x + tanh(g) * (mix - x)`; `g = 0` is the exact identity). The mixing is token-local, so it
  is transparent to the KV / recurrent caches.
- **Sparse MoE**: every layer has a sparse feed-forward block with 128 routed experts (top-8, renormalized softmax
  routing) plus a shared expert gated by a sigmoid.

The released checkpoint has 40 layers, ~2.8B active / ~18B total parameters and a 256k context.

## Usage examples

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "RWB/BerryLM-OS"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
)

# prepare the model input
prompt = "Give me a short introduction to linear attention."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # thinking is on by default; False for direct answers
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024,
    do_sample=True,
    temperature=1.0,
    top_k=20,
    top_p=1.0,
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
```

The fused kernels of the linear-attention layers (`flash-linear-attention`, `causal-conv1d`) are optional: without
them the torch reference implementation runs.

## BerryLMConfig

[[autodoc]] BerryLMConfig

## BerryLMModel

[[autodoc]] BerryLMModel
    - forward

## BerryLMForCausalLM

[[autodoc]] BerryLMForCausalLM
    - forward
