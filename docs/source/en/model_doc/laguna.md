<!--Copyright 2026 Poolside and The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->
*This model was released on 2026-04-28 and added to Hugging Face Transformers on 2026-04-28.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Laguna

Laguna is Poolside's mixture-of-experts language model family. The Laguna-specific
deltas vs a standard SwiGLU MoE transformer are:

- **Per-layer head counts** via `num_attention_heads_per_layer` — different decoder
  layers can have different query-head counts while sharing the same KV cache shape.
- **Sigmoid MoE router with auxiliary-loss-free load balancing**
  ([arXiv:2408.15664](https://huggingface.co/papers/2408.15664)) and optional logit
  soft-capping (`moe_router_logit_softcapping`) — router scores are the element-wise
  sigmoid of the gate logits plus a learned per-expert bias (`e_score_correction_bias`)
  that is added at selection time only.

## Usage

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="poolside/Laguna-XS.2",
    dtype="auto",
    device_map="auto",
)
print(pipe("The capital of France is", max_new_tokens=20, do_sample=False)[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "poolside/Laguna-XS.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generated = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(generated[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Notes

- **Attention backends.** SDPA (default), FlashAttention-2, and flex attention are
  supported. Attention-output gating is applied outside the kernel call and
  therefore works with all backends.
- **`num_attention_heads_per_layer`.** When provided, its length must equal
  `num_hidden_layers`. Each entry must be divisible by `num_key_value_heads`.
- **`layer_types`.** Defaults to `["full_attention"] * num_hidden_layers` when left
  unset. To enable sliding-window attention, pass a list of
  `"full_attention"` / `"sliding_attention"` values.
- **`mlp_layer_types`.** Per-layer MLP type, values `"dense"` or `"sparse"`. Length must
  equal `num_hidden_layers`. Defaults to `["dense"] + ["sparse"] * (num_hidden_layers - 1)`
  (first layer dense, rest MoE) when left unset.
- **`moe_apply_router_weight_on_input=True`** is not currently supported alongside the
  fused experts kernel (`grouped_mm_experts_forward`); `validate_architecture` raises at
  config-construction time. Set it to `False` (the default).

## LagunaConfig

[[autodoc]] LagunaConfig

## LagunaModel

[[autodoc]] LagunaModel
    - forward

## LagunaForCausalLM

[[autodoc]] LagunaForCausalLM
    - forward
