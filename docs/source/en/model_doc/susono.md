<!--Copyright 2025 The Susono Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-06-16.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Susono

Susono is a decoder-only language model that extends the [Qwen3-Next](./qwen3_next) architecture with
two additional components: an **Engram** conditional-memory module and an **mHC-Lite** multi-stream
residual connection. When both are disabled, the model is functionally equivalent to Qwen3-Next.

## Key Architecture Features

- **Hybrid attention scheduling** — layers alternate between full softmax attention and
  GatedDeltaNet-style linear attention, controlled by `full_attention_interval` (every `N`-th layer is
  a full-attention layer). The schedule can also be set explicitly via `layer_types`.
- **Mixture of Experts** — sparse routing (`num_experts`, `num_experts_per_tok`) with a shared expert
  and a sigmoid-gated shared-expert path. The shared-expert gate carries a learnable bias initialised
  from `moe_shared_expert_gate_bias_init`.
- **Engram conditional memory** — a deterministic N-gram hash-lookup memory (up to tri-grams) injected
  at the layers listed in `engram_layer_ids`. Engram reads `input_ids`; it is disabled automatically
  when a forward pass only receives `inputs_embeds`.
- **mHC-Lite** — `mhc_num_streams` parallel residual streams combined per layer through a learnable
  convex combination of permutation matrices (spanning the Birkhoff polytope without iterative
  Sinkhorn projection).
- **Attention refinements** — optional QK-LayerNorm (`qk_layernorm`) and an optional Qwen3-Next-style
  attention output gate (`attention_output_gate`), with partial rotary embeddings.

The model supports the standard Transformers features, including SDPA, FlashAttention, gradient
checkpointing and weight tying.

The example below demonstrates how to generate text with Susono using [`Pipeline`] or the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipeline = pipeline(
    task="text-generation",
    model="puwaer/Susono-10B-A1B-Instruct",
    dtype="auto",
    device=0,
)

output = pipeline("The key idea behind conditional memory is")
print(output[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("puwaer/Susono-10B-A1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "puwaer/Susono-10B-A1B-Instruct",
    dtype="auto",
    device_map="auto",
)

inputs = tokenizer("The key idea behind conditional memory is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## SusonoConfig

[[autodoc]] SusonoConfig

## SusonoModel

[[autodoc]] SusonoModel
    - forward

## SusonoForCausalLM

[[autodoc]] SusonoForCausalLM
    - forward

## SusonoForSequenceClassification

[[autodoc]] SusonoForSequenceClassification
    - forward

## SusonoForTokenClassification

[[autodoc]] SusonoForTokenClassification
    - forward

## SusonoForQuestionAnswering

[[autodoc]] SusonoForQuestionAnswering
    - forward
