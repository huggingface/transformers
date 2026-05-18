<!--Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-01-01 and added to Hugging Face Transformers on 2026-02-09.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

[Qwen3.5 MoE](https://qwen.ai/blog?id=qwen3.5) is the sparse-expert variant of Qwen3.5. It keeps the same natively multimodal decoder and 3:1 Gated DeltaNet/Gated Attention backbone, but replaces dense FFNs with a 256-expert sparse mixture — 8 routed experts are activated per token, plus 1 shared expert — so total parameters scale well past the dense checkpoints while active compute per token stays much smaller.

Notable checkpoints include Qwen/Qwen3.5-35B-A3B (35B total/3B active), Qwen/Qwen3.5-122B-A10B, Qwen/Qwen3.5-397B-A17B, and Qwen/Qwen3.6-35B-A3B. Qwen3.6 checkpoints share the same architecture and `model_type` as Qwen3.5 and are loaded with the same classes. The text tower reuses `Qwen3NextSparseMoeBlock` and expert kernels from Qwen3-Next; the vision tower is inherited from Qwen3-VL.

You can find all the official Qwen3.5 MoE checkpoints under the [Qwen](https://huggingface.co/Qwen) organization.

## Quickstart

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="Qwen/Qwen3.5-35B-A3B",
    device_map="auto",
)
print(pipe("The capital of France is", max_new_tokens=20)[0]["generated_text"])
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")
model = Qwen3_5MoeForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-35B-A3B",
    device_map="auto",
)

inputs = tokenizer("Explain mixture-of-experts in one paragraph.", return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Usage tips and notes

- When training or fine-tuning, set `output_router_logits=True` so the forward returns router logits and the load-balancing auxiliary loss is added to the total loss (scaled by `router_aux_loss_coef`, default `0.001`). Without it, experts can collapse to a few popular slots.
- [`Qwen3_5MoeCausalLMOutputWithPast`] includes a `router_logits` field. Downstream code that destructures model outputs by position needs to account for it or switch to keyword access.
- For Qwen3.5-35B-A3B, the text config uses `hidden_size=2048` across 40 layers, 256 experts with 8 routed + 1 shared per token, and `moe_intermediate_size=512` — very different shapes from the dense Qwen3.5 checkpoints, so weights are not interchangeable.
- Native context is 262,144 tokens. To reach the advertised ~1M context, enable YaRN rope scaling via the config's `rope_scaling` field — plain loading gives you the native window only.
- As with Qwen3.5, linear-attention layers depend on optional `causal_conv1d` (from [Dao-AILab](https://github.com/Dao-AILab/causal-conv1d)). Without it, the model silently falls back to slower and more memory hungry PyTorch ops.

## Qwen3_5MoeConfig

[[autodoc]] Qwen3_5MoeConfig

## Qwen3_5MoeTextConfig

[[autodoc]] Qwen3_5MoeTextConfig

## Qwen3_5MoeVisionConfig

[[autodoc]] Qwen3_5MoeVisionConfig

## Qwen3_5MoeVisionModel

[[autodoc]] Qwen3_5MoeVisionModel
    - forward

## Qwen3_5MoeTextModel

[[autodoc]] Qwen3_5MoeTextModel
    - forward

## Qwen3_5MoeModel

[[autodoc]] Qwen3_5MoeModel
    - forward

## Qwen3_5MoeForCausalLM

[[autodoc]] Qwen3_5MoeForCausalLM
    - forward

## Qwen3_5MoeForConditionalGeneration

[[autodoc]] Qwen3_5MoeForConditionalGeneration
    - forward
