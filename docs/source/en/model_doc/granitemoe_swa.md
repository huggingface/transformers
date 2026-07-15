<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-07-15.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# GraniteMoeSWA

GraniteMoeSWA combines the mixture-of-experts (MoE) architecture of [GraniteMoeShared](./granitemoeshared) with the sliding-window attention and learnable attention sinks of [GraniteSWA](./granite_swa):

- **Mixture of experts.** Each block routes every token to a subset of experts (`num_experts_per_tok` of `num_local_experts`). Optional **shared experts** are supported but **disabled by default** (`shared_intermediate_size=0`); set it to a positive value to enable them.
- **Per-layer sliding window attention.** Each layer is either `"full_attention"` or `"sliding_attention"` (configured by `layer_types`). By default every fourth layer (`i % 4 == 0`) keeps full attention and the rest attend only to the most recent `sliding_window` tokens.
- **Learnable per-head attention sinks.** Each head learns a scalar sink that rescales its attention output by `sigmoid(logsumexp(attn_logits) - sink)`, equivalent to appending a single extra learnable logit to the softmax denominator (the attention-sink mechanism used by GPT-OSS).

> [!TIP]
> SDPA is not supported because the attention sink cannot be expressed through `torch.nn.functional.scaled_dot_product_attention`. Supported backends are:
> - **Training + inference:** `"eager"`, `"flex_attention"` (preferred for training)
> - **Inference:** `"flash_attention_3"` (via vLLM FA3 ['hub'](https://github.com/huggingface/kernels) kernel — also the fallback when FlashAttention-3 is not installed but `kernels` is), `"flash_attention_4"`

The example below demonstrates how to generate text with [`Pipeline`] or the [`AutoModelForCausalLM`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline


pipe = pipeline(
    task="text-generation",
    model="ibm-granite/granite-swash-3b-a600m",
)
pipe("Explain quantum computing in simple terms", max_new_tokens=50)
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-swash-3b-a600m")
model = AutoModelForCausalLM.from_pretrained(
    "ibm-granite/granite-swash-3b-a600m",
    device_map="auto",
    # eager default, also supports "flex_attention", "flash_attention_3", "flash_attention_4"
    attn_implementation="eager",
)

inputs = tokenizer("Explain quantum computing in simple terms", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## GraniteMoeSWAConfig

[[autodoc]] GraniteMoeSWAConfig

## GraniteMoeSWAModel

[[autodoc]] GraniteMoeSWAModel
    - forward

## GraniteMoeSWAForCausalLM

[[autodoc]] GraniteMoeSWAForCausalLM
    - forward
