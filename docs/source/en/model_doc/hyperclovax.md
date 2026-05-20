<!--Copyright 2026 NAVER Cloud Corp. and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-07-21 and added to Hugging Face Transformers on 2026-05-06.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# HyperCLOVA X

## Overview

HyperCLOVA X SEED Think is NAVER Cloud's language model combining pruning and knowledge distillation with advanced reasoning capabilities. The 14B model features a Transformer-based architecture with Peri-Layer Normalization and Maximal Update Parameterization (μP), 14.74B parameters, and 32k context length. It supports dual-mode reasoning (think / non-think) and function calling via a ChatML-based format.

The model was trained with a multi-stage RL pipeline (SFT → RLVR → Length Controllability → joint RLHF+RLVR) and achieves strong performance on Korean language benchmarks and reasoning tasks.

HyperCLOVA X shares a high degree of implementation similarity with [Granite](./granite), with the following modifications:

- **Maximal Update Parametrization (MuP)**: uses per-config scaling factors (`attention_multiplier`, `residual_multiplier`, `embedding_multiplier`, `logits_scaling`) to enable stable training across model sizes. `head_dim` (defaults to `hidden_size // num_attention_heads`) is used to compute the default `attention_multiplier`.
- **Peri-Layer Normalization** (optional): applies an extra RMSNorm after each sub-layer output when `use_post_norm=True`.

This model was contributed by [NAVER Cloud HyperCLOVA X Team](https://huggingface.co/naver-hyperclovax). The original model can be found at [naver-hyperclovax/HyperCLOVAX-SEED-Think-14B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B).

## Usage

The model uses a ChatML-based format with special tokens `<|im_start|>`, `<|im_end|>`, `<|endofturn|>`, and `<|stop|>`. The `apply_chat_template` method accepts the following kwargs:

- `force_reasoning=True` — always think before answering
- `skip_reasoning=True` — always answer directly (non-think mode)
- Default (`None`) — model decides based on context

<hfoptions id="usage">
<hfoption id="AutoModelForCausalLM">

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of South Korea?"},
]
# Pass force_reasoning=True to always think, or skip_reasoning=True to skip thinking.
model_inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    # force_reasoning=True,
    # skip_reasoning=True,
).to(model.device)

output = model.generate(
    **model_inputs,
    tokenizer=tokenizer,
)
print(tokenizer.decode(output[0][model_inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## HyperCLOVAXConfig

[[autodoc]] HyperCLOVAXConfig

## HyperCLOVAXModel

[[autodoc]] HyperCLOVAXModel
    - forward

## HyperCLOVAXForCausalLM

[[autodoc]] HyperCLOVAXForCausalLM
    - forward