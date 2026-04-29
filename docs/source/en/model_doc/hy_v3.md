<!--Copyright 2026 THL A29 Limited, a Tencent company and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-04-22.*

# Hy3-preview

## Overview

Hy3-preview is a large-scale Mixture-of-Experts (MoE) language model developed by the Tencent HunYuan team. It features a dense-MoE hybrid architecture with 192 routed experts and 1 always-active shared expert per MoE layer, achieving strong performance with efficient inference via sparse expert activation.

Key architectural features:

- **Dense-MoE hybrid**: The first layer uses a dense FFN; all subsequent layers use MoE with top-k routing (default k=8).
- **Shared experts**: Each MoE layer includes 1 shared expert that processes all tokens alongside the routed experts.
- **Sigmoid routing with expert-bias correction**: Tokens are routed via sigmoid scoring (not softmax) with a learned per-expert bias for load balancing.
- **QK-Norm**: Per-head RMSNorm applied to query and key projections before attention for improved training stability.

## Usage tips

- Load with `AutoModelForCausalLM`. The model requires multiple GPUs due to its size.
- Set `output_router_logits=True` in the config or forward call to collect per-layer MoE router logits. Note that this model does not compute an auxiliary load-balancing loss; `aux_loss` is always `None`.
- The model supports `gradient_checkpointing` to reduce memory during fine-tuning.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "tencent/Hy3-preview"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)

inputs = tokenizer("The future of artificial intelligence is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## HYV3Config

[[autodoc]] HYV3Config

## HYV3Model

[[autodoc]] HYV3Model
    - forward

## HYV3ForCausalLM

[[autodoc]] HYV3ForCausalLM
    - forward
