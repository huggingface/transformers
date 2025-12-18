<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Experts Matrix Multiplication Backends

All Mixture-of-Experts (MoE) implementations perform the same high-level computation: for each token, a router selects $k$ experts, then the token hidden state is projected through the selected experts' parameters and aggregated with routing weights. The difference between experts backends is *how* those expert matrix multiplications / projections are executed.

The [`ExpertsInterface`] provides optimized experts implementations. It decouples the experts implementation from the model implementation to simplify experimentation with different functions. Add new backends easily with this consistent interface.

| experts backend   | description                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| `"eager"` | Reference implementation that loops over active experts and applies projections per-expert.       |
| `"batched_mm"`    | Uses `torch.bmm` to compute per-(token, expert) projections in a batched way.                     |
| `"grouped_mm"`    | Uses `torch._grouped_mm` to group tokens by expert and run grouped GEMMs (requires PyTorch 2.9+). |

## Set an experts backend

Use the `experts_implementation` argument in [`~PreTrainedModel.from_pretrained`] to instantiate a model with a specific experts backend.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(

    "Qwen/Qwen1.5-MoE-A2.7B",
    dtype="bfloat16",
    experts_implementation="batched_mm",
)
```

Switch between experts backends at runtime (without reloading the model) using [`~PreTrainedModel.set_experts_implementation`].

```py
model.set_experts_implementation("eager")
```

### Backbone-specific experts backend

Multimodal models can have multiple sub-configs (for example, different backbones). You can set a different experts backend per sub-config by passing a `dict` to `experts_implementation` at load time.

Keys in the mapping must match sub-config names.

```py
from transformers import AutoModelForImageTextToText

experts_implementation_per_backbone = {

    "text_config": "grouped_mm",
    "vision_config": "eager",
}

model = AutoModelForImageTextToText.from_pretrained(

    "Qwen/Qwen3-VL-Moe",
    experts_implementation=experts_implementation_per_backbone,
)
```

Set the experts backend globally with an empty key.

```py
model = AutoModelForCausalLM.from_pretrained(

    "Qwen/Qwen1.5-MoE-A2.7B",
    experts_implementation={"": "batched_mm"},
)
```

## `torch.compile` support

All three backends (`"eager"`, `"batched_mm"`, `"grouped_mm"`) support `torch.compile`.

- `"eager"` and `"batched_mm"` work with all `torch.compile` modes.
- `"grouped_mm"` compiles, but does **not** support compilation modes that rely on CUDA graphs (for example `mode="max-autotune"` or `mode="reduce-overhead"`). If you want to compile `"grouped_mm"`, use a mode that disables CUDA graphs (for example `mode="max-autotune-no-cudagraphs"`) or leave `mode=None`.

```py
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(

    "Qwen/Qwen1.5-MoE-A2.7B",
    dtype="bfloat16",
    experts_implementation="grouped_mm",
).eval().cuda()

# Works for grouped_mm (no CUDA graphs)
model.forward = torch.compile(model.forward, mode="max-autotune-no-cudagraphs")
```
