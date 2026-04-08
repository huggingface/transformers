<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Experts backends

All Mixture-of-Experts (MoE) implementations perform the same high-level computation. For each token, a router selects *k* experts. The token hidden state is then projected through the selected experts' parameters and aggregated with routing weights. The difference between experts backends is *how* those expert matrix multiplications execute.

The [`ExpertsInterface`] provides optimized experts backends. It decouples the experts implementation from the model code to simplify experimentation with different functions. Add new backends through the same interface.

| experts backend | description                                                                                                                                                                                                                               | GPU                                                                                                                 | CPU                                                         |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `"eager"`       | Reference implementation that loops over selected experts and applies projections on their tokens.                                                                                                                                        | Reasonable baseline performance without requiring compilation.                                                      | Slower than `grouped_mm` but faster than `batched_mm`.      |
| `"batched_mm"`  | Duplicates selected expert parameters for each token and projects all tokens in a single batched GEMM using [`torch.bmm`](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html).                                                 | Fastest for small inputs, especially with compilation. Uses more memory due to parameter duplication.               | Not recommended (significantly slower than other backends). |
| `"grouped_mm"`  | Orders tokens by selected experts and uses [`torch.nn.functional.grouped_mm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html) to project all tokens in a single grouped GEMM (requires PyTorch 2.9+). | Best for larger inputs and more memory efficient as it avoids duplicating expert parameters. Fast with compilation. | Most efficient backend for all input sizes.                 |

> [!NOTE]
> When using `experts_implementation="grouped_mm"` on GPU, the model automatically switches to `"batched_mm"` during the decode stage of generation (after prefill). This is because `batched_mm` is significantly faster on lower token count during autoregressive decoding on GPU. On CPU, `grouped_mm` remains active throughout generation as it is more efficient for all input sizes.

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

Switch between experts backends at runtime without reloading the model using [`~PreTrainedModel.set_experts_implementation`].

```py
model.set_experts_implementation("eager")
```

## Backbone-specific experts backend

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

## torch.compile

All three backends (`"eager"`, `"batched_mm"`, `"grouped_mm"`) are compatible with `torch.compile` to certain extents. The following table summarizes compatibility:

| Implementation          | compilation modes                    | dtypes                           | `fullgraph=True` |
| ----------------------- | ------------------------------------ | -------------------------------- | ---------------- |
| `grouped_mm`            | `None`, `max-autotune-no-cudagraphs` | `bfloat16`                       | Yes              |
| `grouped_mm` (fallback) | `None`, `max-autotune-no-cudagraphs` | `bfloat16`, `float16`, `float32` | Yes              |
| `batched_mm`            | all                                  | `bfloat16`, `float16`, `float32` | Yes              |
| `eager`                 | all                                  | `bfloat16`, `float16`, `float32` | No               |

Notes:

- The `grouped_mm` experts backend currently only supports `bfloat16` when compiled with `torch.compile`. Additionally, it is not compatible with CUDA graphs, so you must use `mode=None` or `mode="max-autotune-no-cudagraphs"` when compiling.
- The `eager` experts backend uses a data-dependent operation to find which experts are used in a forward pass. This operation is not compatible with full graph compilation (`fullgraph=True`).

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

## Benchmarks

This [benchmark](https://github.com/user-attachments/files/24125816/bench.py) compares different input sizes and experts implementations with and without `torch.compile`.

<iframe
	src="https://huggingface-moe-experts-benchmarks.static.hf.space"
	frameborder="0"
	width="100%"
	height="800"
></iframe>

<iframe
  src="https://huggingface.co/datasets/docs-benchmarks/experts-backends/embed/viewer/bs1-seq16/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>
