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

| experts backend | description                                                                                                                                  |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `"eager"`       | Reference implementation that loops over active experts and applies projections per-expert.                                                  |
| `"batched_mm"`  | Uses [torch.bmm](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html) to compute per-(token, expert) projections in a batched way. |
| `"grouped_mm"`  | Uses `torch._grouped_mm` to group tokens by expert and run grouped GEMMs (requires PyTorch 2.9+).                                            |

`batched_mm` is fastest for very small inputs and compilation speeds it up further. `grouped_mm` performs best for larger inputs.

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

| Implementation | compilation modes                    | dtypes                           | `fullgraph=True` |
| -------------- | ------------------------------------ | -------------------------------- | ---------------- |
| `grouped_mm`   | `None`, `max-autotune-no-cudagraphs` | `bfloat16`                       | Yes              |
| `batched_mm`   | all                                  | `bfloat16`, `float16`, `float32` | Yes              |
| `eager`        | all                                  | `bfloat16`, `float16`, `float32` | No               |

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

### Batch Size 1, Sequence Length 16

| Torch Compile              | Implementation | Mean Latency (ms)                            | Median Latency (ms)                          | P90 Latency (ms)                             | Peak Mem (MB) |
| -------------------------- | -------------- | -------------------------------------------- | -------------------------------------------- | -------------------------------------------- | ------------- |
| False                      | eager          | 271.80                                       | 272.94                                       | 295.34                                       | 27324.65      |
| True                       | eager          | 351.86                                       | 351.64                                       | 384.64                                       | 27329.29      |
| max-autotune-no-cudagraphs | eager          | 352.52                                       | 352.15                                       | 382.79                                       | 27329.29      |
| False                      | batched_mm     | 52.03                                        | 52.07                                        | 52.67                                        | 28382.50      |
| True                       | batched_mm     | 53.04                                        | 53.04                                        | 53.11                                        | 28029.63      |
| max-autotune-no-cudagraphs | batched_mm     | **<span style="color: green;">23.87</span>** | **<span style="color: green;">23.86</span>** | **<span style="color: green;">24.02</span>** | **27329.29**  |
| False                      | grouped_mm     | 64.27                                        | 64.09                                        | 65.49                                        | 27329.29      |
| True                       | grouped_mm     | 59.45                                        | 59.52                                        | 60.99                                        | 27329.29      |
| max-autotune-no-cudagraphs | grouped_mm     | 59.61                                        | 59.55                                        | 60.89                                        | 27329.29      |

### Batch Size 1, Sequence Length 128

| Torch Compile              | Implementation | Mean Latency (ms)                            | Median Latency (ms)                          | P90 Latency (ms)                             | Peak Mem (MB) |
| -------------------------- | -------------- | -------------------------------------------- | -------------------------------------------- | -------------------------------------------- | ------------- |
| False                      | eager          | 471.73                                       | 472.65                                       | 487.97                                       | 27396.46      |
| True                       | eager          | <span style="color: red;">637.32</span>      | 613.70                                       | <span style="color: red;">845.01</span>      | 27429.82      |
| max-autotune-no-cudagraphs | eager          | 620.21                                       | 619.35                                       | 657.74                                       | 27429.82      |
| False                      | batched_mm     | 316.67                                       | 316.94                                       | 317.92                                       | 35854.56      |
| True                       | batched_mm     | 370.29                                       | 370.29                                       | 370.57                                       | 33031.64      |
| max-autotune-no-cudagraphs | batched_mm     | 151.87                                       | 150.38                                       | 158.01                                       | 27429.82      |
| False                      | grouped_mm     | 78.50                                        | 78.53                                        | 80.00                                        | **27429.82**  |
| True                       | grouped_mm     | 72.95                                        | 72.99                                        | 74.60                                        | **27429.82**  |
| max-autotune-no-cudagraphs | grouped_mm     | **<span style="color: green;">72.71</span>** | **<span style="color: green;">72.89</span>** | **<span style="color: green;">73.55</span>** | **27429.82**  |

### Batch Size 4, Sequence Length 16

| Torch Compile              | Implementation | Mean Latency (ms)                            | Median Latency (ms)                          | P90 Latency (ms)                             | Peak Mem (MB) |
| -------------------------- | -------------- | -------------------------------------------- | -------------------------------------------- | -------------------------------------------- | ------------- |
| False                      | eager          | 431.87                                       | 433.38                                       | 448.01                                       | 27391.57      |
| True                       | eager          | <span style="color: red;">566.63</span>      | <span style="color: red;">569.74</span>      | <span style="color: red;">598.98</span>      | 27372.12      |
| max-autotune-no-cudagraphs | eager          | 563.13                                       | 567.79                                       | 588.25                                       | 27372.12      |
| False                      | batched_mm     | 163.41                                       | 163.38                                       | 164.84                                       | 31585.54      |
| True                       | batched_mm     | 189.18                                       | 189.08                                       | 189.79                                       | 30173.45      |
| max-autotune-no-cudagraphs | batched_mm     | 79.15                                        | 79.10                                        | 79.74                                        | 27372.11      |
| False                      | grouped_mm     | 75.23                                        | 75.18                                        | 76.74                                        | 27372.11      |
| True                       | grouped_mm     | 70.35                                        | 70.40                                        | 71.71                                        | **27372.12**  |
| max-autotune-no-cudagraphs | grouped_mm     | **<span style="color: green;">70.26</span>** | **<span style="color: green;">70.43</span>** | **<span style="color: green;">71.32</span>** | **27372.12**  |

### Batch Size 4, Sequence Length 128

| Torch Compile              | Implementation | Mean Latency (ms)                            | Median Latency (ms)                          | P90 Latency (ms)                             | Peak Mem (MB)                             |
| -------------------------- | -------------- | -------------------------------------------- | -------------------------------------------- | -------------------------------------------- | ----------------------------------------- |
| False                      | eager          | 526.88                                       | 522.75                                       | 570.01                                       | 27632.62                                  |
| True                       | eager          | 678.18                                       | 677.54                                       | 690.97                                       | 27762.46                                  |
| max-autotune-no-cudagraphs | eager          | 676.22                                       | 677.07                                       | 681.91                                       | 27762.45                                  |
| False                      | batched_mm     | 1235.25                                      | 1235.33                                      | 1237.90                                      | <span style="color: red;">61465.85</span> |
| True                       | batched_mm     | <span style="color: red;">1505.00</span>     | <span style="color: red;">1503.31</span>     | <span style="color: red;">1536.10</span>     | 50174.26                                  |
| max-autotune-no-cudagraphs | batched_mm     | 572.37                                       | 570.81                                       | 589.74                                       | **27762.45**                              |
| False                      | grouped_mm     | 80.95                                        | 81.06                                        | 81.70                                        | **27762.45**                              |
| True                       | grouped_mm     | **<span style="color: green;">79.67</span>** | **<span style="color: green;">79.69</span>** | **<span style="color: green;">80.54</span>** | **27762.45**                              |
| max-autotune-no-cudagraphs | grouped_mm     | 83.29                                        | 79.83                                        | 111.83                                       | **27762.46**                              |
