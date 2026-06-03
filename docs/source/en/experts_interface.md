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


| experts backend | description | GPU | CPU |
| --- | --- | --- | --- |
| `"eager"` | Reference implementation that loops over selected experts and applies projections on their tokens. | Reasonable baseline performance without requiring compilation. | Slower than `grouped_mm` but faster than `batched_mm`. |
| `"batched_mm"` | Duplicates selected expert parameters for each token and projects all tokens in a single batched GEMM using [torch.bmm](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html). | Fastest for small inputs, especially with compilation. Uses more memory due to parameter duplication. | Not recommended (significantly slower than other backends). |
| `"grouped_mm"` | Orders tokens by selected experts and uses [torch.nn.functional.grouped_mm](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html) to project all tokens in a single grouped GEMM (requires PyTorch 2.9+). | Best for larger inputs and more memory efficient as it avoids duplicating expert parameters. Fast with compilation. | Most efficient backend for all input sizes. |
| `"deepgemm"` | Sorts tokens by selected expert and projects all tokens in a single TMA-aligned grouped GEMM using the [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernels from [kernels-community/deep-gemm](https://huggingface.co/kernels-community/deep-gemm). | Native backend for DeepSeek models on Hopper (SM90+) and Blackwell (SM100+); supports `bfloat16` and FP8/FP4-quantized experts. | Not supported (CUDA-only). |
| `"deepgemm_megamoe"` | Fuses expert-parallel dispatch, the gated MLP (up projection, SwiGLU, down projection), and the EP combine into a single DeepGEMM Mega MoE kernel, overlapping NVLink transfers with tensor-core compute. | Blackwell (SM100+) only, for FP4-quantized experts run with expert parallelism. | Not supported (CUDA-only). |
| `"sonicmoe"` | Fuses the routed `bfloat16` MoE forward (router dispatch, gated up projection, activation, down projection) into CuteDSL grouped-GEMM kernels (from the [quack](https://github.com/Dao-AILab/quack) library) from [kernels-community/sonic-moe](https://huggingface.co/kernels-community/sonic-moe). | State-of-the-art throughput on Hopper (SM90+) for `bfloat16` experts with a gated activation (SwiGLU/GeGLU/ReGLU), especially for training. | Not supported (CUDA-only). |


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

## DeepGEMM

The `"deepgemm"` backend routes expert matmuls through the [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) kernels distributed by [kernels-community/deep-gemm](https://huggingface.co/kernels-community/deep-gemm). It works with unquantized `bfloat16` experts and with FP8/FP4-quantized experts loaded through [Fine-grained FP8](./quantization/finegrained_fp8).

The `"deepgemm"` backend requires:

- CUDA GPU with compute capability ≥ 9.0 (Hopper or newer).
- CUDA runtime 12.3 or later on Hopper, 12.9 or later on Blackwell.
- `nvcc`/`nvrtc` available on the system for the kernel's JIT compilation.
- The [kernels](https://github.com/huggingface/kernels) package.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3",
    dtype="bfloat16",
    experts_implementation="deepgemm",
)
```

The kernel is loaded lazily on the first forward.

### FP8 and FP4 quantized experts

DeepSeek-style checkpoints are usually pre-quantized and carry their own quantization config, so you don't need to pass a [`FineGrainedFP8Config`]. The `"deepgemm"` backend automatically picks the FP8 (or FP4 on Blackwell) grouped-GEMM kernel. DeepGEMM requires dynamic per-row activation scales (`activation_scheme="dynamic"`) and rejects static (per-tensor) activation quantization.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3",
    experts_implementation="deepgemm",
)
```

For FP4-packed expert weights (DeepSeek V4-style), the GPU must be SM100+ (Blackwell). The checkpoint config typically sets `expert_dtype="fp4"` and `scale_fmt="ue8m0"`.

The main reason to pass a [`FineGrainedFP8Config`] for a pre-quantized checkpoint is to dequantize it back to `bfloat16`, in which case the experts run in `bfloat16` rather than on the FP8/FP4 DeepGEMM path.

```py
from transformers import AutoModelForCausalLM, FineGrainedFP8Config

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V3",
    quantization_config=FineGrainedFP8Config(dequantize=True),
    experts_implementation="deepgemm",
)
```

### Fused Mega MoE on Blackwell

On Blackwell (SM100+), set `experts_implementation="deepgemm_megamoe"` to run a single fused kernel that combines expert-parallel dispatch, the up projection, SwiGLU, the down projection, and the EP combine, overlapping NVLink transfers with tensor-core compute.

This backend requires:

- A Blackwell GPU (compute capability ≥ 10.0) with CUDA runtime 12.9 or later.
- FP4-packed expert weights paired with UE8M0 weight scales (the pre-quantized checkpoint typically declares `expert_dtype="fp4"` and `scale_fmt="ue8m0"` in its config).
- A `torch.distributed` process group for the expert-parallel group, which the tensor-parallel wrapping supplies automatically.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V4",
    experts_implementation="deepgemm_megamoe",
    tp_plan="auto",
)
```

## SonicMoE

The `"sonicmoe"` backend fuses the routed MoE forward (dispatch, gated up projection, activation, down projection) into a set of highly optimized CuteDSL grouped-GEMM kernels, built on the [quack](https://github.com/Dao-AILab/quack) library and distributed by [kernels-community/sonic-moe](https://huggingface.co/kernels-community/sonic-moe).

The `"sonicmoe"` backend requires:

- CUDA GPU with compute capability ≥ 9.0 (Hopper or newer).
- The [kernels](https://github.com/huggingface/kernels) package and the `nvidia-cutlass-dsl` package.
- Experts with a gated activation (`silu`, `gelu`, or `relu`, mapped to SwiGLU/GeGLU/ReGLU).

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B",
    dtype="bfloat16",
    experts_implementation="sonicmoe",
)
```

If the requirements aren't met, the forward raises `ImportError` and you should pick a different `experts_implementation`.

## torch.compile

The `"eager"`, `"batched_mm"`, and `"grouped_mm"` backends are compatible with `torch.compile` to varying degrees. The following table summarizes their compatibility. The `"deepgemm"`, `"deepgemm_megamoe"`, and `"sonicmoe"` backends route through external CUDA kernels and aren't covered by this table.


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