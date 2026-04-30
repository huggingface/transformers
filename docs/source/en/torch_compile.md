<!---Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# torch.compile

[torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) compiles PyTorch code to fused kernels to make it run faster. For training, it traces both the forward and backward pass together and compiles them into optimized kernels, reducing the overhead of individual op launches and fusing operations to cut memory bandwidth usage.

Set `torch_compile=True` in [`TrainingArguments`] to enable it. Training compiles both the forward and backward pass, unlike inference which only compiles the forward pass. Compilation happens on the first training step, so expect it to be significantly slower than subsequent steps.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    ...,
    torch_compile=True,
    torch_compile_backend="inductor",
    torch_compile_mode="reduce-overhead",
)
```

## Backend

The default backend is `inductor`, which compiles to Triton kernels with AOTAutograd. This is the right choice for most training workloads. Use `cudagraphs` for fixed-shape inputs, or `ipex` for Intel CPU training.

## Compile mode

Use the table below to help select a `torch.compile` mode.

| mode | description |
|---|---|
| default | balanced compile time vs runtime   |
| reduce-overhead | reduces Python/CPU overhead using CUDA graphs at the cost of some extra memory |
| max-autotune | benchmarks multiple kernel implementations at compile and picks the fastest (longer compilation) |
| max-autotune-no-cudagraphs | same as max-autotune but without CUDA graphs |

## Next steps

- See the [torch.compile for inference](./perf_torch_compile) guide for details on fullgraph compilation and inference benchmarks.
