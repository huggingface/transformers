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

The table below lists the backends that support training.

| backend | description |
|---|---|
| `inductor` | default; traces forward/backward graphs ahead of time and compiles to Triton kernels that run on CUDA |
| `aot_cudagraphs` | captures the GPU operation sequence and replays it with minimal CPU dispatch overhead; best for fixed-shape workloads |
| `aot_nvfuser` / `nvfuser` | NVIDIA's fusion compiler via AOTAutograd or TorchScript |

Set `torch_compile=True` in [`TrainingArguments`] to enable it. Unlike inference, which only compiles the forward pass, training must also trace and compile the backward pass. Expect the first step to be slower.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    ...,
    torch_compile=True,
    torch_compile_backend="inductor",
    torch_compile_mode="reduce-overhead",
)
```

## Next steps

- See the [torch.compile for inference](./perf_torch_compile) guide for details on modes and fullgraph compilation.
