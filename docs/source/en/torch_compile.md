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

[torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) compiles PyTorch code to fused kernels to make it run faster. The backend choice is important because not all backends support training.

| backend | description |
|---|---|
| inductor | default backend that traces forward/backward graphs ahead of time and compiles to Triton kernels that run on CUDA |
| aot_cudagraphs | uses CUDA graphs with AOTAutograd to capture the entire sequence of GPU operations and replay them with minimal CPU overhead |
| nvfuser/aot_nvfuser | NVIDIA's fusion compiler integrated with TorchScript or AOTAutograd |

Set `torch_compile=True` in [`TrainingArguments`] to enable it.

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

- See [torch.compile for inference](./perf_torch_compile) guide for more details about mode and fullgraph..
