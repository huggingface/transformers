<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Writing kernels

This guide explains how to write kernels that go beyond a stateless `forward` replacement. It covers two capabilities the extended `KernelConfig` API supports:

1. **Parameter transformation** — the kernel expects weights in a different layout than the original model (for example, renamed or merged parameters).
2. **Module fusion** — the kernel replaces multiple adjacent modules with a single fused implementation.

For basic kernels (stateless `forward` replacements with no parameter changes), see the [kernels library](https://github.com/huggingface/kernels) documentation.

## Two-class pattern

Any kernel that carries its own parameters follows a two-class pattern.

- `KernelName`: contains only the `forward` pass. The `kernels` library uses this class to kernelize the model. It must be stateless — the library does not allow stateful kernel classes.
- `KernelNameLayout`: an `nn.Module` that holds the parameters and monkey-patches the original module before the checkpoint is loaded. At runtime, `kernelize` replaces its `forward` with `KernelName`'s `forward`.

The naming convention is strict: the layout class must be named `{KernelName}Layout` and defined in the same module as `KernelName`. Transformers discovers it automatically.

## Parameter transformation

Use this pattern when the kernel expects weights under different names or in a different shape than the original model checkpoint.

The `KernelNameLayout` class has the **same `__init__` signature as the module it replaces** and declares a `conversion_mapping` class attribute that tells Transformers how to remap checkpoint keys to the new parameter names.

```python
import torch
import torch.nn as nn

class CustomRMSNormLayout(nn.Module):
    conversion_mapping = [...]  # rules that remap checkpoint keys to the new parameter names

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass  # replaced at runtime by kernelize


class CustomRMSNorm(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.scale * hidden_states.to(input_dtype)


class layers:
    CustomRMSNorm = CustomRMSNorm
```

> [!NOTE]
> The `layers` class is required by the `kernels` library to expose the kernel entry point.

Load this kernel by passing the repo and class name to `KernelConfig`. The key is the **original** module class name from the model; the value points to the `KernelName` class (not the Layout) in the repo.

```python
from transformers import AutoModelForCausalLM, KernelConfig

kernel_config = KernelConfig({"RMSNorm": "owner/my-kernel:CustomRMSNorm"})
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_kernels=True,
    kernel_config=kernel_config,
    device_map="cuda",
)
```

When the model loads, Transformers:
1. Loads `CustomRMSNorm` from the repo and looks for `CustomRMSNormLayout` in the same module.
2. Monkey-patches every `RMSNorm` in the model with `CustomRMSNormLayout`.
3. Remaps checkpoint weights using `conversion_mapping` so they load into the new parameter names.
4. Calls `kernelize`, which replaces `CustomRMSNormLayout.forward` with `CustomRMSNorm.forward`.

## Module fusion

Use this pattern when a kernel replaces multiple adjacent modules with a single fused implementation. Because the fused module combines parameters from several original modules, the `KernelNameLayout.__init__` receives the **instantiated child modules** rather than their constructor arguments.

```python
import torch
import torch.nn as nn

class RMSNormMLPLayout(nn.Module):
    conversion_mapping = [...]  # rules that remap checkpoint keys to the fused parameter names

    def __init__(self, norm, mlp):
        super().__init__()
        self.variance_epsilon = norm.variance_epsilon
        self.scale = nn.Parameter(torch.empty_like(norm.weight))
        self.gate_up_proj = nn.Linear(
            mlp.gate_proj.in_features,
            mlp.gate_proj.out_features + mlp.up_proj.out_features,
            bias=False,
            device=mlp.gate_proj.weight.device,
            dtype=mlp.gate_proj.weight.dtype,
        )
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass  # replaced at runtime by kernelize


class RMSNormMLP(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.scale * hidden_states.to(input_dtype)
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class layers:
    RMSNormMLP = RMSNormMLP
```

To fuse modules, pass a **tuple of `(class_name, path_pattern)` pairs** as the key in `KernelConfig` instead of a plain string. All patterns must share the same parent module (Transformers fuses the children in that parent). The `*` wildcard matches any single path segment.

```python
from transformers import AutoModelForCausalLM, KernelConfig

kernel_config = KernelConfig(
    {
        (
            ("RMSNorm", "model.layers.*.post_attention_layernorm"),
            ("MLP",     "model.layers.*.mlp"),
        ): "owner/my-kernel:RMSNormMLP",
    }
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    use_kernels=True,
    kernel_config=kernel_config,
    device_map="cuda",
)
```

When the model loads, Transformers:
1. Loads `RMSNormMLP` from the repo and finds `RMSNormMLPLayout` in the same module.
2. Matches every decoder layer at `model.layers.*` and builds a fused parent class whose `__init__` calls `RMSNormMLPLayout(post_attention_layernorm, mlp)`.
3. Replaces the remaining child (`mlp`) with `nn.Identity()` to preserve the parent module's interface.
4. Remaps checkpoint weights using `conversion_mapping`.
5. Calls `kernelize`, which replaces `RMSNormMLPLayout.forward` with `RMSNormMLP.forward`.

> [!TIP]
> The order of pairs in the fusion tuple determines the argument order passed to `KernelNameLayout.__init__`.
