# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import torch


def get_module_from_name(module, tensor_name: str) -> tuple[Any, str]:
    if "." in tensor_name:
        module_name, tensor_name = tensor_name.rsplit(".", 1)
        module = module.get_submodule(module_name)
    return module, tensor_name


def is_packed_experts_module(module: "torch.nn.Module") -> bool:
    """Returns whether `module` is a fused-experts module whose weights can be swapped for packed (quantized) ones. This
    requires the module to have the right projects (gate_up_proj, down_proj), to be decorated with the
    `use_experts_implementation` (which sets the `has_gate` and `num_experts` attributes), and to not be transposed.
    """
    return (
        hasattr(module, "gate_up_proj")
        and hasattr(module, "down_proj")
        and hasattr(module, "has_gate")
        and hasattr(module, "num_experts")
        and not getattr(module, "is_transposed", False)
    )


def try_set_experts_implementation(model, module_names: list[str], implementation: str) -> list[str]:
    """Attempts to switch `model`'s named `.experts` modules to `implementation` via `set_experts_implementation`,
    and returns the subset of `module_names` whose module failed to adopt it (e.g. because the model architecture
    does not support that implementation). Does nothing and returns `[]` if `module_names` is empty.

    This is not mxfp4-specific: `set_experts_implementation` is the general mechanism MoE modules use to switch
    between any registered experts implementation (`"eager"`, `"grouped_mm"`, a quantizer-provided one, ...).
    """
    if not module_names:
        return []
    model.set_experts_implementation(implementation)
    return [
        name for name in module_names if model.get_submodule(name).config._experts_implementation != implementation
    ]


def should_convert_module(full_name, patterns: list[str] | None = None):
    if patterns is None:
        return True

    # We should avoid converting in the following situations:
    # 1. The pattern appears as a prefix followed by a dot in `full_name`
    #    (e.g., "model.decoder.layer.11." matches "model.decoder.layer.11.attn.weight").
    # 2. The pattern matches `full_name` exactly or via regex
    #    (e.g., "lm_head" matches "lm_head"; "model.decoder.layer.*" matches "model.decoder.layer.11.attn.weight").
    # 3. `full_name` ends with the pattern
    #    (e.g., "fc1" matches "model.decoder.layers.23.fc1").

    should_not_convert = any(
        re.match(f"{key}\\.", full_name) or re.match(f"{key}", full_name) or full_name.endswith(key)
        for key in patterns
    )
    return not should_not_convert


@contextmanager
def on_device(device):
    """Align the current accelerator device with a tensor or device-like object."""
    from ..utils import is_torch_available

    if is_torch_available():
        import torch

        if isinstance(device, torch.Tensor):
            device = device.device
        elif isinstance(device, str):
            device = torch.device(device)

        device_type = getattr(device, "type", None)
        if device_type == "cuda":
            with torch.cuda.device(device):
                yield
                return
        if device_type == "xpu" and hasattr(torch, "xpu"):
            with torch.xpu.device(device):
                yield
                return

    yield
