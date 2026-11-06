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
from typing import Any


def get_module_from_name(module, tensor_name: str) -> tuple[Any, str]:
    if "." in tensor_name:
        module_name, tensor_name = tensor_name.rsplit(".", 1)
        module = module.get_submodule(module_name)
    return module, tensor_name


def should_convert_module(full_name, patterns: list[str] | None = None):
    if patterns is None:
        return True

    # Skip converting when a pattern matches `full_name`:
    # 1. from the start up to a module boundary ("layer.1" matches "layer.1.attn", not "layer.11.attn")
    # 2. as a regex, with the same boundary ("layer.*" matches "layer.11.attn"; "mlp.gate" not "mlp.gate_proj")
    # 3. as a suffix ("fc1" matches "model.decoder.layers.23.fc1")

    boundary = r"(?:(?<=\.)|(?=\.|$))"
    # Group the key so the boundary applies to every alternative.
    return not any(re.match(rf"(?:{key}){boundary}", full_name) or full_name.endswith(key) for key in patterns)


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
