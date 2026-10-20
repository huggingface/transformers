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
    """Whether no entry of the skip-list `patterns` covers the module `full_name`: one that matches it from its start
    (a name, its sub-tree, or a regex such as `"model.decoder.layer.*"`) or that it ends with (`"fc1"` covers
    `model.decoder.layers.23.fc1`). A match from the start must end at a module boundary: `"model.layers.1"` does not
    cover `model.layers.10`, nor `"mlp.gate"` cover `mlp.gate_proj`."""
    if patterns is None:
        return True
    # the match ends on a "." it consumed, or right before a "." or the end of the name
    boundary = r"(?:(?<=\.)|(?=\.|$))"
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
