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


def subtree_pattern_to_regex(pattern: str) -> str:
    """A glob-style skip entry as the bounded regex `should_convert_module` matches.

    Checkpoints name whole subtrees with a trailing star ("model.layers.0*", "model.layers.1.*")
    or bare ("self_attn", "model.layers.0."). Those are globs, not regexes, and handing them over
    as-is is wrong twice: the dots match any character, and the star is greedy, so
    "model.layers.1.*" also swallows layers 10-19. The bounded form matches the named module and
    everything under it, nothing else.
    """
    prefix = pattern[:-2] if pattern.endswith(".*") else pattern.rstrip("*").rstrip(".")
    # anchored on both sides at a path boundary: a bare entry names that module wherever it sits
    # ("self_attn" reaches "model.layers.2.self_attn.q_proj") without "layers.1" reaching
    # "layers.16" or "attn" reaching "self_attn"
    return r"(?:^|.*\.)" + re.escape(prefix) + r"(\..*)?$"


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
