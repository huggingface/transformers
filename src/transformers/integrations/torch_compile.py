# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Helpers for compiling ``transformers`` models under ``torch.compile``."""

from __future__ import annotations

import torch
import torch.nn as nn


def mark_repeated_decoder_layers_as_compile_regions(model: nn.Module) -> None:
    """Wrap the outermost repeated-block class's ``forward`` with
    ``torch.compiler.nested_compile_region`` so dynamo compiles one instance's
    body and stamps the artifact out across all N siblings — collapses the
    dominant codegen cost for the layer stack. No-op outside ``torch.compile``.

    Structural traversal mirrors ``accelerate.utils.compile_regions``: top-
    down, find the first ``nn.ModuleList`` of single-class siblings (the
    model's main layer stack), mark that class, and stop. Idempotent — the
    class is tagged so re-entry no-ops.
    """

    def _is_repeated_block(m: nn.Module) -> bool:
        return isinstance(m, nn.ModuleList) and len(m) > 1 and all(isinstance(child, m[0].__class__) for child in m)

    def _find_repeated_block(module: nn.Module) -> nn.ModuleList | None:
        for child in module.children():
            if _is_repeated_block(child):
                return child  # type: ignore[return-value]
            found = _find_repeated_block(child)
            if found is not None:
                return found
        return None

    stack = _find_repeated_block(model)
    if stack is None:
        return
    layer_cls = type(stack[0])
    if getattr(layer_cls, "_forward_is_nested_compile_region", False):
        return
    layer_cls.forward = torch.compiler.nested_compile_region(layer_cls.forward)
    layer_cls._forward_is_nested_compile_region = True
