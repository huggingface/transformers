# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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
from contextlib import contextmanager

from ..utils.import_utils import is_torch_available
from ..utils.logging import get_logger


logger = get_logger(__name__)


if is_torch_available():
    import torch


# TODO: There should be some operator registration mechanism in torch.onnx to avoid this kind of monkey patching
# Patch torch.where to handle dtype mismatches between x and y when it's called during export
# Patch torch.unsqueeze to support complex tensors during export

original_torch_where = torch.where
original_tensor_where = torch.Tensor.where
original_torch_unsqueeze = torch.unsqueeze
original_tensor_unsqueeze = torch.Tensor.unsqueeze


def patched_torch_where(condition, x=None, y=None):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.dtype != y.dtype:
        y = y.to(x.dtype)
    elif isinstance(x, torch.Tensor) and isinstance(y, (int, float, bool)):
        y = torch.tensor(y, dtype=x.dtype, device=x.device)
    elif isinstance(y, torch.Tensor) and isinstance(x, (int, float, bool)):
        x = torch.tensor(x, dtype=y.dtype, device=y.device)

    if x is None and y is None:
        return original_torch_where(condition)
    elif y is None:
        return original_torch_where(condition, x)
    else:
        return original_torch_where(condition, x, y)


def patched_tensor_where(self, condition, other):
    return patched_torch_where(condition, self, other)


def patched_unsqueeze(self_or_input, dim):
    if torch.is_complex(self_or_input):
        real = original_torch_unsqueeze(self_or_input.real, dim)
        imag = original_torch_unsqueeze(self_or_input.imag, dim)
        return torch.complex(real, imag)
    else:
        return original_torch_unsqueeze(self_or_input, dim)


@contextmanager
def patch_torch_for_onnx_export():
    torch.where = patched_torch_where
    torch.Tensor.where = patched_tensor_where

    torch.unsqueeze = patched_unsqueeze
    torch.Tensor.unsqueeze = patched_unsqueeze

    try:
        yield
    finally:
        torch.where = original_torch_where
        torch.Tensor.where = original_tensor_where

        torch.unsqueeze = original_torch_unsqueeze
        torch.Tensor.unsqueeze = original_tensor_unsqueeze
