# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import sys
from collections import defaultdict
from contextlib import contextmanager

import torch


# Record all the torch primitives in advance, so that we can use them without them being modified when we patch torch
# in context managers
TORCH_INIT_FUNCTIONS = {
    "uniform_": torch.nn.init.uniform_,
    "normal_": torch.nn.init.normal_,
    "constant_": torch.nn.init.constant_,
    "ones_": torch.nn.init.ones_,
    "zeros_": torch.nn.init.zeros_,
    "eye_": torch.nn.init.eye_,
    "dirac_": torch.nn.init.dirac_,
    "xavier_uniform_": torch.nn.init.xavier_uniform_,
    "xavier_normal_": torch.nn.init.xavier_normal_,
    "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
    "kaiming_normal_": torch.nn.init.kaiming_normal_,
    "trunc_normal_": torch.nn.init.trunc_normal_,
    "orthogonal_": torch.nn.init.orthogonal_,
    "sparse_": torch.nn.init.sparse_,
}


def uniform_(
    tensor: torch.Tensor, a: float = 0.0, b: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["uniform_"](tensor, a=a, b=b, generator=generator)
    return tensor


def normal_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["normal_"](tensor, mean=mean, std=std, generator=generator)
    return tensor


def constant_(tensor: torch.Tensor, val: float) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["constant_"](tensor, val=val)
    return tensor


def ones_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["ones_"](tensor)
    return tensor


def zeros_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["zeros_"](tensor)
    return tensor


def eye_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["eye_"](tensor)
    return tensor


def dirac_(tensor: torch.Tensor, groups: int = 1) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["dirac_"](tensor, groups=groups)
    return tensor


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["xavier_uniform_"](tensor, gain=gain, generator=generator)
    return tensor


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["xavier_normal_"](tensor, gain=gain, generator=generator)
    return tensor


def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["kaiming_uniform_"](
            tensor, a=a, mode=mode, nonlinearity=nonlinearity, generator=generator
        )
    return tensor


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["kaiming_normal_"](
            tensor, a=a, mode=mode, nonlinearity=nonlinearity, generator=generator
        )
    return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["trunc_normal_"](tensor, mean=mean, std=std, a=a, b=b, generator=generator)
    return tensor


def orthogonal_(
    tensor: torch.Tensor,
    gain: float = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["orthogonal_"](tensor, gain=gain, generator=generator)
    return tensor


def sparse_(
    tensor: torch.Tensor, sparsity: float, std: float = 0.01, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["sparse_"](tensor, sparsity=sparsity, std=std, generator=generator)
    return tensor


def copy_(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        with torch.no_grad():
            return tensor.copy_(other)
    return tensor


@contextmanager
def guard_torch_init_functions():
    """
    Guard the `torch.nn.init` primitive functions to behave exactly like the functions in this file, i.e. be
    protected against the `_is_hf_initialized` flag to avoid re-init if the param was already loaded.

    Usually, all models are using the init from `transformers` which are already guarded, but just to make extra sure
    and for remote code, we also use this context manager.
    """
    originals = defaultdict(dict)
    try:
        # Replace all torch funcs by the ones in this file
        for name in TORCH_INIT_FUNCTIONS.keys():
            # Here, we need to check all modules imported, and hot patch all of them, as usually torch does
            # something like `from torch.nn.init import xavier_uniform_` in their internals (e.g in torch.nn.modules,
            # where MultiHeadAttention lives), so the function name is binded at import time and just doing
            # `setattr(torch.nn.init, name, gloabls()[name])` is thus not enough
            for module in sys.modules.copy().values():
                if module and hasattr(module, name):
                    originals[module][name] = getattr(module, name)
                    setattr(module, name, globals()[name])
        yield
    finally:
        # Set back the original functions on all modules
        for module, functions in originals.items():
            for name, func in functions.items():
                setattr(module, name, func)
