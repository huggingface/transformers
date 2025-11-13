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
from contextlib import contextmanager
from functools import wraps

import torch


def uniform_(
    tensor: torch.Tensor, a: float = 0.0, b: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.uniform_(tensor, a=a, b=b, generator=generator)
    return tensor


def normal_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.normal_(tensor, mean=mean, std=std, generator=generator)
    return tensor


def constant_(tensor: torch.Tensor, val: float) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.constant_(tensor, val=val)
    return tensor


def ones_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.ones_(tensor)
    return tensor


def zeros_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.zeros_(tensor)
    return tensor


def eye_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.eye_(tensor)
    return tensor


def dirac_(tensor: torch.Tensor, groups: int = 1) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.dirac_(tensor, groups=groups)
    return tensor


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.xavier_uniform_(tensor, gain=gain, generator=generator)
    return tensor


def xavier_normal_(tensor: torch.Tensor, gain: float = 1.0, generator: torch.Generator | None = None) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.xavier_normal_(tensor, gain=gain, generator=generator)
    return tensor


def kaiming_uniform_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.kaiming_uniform_(tensor, a=a, mode=mode, nonlinearity=nonlinearity, generator=generator)
    return tensor


def kaiming_normal_(
    tensor: torch.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.kaiming_normal_(tensor, a=a, mode=mode, nonlinearity=nonlinearity, generator=generator)
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
        return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b, generator=generator)
    return tensor


def orthogonal_(
    tensor: torch.Tensor,
    gain: float = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.orthogonal_(tensor, gain=gain, generator=generator)
    return tensor


def sparse_(
    tensor: torch.Tensor, sparsity: float, std: float = 0.01, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.sparse_(tensor, sparsity=sparsity, std=std, generator=generator)
    return tensor


def copy_(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        with torch.no_grad():
            return tensor.copy_(other)
    return tensor


TORCH_INIT_FUNCTIONS = (
    "uniform_",
    "normal_",
    "constant_",
    "ones_",
    "zeros_",
    "eye_",
    "dirac_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "trunc_normal_",
    "orthogonal_",
    "sparse_",
)


@contextmanager
def no_init_weights():
    """
    Context manager to globally disable weight initialization to speed up loading large models.
    """
    global _init_weights
    old_init_weights = _init_weights

    _init_weights = False

    def _skip_init(*args, **kwargs):
        pass

    # Save the original initialization functions
    for name, init_func in TORCH_INIT_FUNCTIONS.items():
        setattr(torch.nn.init, name, _skip_init)

    try:
        yield
    finally:
        _init_weights = old_init_weights
        # Restore the original initialization functions
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, init_func)


@contextmanager
def guard_torch_init():
    """
    Guard the `torch.nn.init` primitive functions to behave exactly like the functions in this file, i.e. be
    protected against the `_is_hf_initialized` flag to avoid re-init if the param was already loaded.
    """
    originals = {}

    def make_wrapper(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            # Tensor can come positionally or as a kwarg
            tensor = args[0] if args else kwargs.get("tensor")
            if not getattr(tensor, "_is_hf_initialized", False):
                return fn(*args, **kwargs)
            return tensor

        return wrapped

    try:
        for name in TORCH_INIT_FUNCTIONS:
            originals[name] = getattr(torch.nn.init, name)
            setattr(torch.nn.init, name, make_wrapper(originals[name]))
        yield
    finally:
        for name, fn in originals.items():
            setattr(torch.nn.init, name, fn)
