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
import os
import re
from collections.abc import Callable
from functools import partial
from types import ModuleType
from typing import Optional, Union

from ..modeling_flash_attention_utils import lazy_import_flash_attention
from ..utils import ENV_VARS_TRUE_VALUES, logging
from ..utils.import_utils import is_kernels_available
from .flash_attention import flash_attention_forward


logger = logging.get_logger(__name__)

try:
    from kernels import (
        Device,
        LayerRepository,
        Mode,
        get_kernel,
        register_kernel_mapping,
        replace_kernel_forward_from_hub,
    )

    _TRANSFORMERS_USE_HUB_KERNELS = os.environ.get("USE_HUB_KERNELS", "YES").upper()
    _kernels_available = True
    _kernels_enabled = _TRANSFORMERS_USE_HUB_KERNELS in ENV_VARS_TRUE_VALUES

    def use_kernel_forward_from_hub(layer_name: str):
        if _kernels_enabled:
            from kernels import use_kernel_forward_from_hub as _kernels_use_kernel_forward_from_hub

            return _kernels_use_kernel_forward_from_hub(layer_name)
        else:
            logger.warning_once(
                f"kernels hub usage is disabled through the environment USE_HUB_KERNELS={_TRANSFORMERS_USE_HUB_KERNELS}"
            )
            return lambda cls: cls

    _KERNEL_MAPPING: dict[str, dict[Union[Device, str], LayerRepository]] = {
        "MultiScaleDeformableAttention": {
            "cuda": LayerRepository(
                repo_id="kernels-community/deformable-detr",
                layer_name="MultiScaleDeformableAttention",
            )
        },
        "Llama4TextMoe": {
            "cuda": LayerRepository(
                repo_id="kernels-community/moe",
                layer_name="Llama4TextMoe",
            )
        },
        "RMSNorm": {
            "cuda": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/liger_kernels",
                    layer_name="LigerRMSNorm",
                    # revision="pure-layer-test",
                ),
            },
            "rocm": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/liger_kernels",
                    layer_name="LigerRMSNorm",
                )
            },
            "xpu": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/rmsnorm",
                    layer_name="RMSNorm",
                )
            },
            "npu": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/liger_kernels",
                    layer_name="LigerRMSNorm",
                )
            },
        },
        "MLP": {
            "cuda": LayerRepository(
                repo_id="medmekk/triton-llama-mlp",
                layer_name="TritonLlamaMLP",
            )
        },
        "MegaBlocksMoeMLP": {
            "cuda": {
                Mode.TRAINING: LayerRepository(
                    repo_id="kernels-community/megablocks",
                    layer_name="MegaBlocksMoeMLP",
                ),
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/megablocks",
                    layer_name="MegaBlocksMoeMLP",
                ),
            },
            "rocm": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="ahadnagy/megablocks",
                    layer_name="MegaBlocksMoeMLP",
                )
            },
        },
        "FastGELU": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/activation",
                    layer_name="FastGELU",
                    version=">=0.0.4,<0.1.0",
                )
            }
        },
        "QuickGELU": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/activation",
                    layer_name="QuickGELU",
                    version=">=0.0.4,<0.1.0",
                )
            }
        },
        "NewGELU": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/activation",
                    layer_name="NewGELU",
                    version=">=0.0.4,<0.1.0",
                )
            }
        },
        "SiLU": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/activation", layer_name="Silu", version=">=0.1.0"
                )
            }
        },
        "GeLU": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/activation", layer_name="Gelu", version=">=0.1.0"
                )
            }
        },
        "GeluTanh": {
            "cuda": {
                Mode.INFERENCE | Mode.TORCH_COMPILE: LayerRepository(
                    repo_id="kernels-community/activation", layer_name="GeluTanh", version=">=0.1.0"
                )
            }
        },
    }

    def has_key(d, key):
        return key in d or any(isinstance(v, dict) and has_key(v, key) for v in d.values())

    def register_kernel_mapping_transformers(mapping=None):
        if mapping is None:
            mapping = _KERNEL_MAPPING
        if has_key(mapping, "xpu") and not is_kernels_available(MIN_VERSION="0.10.2"):
            raise ImportError(
                "kernels uses an incompatible version. Please install the latest version with `pip install -U kernels`."
            )
        register_kernel_mapping(mapping)


except ImportError:
    _kernels_available = False
    _kernels_enabled = False

    # Stub to make decorators int transformers work when `kernels`
    # is not installed.
    def use_kernel_forward_from_hub(*args, **kwargs):
        def decorator(cls):
            return cls

        return decorator

    class LayerRepository:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("LayerRepository requires `kernels` to be installed. Run `pip install kernels`.")

    def replace_kernel_forward_from_hub(*args, **kwargs):
        raise RuntimeError(
            "replace_kernel_forward_from_hub requires `kernels` to be installed. Run `pip install kernels`."
        )

    def register_kernel_mapping(*args, **kwargs):
        raise RuntimeError("register_kernel_mapping requires `kernels` to be installed. Run `pip install kernels`.")


_HUB_KERNEL_MAPPING: dict[str, dict[str, str]] = {
    "causal-conv1d": {"repo_id": "kernels-community/causal-conv1d"},
}

_KERNEL_MODULE_MAPPING: dict[str, Optional[ModuleType]] = {}


def is_kernel(attn_implementation: Optional[str]) -> bool:
    """Check whether `attn_implementation` matches a kernel pattern from the hub."""
    return (
        attn_implementation is not None
        and re.search(r"^[^/:]+/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$", attn_implementation) is not None
    )


def load_and_register_attn_kernel(attn_implementation: str, attention_wrapper: Optional[Callable] = None) -> None:
    """
    Load and register the kernel associated to `attn_implementation`.

    Args:
        attn_implementation: A string, usually a kernel repo like "kernels-community/flash-mla".
        attn_wrapper: a callable for the wrapper around the attention implementation. In `transformers` we
            have a wrapper around the `flash_attn_var_len` call, and the same goes for `sdpa` and `eager`.
            They just prepare the arguments properly. This is mostly used for continious batching, where we
            want the `paged` wrapper, which calls the paged cache.
    """
    from ..masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

    actual_attn_name = attn_implementation.split("|")[1] if "|" in attn_implementation else attn_implementation
    if not is_kernel(actual_attn_name):
        return
    if not _kernels_available:
        raise ImportError(
            "`kernels` is either not installed or uses an incompatible version. "
            "Please install the latest version with `pip install -U kernels`."
        )

    # Extract repo_id and kernel_name from the string
    if ":" in actual_attn_name:
        repo_id, kernel_name = actual_attn_name.split(":")
        kernel_name = kernel_name.strip()
    else:
        repo_id = actual_attn_name
        kernel_name = None
    repo_id = repo_id.strip()
    # extract the rev after the @ if it exists
    repo_id, _, rev = repo_id.partition("@")
    repo_id = repo_id.strip()
    rev = rev.strip() if rev else None

    # Load the kernel from hub
    try:
        kernel = get_kernel(repo_id, revision=rev)
    except Exception as e:
        raise ValueError(f"An error occurred while trying to load from '{repo_id}': {e}.")
    # correctly wrap the kernel
    if hasattr(kernel, "flash_attn_varlen_func"):
        if attention_wrapper is None:
            attention_wrapper = flash_attention_forward
        kernel_function = partial(attention_wrapper, implementation=kernel)
        lazy_import_flash_attention(kernel, force_import=True)
    elif kernel_name is not None:
        kernel_function = getattr(kernel, kernel_name)
    # Register the kernel as a valid attention
    ALL_ATTENTION_FUNCTIONS.register(attn_implementation, kernel_function)
    ALL_MASK_ATTENTION_FUNCTIONS.register(attn_implementation, ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"])


def lazy_load_kernel(kernel_name: str, mapping: dict[str, Optional[ModuleType]] = _KERNEL_MODULE_MAPPING):
    if kernel_name in mapping and isinstance(mapping[kernel_name], ModuleType):
        return mapping[kernel_name]
    if kernel_name not in _HUB_KERNEL_MAPPING:
        logger.warning(f"Kernel {kernel_name} not found in _HUB_KERNEL_MAPPING")
        mapping[kernel_name] = None
        return None
    if _kernels_available:
        from kernels import get_kernel

        try:
            repo_id = _HUB_KERNEL_MAPPING[kernel_name]["repo_id"]
            version = _HUB_KERNEL_MAPPING[kernel_name].get("version", None)
            kernel = get_kernel(repo_id, version=version)
            mapping[kernel_name] = kernel
        except FileNotFoundError:
            mapping[kernel_name] = None

    else:
        # Try to import is_{kernel_name}_available from ..utils
        import importlib

        new_kernel_name = kernel_name.replace("-", "_")
        func_name = f"is_{new_kernel_name}_available"

        try:
            utils_mod = importlib.import_module("..utils.import_utils", __package__)
            is_kernel_available = getattr(utils_mod, func_name, None)
        except Exception:
            is_kernel_available = None

        if callable(is_kernel_available) and is_kernel_available():
            # Try to import the module "{kernel_name}" from parent package level
            try:
                module = importlib.import_module(f"{kernel_name}")
                mapping[kernel_name] = module
                return module
            except Exception:
                mapping[kernel_name] = None
        else:
            mapping[kernel_name] = None

    return mapping[kernel_name]


__all__ = [
    "LayerRepository",
    "use_kernel_forward_from_hub",
    "register_kernel_mapping",
    "register_kernel_mapping_transformers",
    "replace_kernel_forward_from_hub",
    "lazy_load_kernel",
]
