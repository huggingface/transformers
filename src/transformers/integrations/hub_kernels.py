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
import re
from functools import partial
from typing import Optional, Union

from ..modeling_flash_attention_utils import lazy_import_flash_attention
from .flash_attention import flash_attention_forward


try:
    from kernels import (
        Device,
        LayerRepository,
        Mode,
        get_kernel,
        register_kernel_mapping,
        replace_kernel_forward_from_hub,
        use_kernel_forward_from_hub,
    )

    _kernels_available = True

    _KERNEL_MAPPING: dict[str, dict[Union[Device, str], LayerRepository]] = {
        "MultiScaleDeformableAttention": {
            "cuda": LayerRepository(
                repo_id="kernels-community/deformable-detr",
                layer_name="MultiScaleDeformableAttention",
            )
        },
        "Llama4TextMoe": {
            "cuda": LayerRepository(
                # Move to kernels-community/moe once we release.
                repo_id="kernels-community/moe",
                layer_name="Llama4TextMoe",
            )
        },
        "RMSNorm": {
            "cuda": LayerRepository(
                repo_id="kernels-community/liger_kernels",
                layer_name="LigerRMSNorm",
                # revision="pure-layer-test",
            ),
            "rocm": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-community/liger_kernels",
                    layer_name="LigerRMSNorm",
                    # revision="pure-layer-test",
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
    }

    register_kernel_mapping(_KERNEL_MAPPING)

except ImportError:
    _kernels_available = False

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


def is_kernel(attn_implementation: Optional[str]) -> bool:
    """Check whether `attn_implementation` matches a kernel pattern from the hub."""
    return (
        attn_implementation is not None
        and re.search(r"^[^/:]+/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$", attn_implementation) is not None
    )


def load_and_register_kernel(attn_implementation: str) -> None:
    """Load and register the kernel associated to `attn_implementation`."""
    if not is_kernel(attn_implementation):
        return
    if not _kernels_available:
        raise ImportError("`kernels` is not installed. Please install it with `pip install kernels`.")

    # Need to be imported here as otherwise we have a circular import in `modeling_utils`
    from ..masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

    attention_wrapper = None
    # FIXME: @ArthurZucker this is dirty, did not want to do a lof of extra work
    actual_attn_name = attn_implementation
    if "|" in attn_implementation:
        attention_wrapper, actual_attn_name = attn_implementation.split("|")
        # `transformers` has wrapper for sdpa, paged, flash, flex etc.
        attention_wrapper = ALL_ATTENTION_FUNCTIONS.get(attention_wrapper)
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
        raise ValueError(f"An error occured while trying to load from '{repo_id}': {e}.")
    # correctly wrap the kernel
    if hasattr(kernel, "flash_attn_varlen_func"):
        if attention_wrapper is None:
            attention_wrapper = flash_attention_forward
        kernel_function = partial(attention_wrapper, implementation=kernel)
        lazy_import_flash_attention(kernel)
    elif kernel_name is not None:
        kernel_function = getattr(kernel, kernel_name)
    # Register the kernel as a valid attention
    ALL_ATTENTION_FUNCTIONS.register(attn_implementation, kernel_function)
    ALL_MASK_ATTENTION_FUNCTIONS.register(attn_implementation, ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"])


__all__ = [
    "LayerRepository",
    "use_kernel_forward_from_hub",
    "register_kernel_mapping",
    "replace_kernel_forward_from_hub",
]
