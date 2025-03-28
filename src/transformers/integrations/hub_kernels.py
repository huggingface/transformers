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
from typing import Dict, Union


try:
    from kernels import (
        Device,
        LayerRepository,
        register_kernel_mapping,
        replace_kernel_forward_from_hub,
        use_kernel_forward_from_hub,
    )

    _hub_kernels_available = True

    _KERNEL_MAPPING: Dict[str, Dict[Union[Device, str], LayerRepository]] = {
        "MultiScaleDeformableAttention": {
            "cuda": LayerRepository(
                repo_id="kernels-community/deformable-detr",
                layer_name="MultiScaleDeformableAttention",
            )
        }, 
        "LlamaRMSNorm": {
            "cuda": LayerRepository(
                repo_id="kernels-community/triton-layer-norm",
                layer_name="LlamaRMSNorm",
                revision="pure-layer-test",
            )
        },
        "LlamaMLP": {
            "cuda": LayerRepository(
                repo_id="medmekk/triton-llama-mlp",
                layer_name="TritonLlamaMLP",
            )
        },
        "LlamaAttention": {
            "cuda": LayerRepository(
                repo_id="medmekk/triton-flash-attn",
                layer_name="LlamaAttention",
            )
        },
    }

    register_kernel_mapping(_KERNEL_MAPPING)

    def use_kernel_attn_from_hub(layer_name: str, *, device: str = "cuda", use_fallback: bool = False):
        from transformers import AttentionInterface
        from kernels import get_kernel

        def decorator(cls):
            kernel = _KERNEL_MAPPING.get(layer_name)
            if kernel is None:
                if not use_fallback:
                    raise ValueError(f"No attention implementation for `{layer_name}`")
                return cls

            device_obj = Device(type=device)
            if device_obj is None:
                return cls

            repo = kernel.get(device)
            if repo is None:
                if not use_fallback:
                    raise ValueError(
                        f"No layer mapping for attention `{layer_name}` with device type `{device}`"
                    )
                return cls

            attn_kernel = get_kernel(repo.repo_id)
            AttentionInterface.register("attn_kernel", attn_kernel.attention)
            cls.use_kernel = True
            return cls
        return decorator

except ImportError:
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

    _hub_kernels_available = False


def is_hub_kernels_available():
    return _hub_kernels_available

__all__ = [
    "LayerRepository",
    "is_hub_kernels_available",
    "use_kernel_forward_from_hub",
    "register_kernel_mapping",
    "replace_kernel_forward_from_hub",
]
