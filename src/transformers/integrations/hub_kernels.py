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
from typing import Union


try:
    from kernels import (
        Device,
        LayerRepository,
        register_kernel_mapping,
        replace_kernel_forward_from_hub,
        use_kernel_forward_from_hub,
    )

    _hub_kernels_available = True

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
            )
        },
        "MLP": {
            "cuda": LayerRepository(
                repo_id="medmekk/triton-llama-mlp",
                layer_name="TritonLlamaMLP",
            )
        },
    }

    register_kernel_mapping(_KERNEL_MAPPING)


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
