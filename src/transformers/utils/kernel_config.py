# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from ..integrations.hub_kernels import _FUSION_PATTERNS_REGISTRY, fuse_modules
from ..utils import PushToHubMixin


def infer_device(model):
    """
    Infers the device type from the model parameters.
    Args:
        model: The model instance.

    Returns:
        The device type.
    """
    EXAMPLE_MAPPING = """
    {
        "RMSNorm": {
            "cuda":
                "kernels-community/layer_norm:LlamaRMSNorm",
            ...
        },
        ...
    }
    """
    try:
        param = next(model.parameters())
    except StopIteration:
        raise ValueError(
            f"Cannot determine model device, please provide a device to the mapping. Example: {EXAMPLE_MAPPING}"
        )

    dev_type = param.device.type
    if dev_type == "cuda":
        # Refine based on actual platform
        from ..utils import is_torch_available

        if is_torch_available():
            import torch

            if getattr(torch, "version").hip is not None:
                return "rocm"

    return dev_type


def add_to_mapping(layer_name, device, repo_name, mode, compatible_mapping):
    from kernels import LayerRepository

    if device not in ["cuda", "rocm", "xpu", "npu", "neuron"]:
        raise ValueError(f"Only cuda, rocm, xpu, npu and neuron devices supported, got: {device}")
    repo_layer_name = repo_name.split(":")[1]
    repo_id = repo_name.split(":")[0]
    compatible_mapping[layer_name] = {
        device: {
            mode: LayerRepository(
                repo_id=repo_id,
                layer_name=repo_layer_name,
            )
        }
    }


def add_to_mapping_local(layer_name, device, repo_name, mode, compatible_mapping):
    from pathlib import Path

    from kernels import LocalLayerRepository

    if device not in ["cuda", "rocm", "xpu", "npu", "neuron"]:
        raise ValueError(f"Only cuda, rocm, xpu, npu and neuron devices supported, got: {device}")
    repo_layer_name = repo_name.split(":")[1]
    repo_path = repo_name.split(":")[0]
    repo_package_name = repo_path.split("/")[-1]
    compatible_mapping[layer_name] = {
        device: {
            mode: LocalLayerRepository(
                repo_path=Path(repo_path),
                package_name=repo_package_name,
                layer_name=repo_layer_name,
            )
        }
    }


class KernelConfig(PushToHubMixin):
    """
    Kernel configuration class. This class is used to configure the kernel mapping for a model.
    """

    def __init__(self, kernel_mapping=None, use_local_kernel=False):
        self.kernel_mapping = kernel_mapping if kernel_mapping is not None else {}
        self.registered_layer_names = {}
        self.use_local_kernel = use_local_kernel

    def update_kernel(self, repo_id, registered_name, layer_name, device, mode, revision=None):
        from kernels import LayerRepository

        self.kernel_mapping[registered_name] = {
            device: {
                mode: LayerRepository(
                    repo_id=repo_id,
                    layer_name=layer_name,
                    revision=revision,
                )
            }
        }

    def apply_fusions(self, model):
        """
        For each n-to-1 entry (tuple key) in the kernel mapping, find the fusion patterns
        registered on the model, fuse the corresponding modules in-place, then replace the
        tuple key with the resolved kernel layer name so the rest of the pipeline is unchanged.
        """
        new_mapping = {}
        for layer_name, kernel in self.kernel_mapping.items():
            if not isinstance(layer_name, tuple):
                new_mapping[layer_name] = kernel
                continue

            # Parse the target kernel layer name from the repo string (the part after ':')
            repo_str = kernel if isinstance(kernel, str) else next(iter(kernel.values()))
            if isinstance(repo_str, dict):
                repo_str = next(iter(repo_str.values()))
            kernel_layer_name = repo_str.split(":")[1] if ":" in repo_str else repo_str.split("/")[-1]

            # Only fuse if a kernel is available for the current device
            current_device = infer_device(model)
            has_kernel_for_device = isinstance(kernel, str) or current_device in kernel
            if not has_kernel_for_device:
                continue

            # Detect inline format: tuple of (kernel_layer_name, glob_pattern) pairs.
            # e.g. (("RMSNorm", "model.layers.*.post_attention_layernorm"), ("MLP", "model.layers.*.mlp"))
            is_inline = all(isinstance(item, tuple) and len(item) == 2 for item in layer_name)
            if is_inline:
                source_names = [item[0] for item in layer_name]
                patterns = [item[1] for item in layer_name]
                fuse_modules(model, patterns, kernel_layer_name, source_layer_names=source_names)
            else:
                fusion_patterns = getattr(model, "_kernel_fusion_patterns", None) or _FUSION_PATTERNS_REGISTRY.get(
                    type(model), {}
                )
                if kernel_layer_name not in fusion_patterns:
                    raise ValueError(
                        f"{type(model).__name__} does not define fusion patterns for '{kernel_layer_name}'. "
                        f'Either add `_kernel_fusion_patterns = {{"{kernel_layer_name}": [...]}}` to the model class, '
                        f"call `register_fusion_patterns({type(model).__name__}, ...)` before loading the model, "
                        f"or use the inline pattern format: "
                        f'(("{kernel_layer_name}", "<glob_path>"), ...).'
                    )
                fuse_modules(model, fusion_patterns[kernel_layer_name], kernel_layer_name)
            new_mapping[kernel_layer_name] = kernel

        self.kernel_mapping = new_mapping

    def store_registered_layer_names(self, model):
        for name, module in model.named_modules():
            if hasattr(module, "kernel_layer_name"):
                self.registered_layer_names[name] = module.kernel_layer_name

    def sanitize_kernel_mapping(self, model):
        """
        Validates the kernel_mapping to ensure that:
        1. Each layer_name in the mapping is registered in the model (i.e., the model contains a module with a matching kernel_layer_name).
        2. Each kernel value is either a string of the form 'org/repo:layer_name' or a dict mapping device types ("cuda", "rocm", "xpu", "npu") to such strings.
        3. Each device key in a dict is one of "cuda", "rocm", "xpu", or "npu".
        4. Each repo_name is a valid repository and layer name in the format 'org/repo:layer_name' (i.e., a string containing both a slash and a colon).
        5. If a local path is detected, it should be in the format '/abs/path:layer_name'. The absolute path must include the `package_name`, like "/home/user/layer_norm".

        Args:
            model: The model instance whose modules are checked for registered kernel_layer_name attributes.

        Raises:
            ValueError: If a layer_name is not registered in the model, if a device is not supported,
                        or if a repo_name is not a valid 'org/repo:layer_name' string.
        """
        MAPPING_FORMAT = """
        For single device form remote
        {
            "RMSNorm":
                "kernels-community/layer_norm:LlamaRMSNorm",
            ...
        },
        For multiple devices form remote
        {
            "RMSNorm": {
                "cuda":
                    "kernels-community/layer_norm:LlamaRMSNorm",
                "rocm":
                    "kernels-community/layer_norm:LlamaRMSNorm",
                ...
            },
            ...
        }
        For single device form local
        {
            "RMSNorm":
                "/abs/path:LlamaRMSNorm",
            ...
        },
        For multiple devices form local
        {
            "RMSNorm": {
                "cuda":
                    "/abs/path:LlamaRMSNorm",
                "rocm":
                    "/abs/path:LlamaRMSNorm",
                ...
            },
            ...
        }
        """
        self.store_registered_layer_names(model)
        # Validate that the kernel mapping is a dict
        if not isinstance(self.kernel_mapping, dict):
            raise ValueError(
                f"Kernel mapping must be a dict of the following format: {MAPPING_FORMAT}, got: {type(self.kernel_mapping)}"
            )

        for layer_name, kernel in self.kernel_mapping.items():
            if layer_name not in self.registered_layer_names.values():
                raise ValueError(
                    f"Layer {layer_name} is not registered in the model, please register it first using use_kernel_forward_from_hub"
                )

            if isinstance(kernel, str):
                if "/" not in kernel or ":" not in kernel:
                    raise ValueError(
                        f"Kernel mapping for '{layer_name}' must be a valid repo name with a layer name (e.g., 'org/repo:layer_name' or '/abs/path:layer_name'), got: {kernel}"
                    )

            elif isinstance(kernel, dict):
                for device, repo_name in kernel.items():
                    if device not in ["cuda", "rocm", "xpu", "npu", "neuron"]:
                        raise ValueError(f"Only cuda, rocm, xpu, npu and neuron devices supported, got: {device}")

                    if not isinstance(repo_name, str) or "/" not in repo_name or ":" not in repo_name:
                        raise ValueError(
                            f"Kernel mapping for '{layer_name}' must be a valid repo name with a layer name (e.g., 'org/repo:layer_name' or '/abs/path:layer_name'), got: {repo_name}"
                        )
            else:
                raise ValueError(f"Kernel mapping must follow the format: {MAPPING_FORMAT}, got: {kernel}")

    def create_compatible_mapping(self, model, compile=False):
        """
        Transforms a simple kernel_mapping of the form:
            {
                "RMSNorm":
                    "kernels-community/layer_norm:LlamaRMSNorm",
                ...
            },

            or for local path:

            {
                "RMSNorm":
                    "/home/user/liger_kernels:LigerRMSNorm",
                ...
            },

        into a nested mapping:

            {
                "RMSNorm": {
                    "cuda": {
                        Mode.INFERENCE: LayerRepository(
                            repo_id="kernels-community/layer_norm",
                            layer_name="LlamaRMSNorm",
                        )
                    }
                }
            }

            or for local path:

            {
                "RMSNorm": {
                    "cuda": {
                        Mode.INFERENCE: LocalLayerRepository(
                            repo_path=Path("/home/user/liger_kernels"),
                            package_name="liger_kernels",
                            layer_name="LigerRMSNorm",
                        )
                    }
                }
            }

        that's compatible with the kernels library.

        The device is inferred from the model's parameters if not provided.
        The Mode is inferred from the model's training state.
        """
        from kernels import Mode

        compatible_mapping = {}
        current_device = infer_device(model)
        for layer_name, kernel in self.kernel_mapping.items():
            # Infer Mode: use Mode.TRAINING if model is training, else use Mode.INFERENCE
            mode = Mode.TRAINING if model.training else Mode.INFERENCE
            if compile:
                mode = mode | Mode.TORCH_COMPILE

            if isinstance(kernel, str):
                repo_name = kernel
                if not self.use_local_kernel:
                    add_to_mapping(layer_name, current_device, repo_name, mode, compatible_mapping)
                else:
                    add_to_mapping_local(layer_name, current_device, repo_name, mode, compatible_mapping)
            elif isinstance(kernel, dict):
                for device, repo_name in kernel.items():
                    if device != current_device:
                        continue
                    if not self.use_local_kernel:
                        add_to_mapping(layer_name, device, repo_name, mode, compatible_mapping)
                    else:
                        add_to_mapping_local(layer_name, device, repo_name, mode, compatible_mapping)

        self.kernel_mapping = compatible_mapping
