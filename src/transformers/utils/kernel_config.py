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


def add_to_mapping(
    layer_name, device, repo_name, mode, compatible_mapping, version=1, revision=None, trust_remote_code=False
):
    from kernels import LayerRepository

    if device not in ["cuda", "rocm", "xpu", "npu", "neuron", "tpu"]:
        raise ValueError(f"Only cuda, rocm, xpu, npu, neuron and tpu devices supported, got: {device}")
    repo_layer_name = repo_name.split(":")[1]
    repo_id = repo_name.split(":")[0]
    compatible_mapping[layer_name] = {
        device: {
            mode: LayerRepository(
                repo_id=repo_id,
                layer_name=repo_layer_name,
                version=version,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
        }
    }


def add_to_mapping_local(layer_name, device, repo_name, mode, compatible_mapping):
    from pathlib import Path

    from kernels import LocalLayerRepository

    if device not in ["cuda", "rocm", "xpu", "npu", "neuron", "tpu"]:
        raise ValueError(f"Only cuda, rocm, xpu, npu, neuron and tpu devices supported, got: {device}")
    repo_layer_name = repo_name.split(":")[1]
    repo_path = repo_name.split(":")[0]
    compatible_mapping[layer_name] = {
        device: {
            mode: LocalLayerRepository(
                repo_path=Path(repo_path),
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

    def update_kernel(
        self, repo_id, registered_name, layer_name, device, mode, revision=None, version=1, trust_remote_code=False
    ):
        from kernels import LayerRepository

        self.kernel_mapping[registered_name] = {
            device: {
                mode: LayerRepository(
                    repo_id=repo_id,
                    layer_name=layer_name,
                    revision=revision,
                    version=version,
                    trust_remote_code=trust_remote_code,
                )
            }
        }

    def store_registered_layer_names(self, model):
        for name, module in model.named_modules():
            if hasattr(module, "kernel_layer_name"):
                self.registered_layer_names[name] = module.kernel_layer_name

    def sanitize_kernel_mapping(self, model):
        """
        Validates the kernel_mapping to ensure that:
        1. Each layer_name in the mapping is registered in the model (i.e., the model contains a module with a matching kernel_layer_name).
        2. Each kernel value is
            - either a string of the form 'org/repo:layer_name' or a tuple with the same as string and a dict of {"revision"/"version/trust_remote_code": ...},
            - or a dict mapping device types ("cuda", "rocm", "xpu", "npu") to such values as above.
        3. Each device key in a dict is one of "cuda", "rocm", "xpu", or "npu".
        5. Each trust remote code key must be a bool.
        6. Each revision or version key must exist mutually exclusive if it has been passed explicitly.
        7. Each repo_name is a valid repository and layer name in the format 'org/repo:layer_name' (i.e., a string containing both a slash and a colon).
        8. If a local path is detected, it should be in the format '/abs/path:layer_name', where the absolute path points to the kernel repository, like "/home/user/layer_norm".

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
        You can also pass metadata along to inform about specific kernel information
        {
            "RMSNorm":
                ("kernels-community/layer_norm:LlamaRMSNorm", {"version": 1, "trust_remote_code": True}),
            ...
        },
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

            skip_device_check = False
            if isinstance(kernel, (str, tuple)):
                kernel = {None: kernel}
                skip_device_check = True

            if isinstance(kernel, dict):
                for device, repo in kernel.items():
                    repo_name = repo
                    if not skip_device_check and device not in ["cuda", "rocm", "xpu", "npu", "neuron", "tpu"]:
                        raise ValueError(f"Only cuda, rocm, xpu, npu, neuron and tpu devices supported, got: {device}")

                    # Check case where metadata (revision/version/trust_remote_code) is explicitly passed
                    if isinstance(repo, tuple):
                        repo_name, metadata = repo

                        if not isinstance(metadata, dict):
                            raise ValueError(
                                "The passed metadata as second entry in a tuple needs to be a dict but found: "
                                f"{type(metadata) = } for {metadata}."
                            )

                        if (trust_remote_code := metadata.get("trust_remote_code", None)) is not None:
                            if not isinstance(trust_remote_code, bool):
                                raise ValueError(
                                    f"Expected a bool value for `trust_remote_code` but got {trust_remote_code}"
                                )

                        if (revision := metadata.get("revision", None)) is None and (
                            version := metadata.get("version", None)
                        ) is None:
                            raise ValueError(
                                "Expected valid combination for version/revision (mutually exclusive but one of them) "
                                f"to be passed when passed as tuple, but got {revision= } and {version= }"
                            )

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
                    ("kernels-community/layer_norm:LlamaRMSNorm", {"version": 1, "trust_remote_code": True}),
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
                            version=1,
                            trust_remote_code=True,
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

            if isinstance(kernel, (str, tuple)):
                kernel = {current_device: kernel}

            for device, repo in kernel.items():
                if device != current_device:
                    continue

                if self.use_local_kernel:
                    add_to_mapping_local(layer_name, device, repo, mode, compatible_mapping)
                    continue

                # Infer metadata (revision/version/trust_remote_code)
                if isinstance(repo, tuple):
                    repo_name, metadata = repo
                    revision = metadata.get("revision", None)
                    version = metadata.get("version", None)
                    trust_remote_code = metadata.get("trust_remote_code", False)
                else:
                    repo_name = repo
                    revision = None
                    version = 1
                    trust_remote_code = False

                add_to_mapping(
                    layer_name,
                    device=device,
                    repo_name=repo_name,
                    mode=mode,
                    compatible_mapping=compatible_mapping,
                    version=version,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                )

        self.kernel_mapping = compatible_mapping
