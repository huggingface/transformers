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
from ..utils import is_kernels_available
from ..integrations.hub_kernels import _KERNEL_MAPPING

if is_kernels_available():
    from kernels import LayerRepository, Mode

class KernelConfig:
    def __init__(self, kernel_mapping = {}):
        self.kernel_mapping = kernel_mapping

    def update_kernel(self, model,repo_id, layer_name, device = "cuda", mode = Mode.INFERENCE, revision=None):
        if self.check_if_layer_name_registered(layer_name, model):
            self.kernel_mapping[layer_name] = {
                device: {
                    mode: LayerRepository(
                        repo_id=repo_id,
                        layer_name=layer_name,
                        revision=revision,
                    )
                }
            }
        else:
            raise ValueError(f"Layer {layer_name} is not registered in the model")

        return self.kernel_mapping

    def check_if_layer_name_registered(self, layer_name, model):
        for name, module in model.named_modules():
            if hasattr(module, "kernel_layer_name") and module.kernel_layer_name == layer_name:
                return True
        return False

    def update_all_kernel_mapping(self, model,new_kernel_mapping):
        for layer_name, mapping in new_kernel_mapping.items():
            if self.check_if_layer_name_registered(layer_name, model):
                self.kernel_mapping[layer_name] = mapping
            else:
                raise ValueError(f"Layer {layer_name} is not registered in the model")
        return self.kernel_mapping

    def check_if_layer_name_in_global_mapping(self, layer_name):
        return layer_name in _KERNEL_MAPPING.keys()

    def sanitize_kernel_mapping(self, model):
        for layer_name, mapping in self.kernel_mapping.items():
            if not self.check_if_layer_name_registered(layer_name, model):
                raise ValueError(f"Layer {layer_name} is not registered in the model")
            if not isinstance(mapping, dict):
                raise ValueError(f"Mapping for layer {layer_name} must be a dict")
            for device, mode_dict in mapping.items():
                if device not in ["cuda", "rocm", "xpu"]:
                    raise ValueError(f"Device {device} is not supported")
                if not isinstance(mode_dict, dict):
                    raise ValueError(f"Device mapping for {device} in layer {layer_name} must be a dict")
                for mode, repo in mode_dict.items():
                    valid_modes = [
                        Mode.INFERENCE,
                        Mode.TRAINING,
                        Mode.TORCH_COMPILE,
                        Mode.TRAINING | Mode.TORCH_COMPILE,
                        Mode.INFERENCE | Mode.TORCH_COMPILE,
                        Mode.FALLBACK,
                    ]
                    if mode not in valid_modes:
                        raise ValueError(f"Mode {mode} is not supported")
                    # Check that the value is a LayerRepository
                    if is_kernels_available():
                        from kernels import LayerRepository
                        if not isinstance(repo, LayerRepository):
                            raise ValueError(
                                f"Value for {layer_name} -> {device} -> {mode} must be a LayerRepository instance"
                            )