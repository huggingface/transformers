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

class KernelsConfig:
    def __init__(self, kernel_mapping):
        self.kernel_mapping = kernel_mapping

    def update_kernel(self, repo_id, layer_name, device = "cuda", mode = Mode.INFERENCE, revision=None):
        self.kernel_mapping[layer_name] = {
            device: {
                mode: LayerRepository(
                    repo_id=repo_id,
                    layer_name=layer_name,
                    revision=revision,
                )
            }
        }
        return self.kernel_mapping

    def update_kernel_mapping(self, new_kernel_mapping):
        self.kernel_mapping = new_kernel_mapping
        return self.kernel_mapping

