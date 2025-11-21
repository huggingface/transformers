# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import Any


def get_module_from_name(module, tensor_name: str) -> tuple[Any, str]:
    if "." in tensor_name:
        module_name, tensor_name = tensor_name.rsplit(".", 1)
        module = module.get_submodule(module_name)
    return module, tensor_name

def get_parameter_or_buffer(module, target: str):
    """
    Return the parameter or buffer given by `target` if it exists, otherwise throw an error. This combines
    `get_parameter()` and `get_buffer()` in a single handy function. If the target is an `_extra_state` attribute,
    it will return the extra state provided by the module. Note that it only work if `target` is a leaf of the model.
    """
    import torch
    try:
        return module.get_parameter(target)
    except AttributeError:
        pass
    try:
        return module.get_buffer(target)
    except AttributeError:
        pass
    module, param_name = get_module_from_name(module, target)
    if (
        param_name == "_extra_state"
        and getattr(module.__class__, "get_extra_state", torch.nn.Module.get_extra_state)
        is not torch.nn.Module.get_extra_state
    ):
        return module.get_extra_state()

    raise AttributeError(f"`{target}` is neither a parameter, buffer, nor extra state.")