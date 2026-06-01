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
import re

from torch.nn import Module


def get_module_from_name(module: Module, tensor_name: str) -> tuple[Module, str]:
    """Split the tensor name into the module its from and the name itself."""
    possible_modules = tensor_name.split(".")
    current_module = module

    # Iterate through the list of possible modules,
    # checking that the next possible sub-module is an attribute of the current module
    for i, part in enumerate(possible_modules):
        # Check if the next segment exists and is a Module
        next_attribute = getattr(current_module, part, None)

        if isinstance(next_attribute, Module):
            current_module = next_attribute
        else:
            # We hit a non-module (Parameter, Buffer, or nested attribute)
            # Everything from this point forward is the parameter name
            param_name = ".".join(possible_modules[i:])
            return current_module, param_name

    return current_module, ""


def should_convert_module(full_name, patterns: list[str] | None = None):
    if patterns is None:
        return True

    # We should avoid converting in the following situations:
    # 1. The pattern appears as a prefix followed by a dot in `full_name`
    #    (e.g., "model.decoder.layer.11." matches "model.decoder.layer.11.attn.weight").
    # 2. The pattern matches `full_name` exactly or via regex
    #    (e.g., "lm_head" matches "lm_head"; "model.decoder.layer.*" matches "model.decoder.layer.11.attn.weight").
    # 3. `full_name` ends with the pattern
    #    (e.g., "fc1" matches "model.decoder.layers.23.fc1").

    should_not_convert = any(
        re.match(f"{key}\\.", full_name) or re.match(f"{key}", full_name) or full_name.endswith(key)
        for key in patterns
    )
    return not should_not_convert
