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
from typing import Any


def get_module_from_name(module, tensor_name: str) -> tuple[Any, str]:
    if "." in tensor_name:
        module_name, tensor_name = tensor_name.rsplit(".", 1)
        module = module.get_submodule(module_name)
    return module, tensor_name


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
