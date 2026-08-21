# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from .gguf_config_mapping import GGUF_CONFIG_ARCHS, get_gguf_config
from .gguf_conversion_mapping import GGUF_ARCHS
from .reader import (
    GgufHeader,
    load_gguf_state_dict,
    read_gguf_metadata,
)
from .utils import (
    get_gguf_conversion_mapping,
    get_gguf_plan,
    is_gguf_arch_supported,
    replace_with_gguf_modules,
)


__all__ = [
    "GGUF_ARCHS",
    "GGUF_CONFIG_ARCHS",
    "get_gguf_config",
    "get_gguf_conversion_mapping",
    "get_gguf_plan",
    "is_gguf_arch_supported",
    "load_gguf_state_dict",
    "read_gguf_metadata",
    "GgufHeader",
    "replace_with_gguf_modules",
]
