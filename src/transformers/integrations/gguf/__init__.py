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
from typing import TYPE_CHECKING

from ...utils import _LazyModule


_import_structure = {
    "gguf_config_mapping": ["GGUF_CONFIG_ARCHS", "get_gguf_config"],
    "gguf_conversion_mapping": ["GGUF_ARCHS"],
    "gguf_tokenizer_mapping": ["GGUF_TOKENIZER_MAPPING", "convert_gguf_tokenizer", "get_gguf_tokenizer"],
    "reader": ["GgufHeader", "load_gguf_state_dict", "read_gguf_metadata"],
    "utils": [
        "get_gguf_conversion_mapping",
        "get_gguf_plan",
        "is_gguf_arch_supported",
        "replace_with_gguf_modules",
    ],
}

if TYPE_CHECKING:
    from .gguf_config_mapping import GGUF_CONFIG_ARCHS, get_gguf_config
    from .gguf_conversion_mapping import GGUF_ARCHS
    from .gguf_tokenizer_mapping import GGUF_TOKENIZER_MAPPING, convert_gguf_tokenizer, get_gguf_tokenizer
    from .reader import GgufHeader, load_gguf_state_dict, read_gguf_metadata
    from .utils import (
        get_gguf_conversion_mapping,
        get_gguf_plan,
        is_gguf_arch_supported,
        replace_with_gguf_modules,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
