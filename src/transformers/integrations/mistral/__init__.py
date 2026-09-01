# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

"""Mistral native format integration: tokenizer and config conversion utilities."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule


_import_structure = {
    "native_config": [
        "MistralNativeConfig",
        "mistral_native_config_from_params",
    ],
    "params_conversion": [
        "mistral_native_config_from_hf_config",
        "mistral_native_config_to_hf_config",
    ],
    "tokenizer": [
        "MistralConverter",
        "convert_tekken_tokenizer",
        "resolve_mistral_format",
    ],
}

if TYPE_CHECKING:
    from .native_config import (
        MistralNativeConfig,
        mistral_native_config_from_params,
    )
    from .params_conversion import mistral_native_config_from_hf_config, mistral_native_config_to_hf_config
    from .tokenizer import MistralConverter, convert_tekken_tokenizer, resolve_mistral_format
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
