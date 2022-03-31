# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from ...utils import (
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {}

if is_sentencepiece_available():
    _import_structure["tokenization_layoutxlm"] = ["LayoutXLMTokenizer"]

if is_tokenizers_available():
    _import_structure["tokenization_layoutxlm_fast"] = ["LayoutXLMTokenizerFast"]

if is_vision_available():
    _import_structure["processing_layoutxlm"] = ["LayoutXLMProcessor"]

if TYPE_CHECKING:
    if is_sentencepiece_available():
        from .tokenization_layoutxlm import LayoutXLMTokenizer

    if is_tokenizers_available():
        from .tokenization_layoutxlm_fast import LayoutXLMTokenizerFast

    if is_vision_available():
        from .processing_layoutlmv2 import LayoutXLMProcessor

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
