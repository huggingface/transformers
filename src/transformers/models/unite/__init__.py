# Copyright 2022 The HuggingFace Team. All rights reserved.
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
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_unite": [
        "UNITE_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "UniTEConfig",
    ],
}

if is_sentencepiece_available():
    _import_structure["tokenization_unite"] = ["UniTETokenizer"]

if is_tokenizers_available():
    _import_structure["tokenization_unite_fast"] = ["UniTETokenizerFast"]

if is_torch_available():
    _import_structure["modeling_unite"] = [
        "UNITE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "UniTEForSequenceClassification",
        "UniTEModel",
    ]

if TYPE_CHECKING:
    from .configuration_unite import (
        UNITE_PRETRAINED_CONFIG_ARCHIVE_MAP,
        UniTEConfig,
    )

    if is_sentencepiece_available():
        from .tokenization_unite import UniTETokenizer

    if is_tokenizers_available():
        from .tokenization_unite_fast import UniTETokenizerFast

    if is_torch_available():
        from .modeling_unite import (
            UNITE_PRETRAINED_MODEL_ARCHIVE_LIST,
            UniTEForSequenceClassification,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
