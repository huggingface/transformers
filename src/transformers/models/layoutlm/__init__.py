# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from ...file_utils import _LazyModule, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig
from .tokenization_layoutlm import LayoutLMTokenizer


_import_structure = {
    "configuration_layoutlm": ["LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "LayoutLMConfig", "LayoutLMOnnxConfig"],
    "tokenization_layoutlm": ["LayoutLMTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_layoutlm_fast"] = ["LayoutLMTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_layoutlm"] = [
        "LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LayoutLMForMaskedLM",
        "LayoutLMForSequenceClassification",
        "LayoutLMForTokenClassification",
        "LayoutLMModel",
        "LayoutLMPreTrainedModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_layoutlm"] = [
        "TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFLayoutLMForMaskedLM",
        "TFLayoutLMForSequenceClassification",
        "TFLayoutLMForTokenClassification",
        "TFLayoutLMMainLayer",
        "TFLayoutLMModel",
        "TFLayoutLMPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig, LayoutLMOnnxConfig
    from .tokenization_layoutlm import LayoutLMTokenizer

    if is_tokenizers_available():
        from .tokenization_layoutlm_fast import LayoutLMTokenizerFast

    if is_torch_available():
        from .modeling_layoutlm import (
            LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            LayoutLMForMaskedLM,
            LayoutLMForSequenceClassification,
            LayoutLMForTokenClassification,
            LayoutLMModel,
            LayoutLMPreTrainedModel,
        )
    if is_tf_available():
        from .modeling_tf_layoutlm import (
            TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFLayoutLMForMaskedLM,
            TFLayoutLMForSequenceClassification,
            TFLayoutLMForTokenClassification,
            TFLayoutLMMainLayer,
            TFLayoutLMModel,
            TFLayoutLMPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
