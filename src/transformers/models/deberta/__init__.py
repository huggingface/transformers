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


_import_structure = {
    "configuration_deberta": ["DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "DebertaConfig"],
    "tokenization_deberta": ["DebertaTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_deberta_fast"] = ["DebertaTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_deberta"] = [
        "DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DebertaForMaskedLM",
        "DebertaForQuestionAnswering",
        "DebertaForSequenceClassification",
        "DebertaForTokenClassification",
        "DebertaModel",
        "DebertaPreTrainedModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_deberta"] = [
        "TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDebertaForMaskedLM",
        "TFDebertaForQuestionAnswering",
        "TFDebertaForSequenceClassification",
        "TFDebertaForTokenClassification",
        "TFDebertaModel",
        "TFDebertaPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaConfig
    from .tokenization_deberta import DebertaTokenizer

    if is_tokenizers_available():
        from .tokenization_deberta_fast import DebertaTokenizerFast

    if is_torch_available():
        from .modeling_deberta import (
            DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            DebertaForMaskedLM,
            DebertaForQuestionAnswering,
            DebertaForSequenceClassification,
            DebertaForTokenClassification,
            DebertaModel,
            DebertaPreTrainedModel,
        )

    if is_tf_available():
        from .modeling_tf_deberta import (
            TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDebertaForMaskedLM,
            TFDebertaForQuestionAnswering,
            TFDebertaForSequenceClassification,
            TFDebertaForTokenClassification,
            TFDebertaModel,
            TFDebertaPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
