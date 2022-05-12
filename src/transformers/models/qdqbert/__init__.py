# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021 NVIDIA Corporation and The HuggingFace Team. All rights reserved.
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

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_qdqbert": ["QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "QDQBertConfig"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_qdqbert"] = [
        "QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "QDQBertForMaskedLM",
        "QDQBertForMultipleChoice",
        "QDQBertForNextSentencePrediction",
        "QDQBertForQuestionAnswering",
        "QDQBertForSequenceClassification",
        "QDQBertForTokenClassification",
        "QDQBertLayer",
        "QDQBertLMHeadModel",
        "QDQBertModel",
        "QDQBertPreTrainedModel",
        "load_tf_weights_in_qdqbert",
    ]


if TYPE_CHECKING:
    from .configuration_qdqbert import QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, QDQBertConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_qdqbert import (
            QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            QDQBertForMaskedLM,
            QDQBertForMultipleChoice,
            QDQBertForNextSentencePrediction,
            QDQBertForQuestionAnswering,
            QDQBertForSequenceClassification,
            QDQBertForTokenClassification,
            QDQBertLayer,
            QDQBertLMHeadModel,
            QDQBertModel,
            QDQBertPreTrainedModel,
            load_tf_weights_in_qdqbert,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
