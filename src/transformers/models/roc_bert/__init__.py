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

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_roc_bert": ["ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RoCBertConfig"],
    "tokenization_roc_bert": ["RoCBertTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    pass

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_roc_bert"] = [
        "ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RoCBertForCausalLM",
        "RoCBertForMaskedLM",
        "RoCBertForMultipleChoice",
        "RoCBertForPreTraining",
        "RoCBertForQuestionAnswering",
        "RoCBertForSequenceClassification",
        "RoCBertForTokenClassification",
        "RoCBertLayer",
        "RoCBertModel",
        "RoCBertPreTrainedModel",
        "load_tf_weights_in_roc_bert",
    ]

if TYPE_CHECKING:
    from .configuration_roc_bert import ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RoCBertConfig
    from .tokenization_roc_bert import RoCBertTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        raise OptionalDependencyNotAvailable()

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_roc_bert import (
            ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            RoCBertForCausalLM,
            RoCBertForMaskedLM,
            RoCBertForMultipleChoice,
            RoCBertForPreTraining,
            RoCBertForQuestionAnswering,
            RoCBertForSequenceClassification,
            RoCBertForTokenClassification,
            RoCBertLayer,
            RoCBertModel,
            RoCBertPreTrainedModel,
            load_tf_weights_in_roc_bert,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
