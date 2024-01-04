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

from ...utils import  _LazyModule, OptionalDependencyNotAvailable, is_tokenizers_available
from ...utils import is_torch_available




_import_structure = {
    "configuration_narrow_bert": ["NARROW_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "NarrowBertConfig"],
    "tokenization_narrow_bert": ["NarrowBertTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_narrow_bert_fast"] = ["NarrowBertTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_narrow_bert"] = [
        "NARROW_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "NarrowBertForMaskedLM",
        "NarrowBertForMultipleChoice",
        "NarrowBertForSequenceClassification",
        "NarrowBertForTokenClassification",
        "NarrowBertLayer",
        "NarrowBertModel",
        "NarrowBertPreTrainedModel",
        "load_tf_weights_in_narrow_bert",
    ]




if TYPE_CHECKING:
    from .configuration_narrow_bert import NARROW_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, NarrowBertConfig
    from .tokenization_narrow_bert import NarrowBertTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_narrow_bert_fast import NarrowBertTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_narrow_bert import (
            NARROW_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            NarrowBertForMaskedLM,
            NarrowBertForMultipleChoice,
            NarrowBertForSequenceClassification,
            NarrowBertForTokenClassification,
            NarrowBertLayer,
            NarrowBertModel,
            NarrowBertPreTrainedModel,
            load_tf_weights_in_narrow_bert,
        )



else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
