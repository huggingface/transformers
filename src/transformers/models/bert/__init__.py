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

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_bert": ["BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BertConfig", "BertOnnxConfig"],
    "tokenization_bert": ["BasicTokenizer", "BertTokenizer", "WordpieceTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_bert_fast"] = ["BertTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_bert"] = [
        "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BertForMaskedLM",
        "BertForMultipleChoice",
        "BertForNextSentencePrediction",
        "BertForPreTraining",
        "BertForQuestionAnswering",
        "BertForSequenceClassification",
        "BertForTokenClassification",
        "BertLayer",
        "BertLMHeadModel",
        "BertModel",
        "BertPreTrainedModel",
        "load_tf_weights_in_bert",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_bert"] = [
        "TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFBertEmbeddings",
        "TFBertForMaskedLM",
        "TFBertForMultipleChoice",
        "TFBertForNextSentencePrediction",
        "TFBertForPreTraining",
        "TFBertForQuestionAnswering",
        "TFBertForSequenceClassification",
        "TFBertForTokenClassification",
        "TFBertLMHeadModel",
        "TFBertMainLayer",
        "TFBertModel",
        "TFBertPreTrainedModel",
    ]
try:
    if not is_tensorflow_text_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_bert_tf"] = ["TFBertTokenizer"]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_bert"] = [
        "FlaxBertForCausalLM",
        "FlaxBertForMaskedLM",
        "FlaxBertForMultipleChoice",
        "FlaxBertForNextSentencePrediction",
        "FlaxBertForPreTraining",
        "FlaxBertForQuestionAnswering",
        "FlaxBertForSequenceClassification",
        "FlaxBertForTokenClassification",
        "FlaxBertModel",
        "FlaxBertPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig, BertOnnxConfig
    from .tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_bert_fast import BertTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_bert import (
            BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BertForMaskedLM,
            BertForMultipleChoice,
            BertForNextSentencePrediction,
            BertForPreTraining,
            BertForQuestionAnswering,
            BertForSequenceClassification,
            BertForTokenClassification,
            BertLayer,
            BertLMHeadModel,
            BertModel,
            BertPreTrainedModel,
            load_tf_weights_in_bert,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_bert import (
            TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFBertEmbeddings,
            TFBertForMaskedLM,
            TFBertForMultipleChoice,
            TFBertForNextSentencePrediction,
            TFBertForPreTraining,
            TFBertForQuestionAnswering,
            TFBertForSequenceClassification,
            TFBertForTokenClassification,
            TFBertLMHeadModel,
            TFBertMainLayer,
            TFBertModel,
            TFBertPreTrainedModel,
        )

    try:
        if not is_tensorflow_text_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_bert_tf import TFBertTokenizer

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_bert import (
            FlaxBertForCausalLM,
            FlaxBertForMaskedLM,
            FlaxBertForMultipleChoice,
            FlaxBertForNextSentencePrediction,
            FlaxBertForPreTraining,
            FlaxBertForQuestionAnswering,
            FlaxBertForSequenceClassification,
            FlaxBertForTokenClassification,
            FlaxBertModel,
            FlaxBertPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
