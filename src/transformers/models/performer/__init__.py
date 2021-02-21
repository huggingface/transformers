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

from ...file_utils import _BaseLazyModule, is_tf_available, is_tokenizers_available, is_torch_available


if is_torch_available():
    _import_structure["modeling_distilbert_performer"] = [
        "DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DistilBertPerformerForMaskedLM",
        "DistilBertPerformerForMultipleChoice",
        "DistilBertPerformerForQuestionAnswering",
        "DistilBertPerformerForSequenceClassification",
        "DistilBertPerformerForTokenClassification",
        "DistilBertPerformerModel",
        "DistilBertPerformerPreTrainedModel",
    ]

    _import_structure["modeling_bert_performer"] = [
        "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BertPerformerForMaskedLM",
        "BertPerformerForMultipleChoice",
        "BertPerformerForQuestionAnswering",
        "BertPerformerForSequenceClassification",
        "BertPerformerForTokenClassification",
        "BertPerformerModel",
        "BertPerformerPreTrainedModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_distilbert_performer"] = [
        "TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDistilBertPerformerForMaskedLM",
        "TFDistilBertPerformerForMultipleChoice",
        "TFDistilBertPerformerForQuestionAnswering",
        "TFDistilBertPerformerForSequenceClassification",
        "TFDistilBertPerformerForTokenClassification",
        "TFDistilBertPerformerMainLayer",
        "TFDistilBertPerformerModel",
        "TFDistilBertPerformerPreTrainedModel",
    ]

    _import_structure["modeling_tf_bert_performer"] = [
        "TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFBertPerformerForMaskedLM",
        "TFBertPerformerForMultipleChoice",
        "TFBertPerformerForQuestionAnswering",
        "TFBertPerformerForSequenceClassification",
        "TFBertPerformerForTokenClassification",
        "TFBertPerformerMainLayer",
        "TFBertPerformerModel",
        "TFBertPerformerPreTrainedModel",
    ]


if TYPE_CHECKING:
    from distilbert.configuration_distilbert_performer import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertPerformerConfig
    from bert.configuration_bert_performer import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertPerformerConfig

    from ..distilbert.tokenization_distilbert import DistilBertTokenizer
    from ..bert.tokenization_bert import BertTokenizer

    if is_tokenizers_available():
        from ..distilbert.tokenization_distilbert_fast import DistilBertTokenizerFast
        from ..bert.tokenization_bert_fast import BertTokenizerFast

    if is_torch_available():
        from distilbert.modeling_distilbert_performer import (
            DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DistilBertPerformerForMaskedLM,
            DistilBertPerformerForMultipleChoice,
            DistilBertPerformerForQuestionAnswering,
            DistilBertPerformerForSequenceClassification,
            DistilBertPerformerForTokenClassification,
            DistilBertPerformerModel,
            DistilBertPerformerPreTrainedModel,
        )
        from bert.modeling_distilbert_performer import (
            BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BertPerformerForMaskedLM,
            BertPerformerForMultipleChoice,
            BertPerformerForQuestionAnswering,
            BertPerformerForSequenceClassification,
            BertPerformerForTokenClassification,
            BertPerformerModel,
            BertPerformerPreTrainedModel,
        )

    if is_tf_available():
        from distilbert.modeling_tf_distilbert_performer import (
            TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDistilBertPerformerForMaskedLM,
            TFDistilBertPerformerForMultipleChoice,
            TFDistilBertPerformerForQuestionAnswering,
            TFDistilBertPerformerForSequenceClassification,
            TFDistilBertPerformerForTokenClassification,
            TFDistilBertPerformerMainLayer,
            TFDistilBertPerformerModel,
            TFDistilBertPerformerPreTrainedModel,
        )
        from bert.modeling_tf_bert_performer import (
            TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFBertPerformerForMaskedLM,
            TFBertPerformerForMultipleChoice,
            TFBertPerformerForQuestionAnswering,
            TFBertPerformerForSequenceClassification,
            TFBertPerformerForTokenClassification,
            TFBertPerformerMainLayer,
            TFBertPerformerModel,
            TFBertPerformerPreTrainedModel,
        )

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
