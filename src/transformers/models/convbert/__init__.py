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
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_convbert": ["CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvBertConfig", "ConvBertOnnxConfig"],
    "tokenization_convbert": ["ConvBertTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_convbert_fast"] = ["ConvBertTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_convbert"] = [
        "CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ConvBertForMaskedLM",
        "ConvBertForMultipleChoice",
        "ConvBertForQuestionAnswering",
        "ConvBertForSequenceClassification",
        "ConvBertForTokenClassification",
        "ConvBertLayer",
        "ConvBertModel",
        "ConvBertPreTrainedModel",
        "load_tf_weights_in_convbert",
    ]


try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_convbert"] = [
        "TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFConvBertForMaskedLM",
        "TFConvBertForMultipleChoice",
        "TFConvBertForQuestionAnswering",
        "TFConvBertForSequenceClassification",
        "TFConvBertForTokenClassification",
        "TFConvBertLayer",
        "TFConvBertModel",
        "TFConvBertPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_convbert import CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvBertConfig, ConvBertOnnxConfig
    from .tokenization_convbert import ConvBertTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_convbert_fast import ConvBertTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_convbert import (
            CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ConvBertForMaskedLM,
            ConvBertForMultipleChoice,
            ConvBertForQuestionAnswering,
            ConvBertForSequenceClassification,
            ConvBertForTokenClassification,
            ConvBertLayer,
            ConvBertModel,
            ConvBertPreTrainedModel,
            load_tf_weights_in_convbert,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_convbert import (
            TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFConvBertForMaskedLM,
            TFConvBertForMultipleChoice,
            TFConvBertForQuestionAnswering,
            TFConvBertForSequenceClassification,
            TFConvBertForTokenClassification,
            TFConvBertLayer,
            TFConvBertModel,
            TFConvBertPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
