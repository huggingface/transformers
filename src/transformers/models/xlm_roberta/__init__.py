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
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_xlm_roberta": [
        "XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "XLMRobertaConfig",
        "XLMRobertaOnnxConfig",
    ],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_xlm_roberta"] = ["XLMRobertaTokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_xlm_roberta_fast"] = ["XLMRobertaTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_xlm_roberta"] = [
        "XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XLMRobertaForCausalLM",
        "XLMRobertaForMaskedLM",
        "XLMRobertaForMultipleChoice",
        "XLMRobertaForQuestionAnswering",
        "XLMRobertaForSequenceClassification",
        "XLMRobertaForTokenClassification",
        "XLMRobertaModel",
        "XLMRobertaPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_xlm_roberta"] = [
        "TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFXLMRobertaForMaskedLM",
        "TFXLMRobertaForMultipleChoice",
        "TFXLMRobertaForQuestionAnswering",
        "TFXLMRobertaForSequenceClassification",
        "TFXLMRobertaForTokenClassification",
        "TFXLMRobertaModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_xlm_roberta"] = [
        "FlaxXLMRobertaForMaskedLM",
        "FlaxXLMRobertaForMultipleChoice",
        "FlaxXLMRobertaForQuestionAnswering",
        "FlaxXLMRobertaForSequenceClassification",
        "FlaxXLMRobertaForTokenClassification",
        "FlaxXLMRobertaModel",
    ]

if TYPE_CHECKING:
    from .configuration_xlm_roberta import (
        XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLMRobertaConfig,
        XLMRobertaOnnxConfig,
    )

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_xlm_roberta import XLMRobertaTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_xlm_roberta import (
            XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLMRobertaForCausalLM,
            XLMRobertaForMaskedLM,
            XLMRobertaForMultipleChoice,
            XLMRobertaForQuestionAnswering,
            XLMRobertaForSequenceClassification,
            XLMRobertaForTokenClassification,
            XLMRobertaModel,
            XLMRobertaPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_xlm_roberta import (
            TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFXLMRobertaForMaskedLM,
            TFXLMRobertaForMultipleChoice,
            TFXLMRobertaForQuestionAnswering,
            TFXLMRobertaForSequenceClassification,
            TFXLMRobertaForTokenClassification,
            TFXLMRobertaModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_xlm_roberta import (
            FlaxXLMRobertaForMaskedLM,
            FlaxXLMRobertaForMultipleChoice,
            FlaxXLMRobertaForQuestionAnswering,
            FlaxXLMRobertaForSequenceClassification,
            FlaxXLMRobertaForTokenClassification,
            FlaxXLMRobertaModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
