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

from ...file_utils import _LazyModule, is_flax_available, is_tf_available, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_electra": ["ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP", "ElectraConfig", "ElectraOnnxConfig"],
    "tokenization_electra": ["ElectraTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_electra_fast"] = ["ElectraTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_electra"] = [
        "ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ElectraForCausalLM",
        "ElectraForMaskedLM",
        "ElectraForMultipleChoice",
        "ElectraForPreTraining",
        "ElectraForQuestionAnswering",
        "ElectraForSequenceClassification",
        "ElectraForTokenClassification",
        "ElectraModel",
        "ElectraPreTrainedModel",
        "load_tf_weights_in_electra",
    ]

if is_tf_available():
    _import_structure["modeling_tf_electra"] = [
        "TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFElectraForMaskedLM",
        "TFElectraForMultipleChoice",
        "TFElectraForPreTraining",
        "TFElectraForQuestionAnswering",
        "TFElectraForSequenceClassification",
        "TFElectraForTokenClassification",
        "TFElectraModel",
        "TFElectraPreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_electra"] = [
        "FlaxElectraForMaskedLM",
        "FlaxElectraForMultipleChoice",
        "FlaxElectraForPreTraining",
        "FlaxElectraForQuestionAnswering",
        "FlaxElectraForSequenceClassification",
        "FlaxElectraForTokenClassification",
        "FlaxElectraModel",
        "FlaxElectraPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig, ElectraOnnxConfig
    from .tokenization_electra import ElectraTokenizer

    if is_tokenizers_available():
        from .tokenization_electra_fast import ElectraTokenizerFast

    if is_torch_available():
        from .modeling_electra import (
            ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
            ElectraForCausalLM,
            ElectraForMaskedLM,
            ElectraForMultipleChoice,
            ElectraForPreTraining,
            ElectraForQuestionAnswering,
            ElectraForSequenceClassification,
            ElectraForTokenClassification,
            ElectraModel,
            ElectraPreTrainedModel,
            load_tf_weights_in_electra,
        )

    if is_tf_available():
        from .modeling_tf_electra import (
            TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFElectraForMaskedLM,
            TFElectraForMultipleChoice,
            TFElectraForPreTraining,
            TFElectraForQuestionAnswering,
            TFElectraForSequenceClassification,
            TFElectraForTokenClassification,
            TFElectraModel,
            TFElectraPreTrainedModel,
        )

    if is_flax_available():
        from .modeling_flax_electra import (
            FlaxElectraForMaskedLM,
            FlaxElectraForMultipleChoice,
            FlaxElectraForPreTraining,
            FlaxElectraForQuestionAnswering,
            FlaxElectraForSequenceClassification,
            FlaxElectraForTokenClassification,
            FlaxElectraModel,
            FlaxElectraPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
