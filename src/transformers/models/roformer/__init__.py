# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from ...utils import _LazyModule, is_flax_available, is_tf_available, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_roformer": ["ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "RoFormerConfig", "RoFormerOnnxConfig"],
    "tokenization_roformer": ["RoFormerTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_roformer_fast"] = ["RoFormerTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_roformer"] = [
        "ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RoFormerForCausalLM",
        "RoFormerForMaskedLM",
        "RoFormerForMultipleChoice",
        "RoFormerForQuestionAnswering",
        "RoFormerForSequenceClassification",
        "RoFormerForTokenClassification",
        "RoFormerLayer",
        "RoFormerModel",
        "RoFormerPreTrainedModel",
        "load_tf_weights_in_roformer",
    ]


if is_tf_available():
    _import_structure["modeling_tf_roformer"] = [
        "TF_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFRoFormerForCausalLM",
        "TFRoFormerForMaskedLM",
        "TFRoFormerForMultipleChoice",
        "TFRoFormerForQuestionAnswering",
        "TFRoFormerForSequenceClassification",
        "TFRoFormerForTokenClassification",
        "TFRoFormerLayer",
        "TFRoFormerModel",
        "TFRoFormerPreTrainedModel",
    ]


if is_flax_available():
    _import_structure["modeling_flax_roformer"] = [
        "FLAX_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FlaxRoFormerForMaskedLM",
        "FlaxRoFormerForMultipleChoice",
        "FlaxRoFormerForQuestionAnswering",
        "FlaxRoFormerForSequenceClassification",
        "FlaxRoFormerForTokenClassification",
        "FlaxRoFormerModel",
        "FlaxRoFormerPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_roformer import ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, RoFormerConfig, RoFormerOnnxConfig
    from .tokenization_roformer import RoFormerTokenizer

    if is_tokenizers_available():
        from .tokenization_roformer_fast import RoFormerTokenizerFast

    if is_torch_available():
        from .modeling_roformer import (
            ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            RoFormerForCausalLM,
            RoFormerForMaskedLM,
            RoFormerForMultipleChoice,
            RoFormerForQuestionAnswering,
            RoFormerForSequenceClassification,
            RoFormerForTokenClassification,
            RoFormerLayer,
            RoFormerModel,
            RoFormerPreTrainedModel,
            load_tf_weights_in_roformer,
        )

    if is_tf_available():
        from .modeling_tf_roformer import (
            TF_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRoFormerForCausalLM,
            TFRoFormerForMaskedLM,
            TFRoFormerForMultipleChoice,
            TFRoFormerForQuestionAnswering,
            TFRoFormerForSequenceClassification,
            TFRoFormerForTokenClassification,
            TFRoFormerLayer,
            TFRoFormerModel,
            TFRoFormerPreTrainedModel,
        )

    if is_flax_available():
        from .modeling_flax_roformer import (
            FLAX_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            FlaxRoFormerForMaskedLM,
            FlaxRoFormerForMultipleChoice,
            FlaxRoFormerForQuestionAnswering,
            FlaxRoFormerForSequenceClassification,
            FlaxRoFormerForTokenClassification,
            FlaxRoFormerModel,
            FlaxRoFormerPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
