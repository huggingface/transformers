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

from ...file_utils import (
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_mbart": ["MBART_PRETRAINED_CONFIG_ARCHIVE_MAP", "MBartConfig", "MBartOnnxConfig"],
}

if is_sentencepiece_available():
    _import_structure["tokenization_mbart"] = ["MBartTokenizer"]

if is_tokenizers_available():
    _import_structure["tokenization_mbart_fast"] = ["MBartTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_mbart"] = [
        "MBART_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MBartForCausalLM",
        "MBartForConditionalGeneration",
        "MBartForQuestionAnswering",
        "MBartForSequenceClassification",
        "MBartModel",
        "MBartPreTrainedModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_mbart"] = [
        "TFMBartForConditionalGeneration",
        "TFMBartModel",
        "TFMBartPreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_mbart"] = [
        "FlaxMBartForConditionalGeneration",
        "FlaxMBartForQuestionAnswering",
        "FlaxMBartForSequenceClassification",
        "FlaxMBartModel",
        "FlaxMBartPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_mbart import MBART_PRETRAINED_CONFIG_ARCHIVE_MAP, MBartConfig, MBartOnnxConfig

    if is_sentencepiece_available():
        from .tokenization_mbart import MBartTokenizer

    if is_tokenizers_available():
        from .tokenization_mbart_fast import MBartTokenizerFast

    if is_torch_available():
        from .modeling_mbart import (
            MBART_PRETRAINED_MODEL_ARCHIVE_LIST,
            MBartForCausalLM,
            MBartForConditionalGeneration,
            MBartForQuestionAnswering,
            MBartForSequenceClassification,
            MBartModel,
            MBartPreTrainedModel,
        )

    if is_tf_available():
        from .modeling_tf_mbart import TFMBartForConditionalGeneration, TFMBartModel, TFMBartPreTrainedModel

    if is_flax_available():
        from .modeling_flax_mbart import (
            FlaxMBartForConditionalGeneration,
            FlaxMBartForQuestionAnswering,
            FlaxMBartForSequenceClassification,
            FlaxMBartModel,
            FlaxMBartPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
