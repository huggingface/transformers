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
    _BaseLazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_bart": ["BART_PRETRAINED_CONFIG_ARCHIVE_MAP", "BartConfig"],
    "tokenization_bart": ["BartTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_bart_fast"] = ["BartTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_bart"] = [
        "BART_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BartForCausalLM",
        "BartForConditionalGeneration",
        "BartForQuestionAnswering",
        "BartForSequenceClassification",
        "BartModel",
        "BartPretrainedModel",
        "PretrainedBartModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_bart"] = ["TFBartForConditionalGeneration", "TFBartModel", "TFBartPretrainedModel"]

if is_flax_available():
    _import_structure["modeling_flax_bart"] = [
        "FlaxBartForConditionalGeneration",
        "FlaxBartForQuestionAnswering",
        "FlaxBartForSequenceClassification",
        "FlaxBartModel",
        "FlaxBartPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_bart import BART_PRETRAINED_CONFIG_ARCHIVE_MAP, BartConfig
    from .tokenization_bart import BartTokenizer

    if is_tokenizers_available():
        from .tokenization_bart_fast import BartTokenizerFast

    if is_torch_available():
        from .modeling_bart import (
            BART_PRETRAINED_MODEL_ARCHIVE_LIST,
            BartForCausalLM,
            BartForConditionalGeneration,
            BartForQuestionAnswering,
            BartForSequenceClassification,
            BartModel,
            BartPretrainedModel,
            PretrainedBartModel,
        )

    if is_tf_available():
        from .modeling_tf_bart import TFBartForConditionalGeneration, TFBartModel, TFBartPretrainedModel

    if is_flax_available():
        from .modeling_flax_bart import (
            FlaxBartForConditionalGeneration,
            FlaxBartForQuestionAnswering,
            FlaxBartForSequenceClassification,
            FlaxBartModel,
            FlaxBartPreTrainedModel,
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
