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
from ...file_utils import _BaseLazyModule, is_tf_available, is_torch_available, is_tokenizers_available
_import_structure = {
    "configuration_linformer": ["LINFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "LinformerConfig"],
    "tokenization_linformer": ["LinformerTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_linformer_fast"] = ["LinformerTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_linformer"] = [
        "LINFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LinformerForMaskedLM",
        "LinformerForMultipleChoice",
        "LinformerForQuestionAnswering",
        "LinformerForSequenceClassification",
        "LinformerForTokenClassification",
        "LinformerLayer",
        "LinformerModel",
        "LinformerPreTrainedModel",
        "load_tf_weights_in_linformer",
    ]



if is_tf_available():
    _import_structure["modeling_tf_linformer"] = [
        "TF_LINFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFLinformerForMaskedLM",
        "TFLinformerForMultipleChoice",
        "TFLinformerForQuestionAnswering",
        "TFLinformerForSequenceClassification",
        "TFLinformerForTokenClassification",
        "TFLinformerLayer",
        "TFLinformerModel",
        "TFLinformerPreTrainedModel",
    ]




if TYPE_CHECKING:
    from .configuration_linformer import LINFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LinformerConfig
    from .tokenization_linformer import LinformerTokenizer

    if is_tokenizers_available():
        from .tokenization_linformer_fast import LinformerTokenizerFast

    if is_torch_available():
        from .modeling_linformer import (
            LINFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            LinformerForMaskedLM,
            LinformerForCausalLM,
            LinformerForMultipleChoice,
            LinformerForQuestionAnswering,
            LinformerForSequenceClassification,
            LinformerForTokenClassification,
            LinformerLayer,
            LinformerModel,
            LinformerPreTrainedModel,
            load_tf_weights_in_linformer,
        )



    if is_tf_available():
        from .modeling_tf_linformer import (
            TF_LINFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFLinformerForMaskedLM,
            TFLinformerForCausalLM,
            TFLinformerForMultipleChoice,
            TFLinformerForQuestionAnswering,
            TFLinformerForSequenceClassification,
            TFLinformerForTokenClassification,
            TFLinformerLayer,
            TFLinformerModel,
            TFLinformerPreTrainedModel,
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
