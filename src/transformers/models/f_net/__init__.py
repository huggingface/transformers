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
from ...file_utils import _BaseLazyModule, is_torch_available, is_tokenizers_available
_import_structure = {
    "configuration_f_net": ["F_NET_PRETRAINED_CONFIG_ARCHIVE_MAP", "FNetConfig"],
    "tokenization_f_net": ["FNetTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_f_net_fast"] = ["FNetTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_f_net"] = [
        "F_NET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FNetForMaskedLM",
        "FNetForCausalLM",
        "FNetForMultipleChoice",
        "FNetForQuestionAnswering",
        "FNetForSequenceClassification",
        "FNetForTokenClassification",
        "FNetLayer",
        "FNetModel",
        "FNetPreTrainedModel",
        "load_tf_weights_in_f_net",
    ]




if TYPE_CHECKING:
    from .configuration_f_net import F_NET_PRETRAINED_CONFIG_ARCHIVE_MAP, FNetConfig
    from .tokenization_f_net import FNetTokenizer

    if is_tokenizers_available():
        from .tokenization_f_net_fast import FNetTokenizerFast

    if is_torch_available():
        from .modeling_f_net import (
            F_NET_PRETRAINED_MODEL_ARCHIVE_LIST,
            FNetForMaskedLM,
            FNetForCausalLM,
            FNetForMultipleChoice,
            FNetForQuestionAnswering,
            FNetForSequenceClassification,
            FNetForTokenClassification,
            FNetLayer,
            FNetModel,
            FNetPreTrainedModel,
            load_tf_weights_in_f_net,
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
