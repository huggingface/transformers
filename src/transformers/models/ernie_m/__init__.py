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

# rely on isort to merge the imports
from ...utils import  _LazyModule, OptionalDependencyNotAvailable, is_tokenizers_available
from ...utils import is_torch_available




_import_structure = {
    "configuration_ernie_m": ["ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP", "ErnieMConfig"],
    "tokenization_ernie_m": ["ErnieMTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_ernie_m_fast"] = ["ErnieMTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_ernie_m"] = [
        "ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ErnieMForMaskedLM",
        "ErnieMForCausalLM",
        "ErnieMForMultipleChoice",
        "ErnieMForQuestionAnswering",
        "ErnieMForSequenceClassification",
        "ErnieMForTokenClassification",
        "ErnieMLayer",
        "ErnieMModel",
        "ErnieMPreTrainedModel",
        "load_tf_weights_in_ernie_m",
    ]




if TYPE_CHECKING:
    from .configuration_ernie_m import ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieMConfig
    from .tokenization_ernie_m import ErnieMTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_ernie_m_fast import ErnieMTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_ernie_m import (
            ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST,
            ErnieMForMaskedLM,
            ErnieMForCausalLM,
            ErnieMForMultipleChoice,
            ErnieMForQuestionAnswering,
            ErnieMForSequenceClassification,
            ErnieMForTokenClassification,
            ErnieMLayer,
            ErnieMModel,
            ErnieMPreTrainedModel,
            load_tf_weights_in_ernie_m,
        )



else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
