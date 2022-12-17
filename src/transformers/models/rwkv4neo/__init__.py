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
    "configuration_rwkv4neo": ["RWKV4NEO_PRETRAINED_CONFIG_ARCHIVE_MAP", "Rwkv4NeoConfig"],
    "tokenization_rwkv4neo": ["Rwkv4NeoTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_rwkv4neo_fast"] = ["Rwkv4NeoTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_rwkv4neo"] = [
        "RWKV4NEO_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Rwkv4NeoForMaskedLM",
        "Rwkv4NeoForCausalLM",
        "Rwkv4NeoForMultipleChoice",
        "Rwkv4NeoForQuestionAnswering",
        "Rwkv4NeoForSequenceClassification",
        "Rwkv4NeoForTokenClassification",
        "Rwkv4NeoLayer",
        "Rwkv4NeoModel",
        "Rwkv4NeoPreTrainedModel",
        "load_tf_weights_in_rwkv4neo",
    ]




if TYPE_CHECKING:
    from .configuration_rwkv4neo import RWKV4NEO_PRETRAINED_CONFIG_ARCHIVE_MAP, Rwkv4NeoConfig
    from .tokenization_rwkv4neo import Rwkv4NeoTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_rwkv4neo_fast import Rwkv4NeoTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_rwkv4neo import (
            RWKV4NEO_PRETRAINED_MODEL_ARCHIVE_LIST,
            Rwkv4NeoForMaskedLM,
            Rwkv4NeoForCausalLM,
            Rwkv4NeoForMultipleChoice,
            Rwkv4NeoForQuestionAnswering,
            Rwkv4NeoForSequenceClassification,
            Rwkv4NeoForTokenClassification,
            Rwkv4NeoLayer,
            Rwkv4NeoModel,
            Rwkv4NeoPreTrainedModel,
            load_tf_weights_in_rwkv4neo,
        )



else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
