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



from ...utils import is_flax_available




_import_structure = {
    "configuration_pegasus_x": ["PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP", "PegasusXConfig"],
    "tokenization_pegasus_x": ["PegasusXTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_pegasus_x_fast"] = ["PegasusXTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_pegasus_x"] = [
        "PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PegasusXForConditionalGeneration",
        "PegasusXForQuestionAnswering",
        "PegasusXForSequenceClassification",
        "PegasusXForCausalLM",
        "PegasusXModel",
        "PegasusXPreTrainedModel",
    ]



try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_pegasus_x"] = [
        "FlaxPegasusXForConditionalGeneration",
        "FlaxPegasusXForQuestionAnswering",
        "FlaxPegasusXForSequenceClassification",
        "FlaxPegasusXModel",
        "FlaxPegasusXPreTrainedModel",
    ]




if TYPE_CHECKING:
    from .configuration_pegasus_x import PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP, PegasusXConfig
    from .tokenization_pegasus_x import PegasusXTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_pegasus_x_fast import PegasusXTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_pegasus_x import (
            PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST,
            PegasusXForConditionalGeneration,
            PegasusXForCausalLM,
            PegasusXForQuestionAnswering,
            PegasusXForSequenceClassification,
            PegasusXModel,
            PegasusXPreTrainedModel,
        )



    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_pegasus_x import (
            FlaxPegasusXForConditionalGeneration,
            FlaxPegasusXForQuestionAnswering,
            FlaxPegasusXForSequenceClassification,
            FlaxPegasusXModel,
            FlaxPegasusXPreTrainedModel,
        )



else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
