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

from ...utils import  _LazyModule, OptionalDependencyNotAvailable, is_tokenizers_available
from ...utils import is_torch_available




_import_structure = {
    "configuration_code_shell": ["CODE_SHELL_PRETRAINED_CONFIG_ARCHIVE_MAP", "CodeShellConfig"],
    "tokenization_code_shell": ["CodeShellTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_code_shell_fast"] = ["CodeShellTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_code_shell"] = [
        "CODE_SHELL_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CodeShellForMaskedLM",
        "CodeShellForCausalLM",
        "CodeShellForMultipleChoice",
        "CodeShellForQuestionAnswering",
        "CodeShellForSequenceClassification",
        "CodeShellForTokenClassification",
        "CodeShellLayer",
        "CodeShellModel",
        "CodeShellPreTrainedModel",
        "load_tf_weights_in_code_shell",
    ]




if TYPE_CHECKING:
    from .configuration_code_shell import CODE_SHELL_PRETRAINED_CONFIG_ARCHIVE_MAP, CodeShellConfig
    from .tokenization_code_shell import CodeShellTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_code_shell_fast import CodeShellTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_code_shell import (
            CODE_SHELL_PRETRAINED_MODEL_ARCHIVE_LIST,
            CodeShellForMaskedLM,
            CodeShellForCausalLM,
            CodeShellForMultipleChoice,
            CodeShellForQuestionAnswering,
            CodeShellForSequenceClassification,
            CodeShellForTokenClassification,
            CodeShellLayer,
            CodeShellModel,
            CodeShellPreTrainedModel,
            load_tf_weights_in_code_shell,
        )



else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
