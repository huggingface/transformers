# Copyright 2023 The HuggingFace and Baidu Team. All rights reserved.
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
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_torch_available


_import_structure = {
    "configuration_ernie_m": ["ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP", "ErnieMConfig"],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_ernie_m"] = ["ErnieMTokenizer"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_ernie_m"] = [
        "ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ErnieMForMultipleChoice",
        "ErnieMForQuestionAnswering",
        "ErnieMForSequenceClassification",
        "ErnieMForTokenClassification",
        "ErnieMModel",
        "ErnieMPreTrainedModel",
        "ErnieMForInformationExtraction",
    ]


if TYPE_CHECKING:
    from .configuration_ernie_m import ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieMConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_ernie_m import ErnieMTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_ernie_m import (
            ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST,
            ErnieMForInformationExtraction,
            ErnieMForMultipleChoice,
            ErnieMForQuestionAnswering,
            ErnieMForSequenceClassification,
            ErnieMForTokenClassification,
            ErnieMModel,
            ErnieMPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
