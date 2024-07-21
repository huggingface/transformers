# Copyright 2022 Cognitivess and The HuggingFace Inc. team. All rights reserved.
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

from ...utils import (OptionalDependencyNotAvailable, _LazyModule,
                      is_flax_available, is_sentencepiece_available,
                      is_tokenizers_available, is_torch_available)

_import_structure = {
    "configuration_Cognitivess": ["CognitivessConfig"],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_Cognitivess"] = ["CognitivessTokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_Cognitivess_fast"] = ["CognitivessTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_Cognitivess"] = [
        "CognitivessForCausalLM",
        "CognitivessModel",
        "CognitivessPreTrainedModel",
        "CognitivessForSequenceClassification",
        "CognitivessForQuestionAnswering",
        "CognitivessForTokenClassification",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_Cognitivess"] = [
        "FlaxCognitivessForCausalLM",
        "FlaxCognitivessModel",
        "FlaxCognitivessPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_Cognitivess import CognitivessConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_Cognitivess import CognitivessTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_Cognitivess_fast import CognitivessTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_Cognitivess import (
            CognitivessForCausalLM, CognitivessForQuestionAnswering,
            CognitivessForSequenceClassification,
            CognitivessForTokenClassification, CognitivessModel,
            CognitivessPreTrainedModel)

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_Cognitivess import (FlaxCognitivessForCausalLM,
                                                FlaxCognitivessModel,
                                                FlaxCognitivessPreTrainedModel)


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
