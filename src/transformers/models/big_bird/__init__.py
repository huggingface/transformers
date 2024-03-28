# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_big_bird": ["BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP", "BigBirdConfig", "BigBirdOnnxConfig"],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_big_bird"] = ["BigBirdTokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_big_bird_fast"] = ["BigBirdTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_big_bird"] = [
        "BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BigBirdForCausalLM",
        "BigBirdForMaskedLM",
        "BigBirdForMultipleChoice",
        "BigBirdForPreTraining",
        "BigBirdForQuestionAnswering",
        "BigBirdForSequenceClassification",
        "BigBirdForTokenClassification",
        "BigBirdLayer",
        "BigBirdModel",
        "BigBirdPreTrainedModel",
        "load_tf_weights_in_big_bird",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_big_bird"] = [
        "FlaxBigBirdForCausalLM",
        "FlaxBigBirdForMaskedLM",
        "FlaxBigBirdForMultipleChoice",
        "FlaxBigBirdForPreTraining",
        "FlaxBigBirdForQuestionAnswering",
        "FlaxBigBirdForSequenceClassification",
        "FlaxBigBirdForTokenClassification",
        "FlaxBigBirdModel",
        "FlaxBigBirdPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_big_bird import BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP, BigBirdConfig, BigBirdOnnxConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_big_bird import BigBirdTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_big_bird_fast import BigBirdTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_big_bird import (
            BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST,
            BigBirdForCausalLM,
            BigBirdForMaskedLM,
            BigBirdForMultipleChoice,
            BigBirdForPreTraining,
            BigBirdForQuestionAnswering,
            BigBirdForSequenceClassification,
            BigBirdForTokenClassification,
            BigBirdLayer,
            BigBirdModel,
            BigBirdPreTrainedModel,
            load_tf_weights_in_big_bird,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_big_bird import (
            FlaxBigBirdForCausalLM,
            FlaxBigBirdForMaskedLM,
            FlaxBigBirdForMultipleChoice,
            FlaxBigBirdForPreTraining,
            FlaxBigBirdForQuestionAnswering,
            FlaxBigBirdForSequenceClassification,
            FlaxBigBirdForTokenClassification,
            FlaxBigBirdModel,
            FlaxBigBirdPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
