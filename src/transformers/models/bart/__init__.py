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

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_bart": ["BartConfig", "BartOnnxConfig"],
    "tokenization_bart": ["BartTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_bart_fast"] = ["BartTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_bart"] = [
        "BartForCausalLM",
        "BartForConditionalGeneration",
        "BartForQuestionAnswering",
        "BartForSequenceClassification",
        "BartModel",
        "BartPreTrainedModel",
        "BartPretrainedModel",
        "PretrainedBartModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_bart"] = [
        "TFBartForConditionalGeneration",
        "TFBartForSequenceClassification",
        "TFBartModel",
        "TFBartPretrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_bart"] = [
        "FlaxBartDecoderPreTrainedModel",
        "FlaxBartForCausalLM",
        "FlaxBartForConditionalGeneration",
        "FlaxBartForQuestionAnswering",
        "FlaxBartForSequenceClassification",
        "FlaxBartModel",
        "FlaxBartPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_bart import BartConfig, BartOnnxConfig
    from .tokenization_bart import BartTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_bart_fast import BartTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_bart import (
            BartForCausalLM,
            BartForConditionalGeneration,
            BartForQuestionAnswering,
            BartForSequenceClassification,
            BartModel,
            BartPreTrainedModel,
            BartPretrainedModel,
            PretrainedBartModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_bart import (
            TFBartForConditionalGeneration,
            TFBartForSequenceClassification,
            TFBartModel,
            TFBartPretrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_bart import (
            FlaxBartDecoderPreTrainedModel,
            FlaxBartForCausalLM,
            FlaxBartForConditionalGeneration,
            FlaxBartForQuestionAnswering,
            FlaxBartForSequenceClassification,
            FlaxBartModel,
            FlaxBartPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
