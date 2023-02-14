# Copyright 2022 The HuggingFace Team. All rights reserved.
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
    is_torch_available,
)


_import_structure = {
    "configuration_roberta_prelayernorm": [
        "ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "RobertaPreLayerNormConfig",
        "RobertaPreLayerNormOnnxConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_roberta_prelayernorm"] = [
        "ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RobertaPreLayerNormForCausalLM",
        "RobertaPreLayerNormForMaskedLM",
        "RobertaPreLayerNormForMultipleChoice",
        "RobertaPreLayerNormForQuestionAnswering",
        "RobertaPreLayerNormForSequenceClassification",
        "RobertaPreLayerNormForTokenClassification",
        "RobertaPreLayerNormModel",
        "RobertaPreLayerNormPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_roberta_prelayernorm"] = [
        "TF_ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFRobertaPreLayerNormForCausalLM",
        "TFRobertaPreLayerNormForMaskedLM",
        "TFRobertaPreLayerNormForMultipleChoice",
        "TFRobertaPreLayerNormForQuestionAnswering",
        "TFRobertaPreLayerNormForSequenceClassification",
        "TFRobertaPreLayerNormForTokenClassification",
        "TFRobertaPreLayerNormMainLayer",
        "TFRobertaPreLayerNormModel",
        "TFRobertaPreLayerNormPreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_roberta_prelayernorm"] = [
        "FlaxRobertaPreLayerNormForCausalLM",
        "FlaxRobertaPreLayerNormForMaskedLM",
        "FlaxRobertaPreLayerNormForMultipleChoice",
        "FlaxRobertaPreLayerNormForQuestionAnswering",
        "FlaxRobertaPreLayerNormForSequenceClassification",
        "FlaxRobertaPreLayerNormForTokenClassification",
        "FlaxRobertaPreLayerNormModel",
        "FlaxRobertaPreLayerNormPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_roberta_prelayernorm import (
        ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RobertaPreLayerNormConfig,
        RobertaPreLayerNormOnnxConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_roberta_prelayernorm import (
            ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST,
            RobertaPreLayerNormForCausalLM,
            RobertaPreLayerNormForMaskedLM,
            RobertaPreLayerNormForMultipleChoice,
            RobertaPreLayerNormForQuestionAnswering,
            RobertaPreLayerNormForSequenceClassification,
            RobertaPreLayerNormForTokenClassification,
            RobertaPreLayerNormModel,
            RobertaPreLayerNormPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_roberta_prelayernorm import (
            TF_ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRobertaPreLayerNormForCausalLM,
            TFRobertaPreLayerNormForMaskedLM,
            TFRobertaPreLayerNormForMultipleChoice,
            TFRobertaPreLayerNormForQuestionAnswering,
            TFRobertaPreLayerNormForSequenceClassification,
            TFRobertaPreLayerNormForTokenClassification,
            TFRobertaPreLayerNormMainLayer,
            TFRobertaPreLayerNormModel,
            TFRobertaPreLayerNormPreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_roberta_prelayernorm import (
            FlaxRobertaPreLayerNormForCausalLM,
            FlaxRobertaPreLayerNormForMaskedLM,
            FlaxRobertaPreLayerNormForMultipleChoice,
            FlaxRobertaPreLayerNormForQuestionAnswering,
            FlaxRobertaPreLayerNormForSequenceClassification,
            FlaxRobertaPreLayerNormForTokenClassification,
            FlaxRobertaPreLayerNormModel,
            FlaxRobertaPreLayerNormPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
