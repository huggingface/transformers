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
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {"configuration_opt": ["OPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "OPTConfig"]}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_opt"] = [
        "OPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "OPTForCausalLM",
        "OPTModel",
        "OPTPreTrainedModel",
        "OPTForSequenceClassification",
        "OPTForQuestionAnswering",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_opt"] = ["TFOPTForCausalLM", "TFOPTModel", "TFOPTPreTrainedModel"]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_opt"] = [
        "FlaxOPTForCausalLM",
        "FlaxOPTModel",
        "FlaxOPTPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_opt import OPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OPTConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_opt import (
            OPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            OPTForCausalLM,
            OPTForQuestionAnswering,
            OPTForSequenceClassification,
            OPTModel,
            OPTPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_opt import TFOPTForCausalLM, TFOPTModel, TFOPTPreTrainedModel

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_opt import FlaxOPTForCausalLM, FlaxOPTModel, FlaxOPTPreTrainedModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
