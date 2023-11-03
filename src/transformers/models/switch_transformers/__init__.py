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
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_switch_transformers": [
        "SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SwitchTransformersConfig",
        "SwitchTransformersOnnxConfig",
    ]
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_switch_transformers"] = [
        "SWITCH_TRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SwitchTransformersEncoderModel",
        "SwitchTransformersForConditionalGeneration",
        "SwitchTransformersModel",
        "SwitchTransformersPreTrainedModel",
        "SwitchTransformersTop1Router",
        "SwitchTransformersSparseMLP",
    ]


if TYPE_CHECKING:
    from .configuration_switch_transformers import (
        SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SwitchTransformersConfig,
        SwitchTransformersOnnxConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_switch_transformers import (
            SWITCH_TRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST,
            SwitchTransformersEncoderModel,
            SwitchTransformersForConditionalGeneration,
            SwitchTransformersModel,
            SwitchTransformersPreTrainedModel,
            SwitchTransformersSparseMLP,
            SwitchTransformersTop1Router,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
