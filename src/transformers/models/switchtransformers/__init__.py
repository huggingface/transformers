# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

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
    "configuration_switchtransformers": [
        "SWITCHTRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SwitchTransformersConfig",
        "SwitchTransformersOnnxConfig",
    ]
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_switchtransformers"] = ["SwitchTransformersTokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_switchtransformers_fast"] = ["SwitchTransformersTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_switchtransformers"] = [
        "SWITCHTRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SwitchTransformersEncoderModel",
        "SwitchTransformersForConditionalGeneration",
        "SwitchTransformersModel",
        "SwitchTransformersPreTrainedModel",
        "load_tf_weights_in_switchtransformers",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_switchtransformers"] = [
        "TF_SWITCHTRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSwitchTransformersEncoderModel",
        "TFSwitchTransformersForConditionalGeneration",
        "TFSwitchTransformersModel",
        "TFSwitchTransformersPreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_switchtransformers"] = [
        "FlaxSwitchTransformersEncoderModel",
        "FlaxSwitchTransformersForConditionalGeneration",
        "FlaxSwitchTransformersModel",
        "FlaxSwitchTransformersPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_switchtransformers import (
        SWITCHTRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SwitchTransformersConfig,
        SwitchTransformersOnnxConfig,
    )

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_switchtransformers import SwitchTransformersTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_switchtransformers_fast import SwitchTransformersTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_switchtransformers import (
            SWITCHTRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST,
            SwitchTransformersEncoderModel,
            SwitchTransformersForConditionalGeneration,
            SwitchTransformersModel,
            SwitchTransformersPreTrainedModel,
            load_tf_weights_in_switchtransformers,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_switchtransformers import (
            TF_SWITCHTRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSwitchTransformersEncoderModel,
            TFSwitchTransformersForConditionalGeneration,
            TFSwitchTransformersModel,
            TFSwitchTransformersPreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_switchtransformers import (
            FlaxSwitchTransformersEncoderModel,
            FlaxSwitchTransformersForConditionalGeneration,
            FlaxSwitchTransformersModel,
            FlaxSwitchTransformersPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
