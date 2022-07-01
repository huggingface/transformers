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
    is_tf_available,
    is_torch_available,
)


_import_structure = {
    "configuration_speecht5": ["SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP", "Speecht5Config"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_speecht5"] = [
        "SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Speecht5ForAudioFrameClassification",
        "Speecht5ForCTC",
        "Speecht5ForMaskedLM",
        "Speecht5ForPreTraining",
        "Speecht5ForSequenceClassification",
        "Speecht5ForXVector",
        "Speecht5Model",
        "Speecht5PreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_speecht5"] = [
        "TF_SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSpeecht5ForCTC",
        "TFSpeecht5Model",
        "TFSpeecht5PreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_speecht5"] = [
        "FlaxSpeecht5ForCTC",
        "FlaxSpeecht5ForPreTraining",
        "FlaxSpeecht5Model",
        "FlaxSpeecht5PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_speecht5 import SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP, Speecht5Config

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_speecht5 import (
            SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST,
            Speecht5ForAudioFrameClassification,
            Speecht5ForCTC,
            Speecht5ForMaskedLM,
            Speecht5ForPreTraining,
            Speecht5ForSequenceClassification,
            Speecht5ForXVector,
            Speecht5Model,
            Speecht5PreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_speecht5 import (
            TF_SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSpeecht5ForCTC,
            TFSpeecht5Model,
            TFSpeecht5PreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_speecht5 import (
            FlaxSpeecht5ForCTC,
            FlaxSpeecht5ForPreTraining,
            FlaxSpeecht5Model,
            FlaxSpeecht5PreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
