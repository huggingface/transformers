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


_import_structure = {"configuration_regnet": ["REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "RegNetConfig"]}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_regnet"] = [
        "REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RegNetForImageClassification",
        "RegNetModel",
        "RegNetPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_regnet"] = [
        "TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFRegNetForImageClassification",
        "TFRegNetModel",
        "TFRegNetPreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_regnet"] = [
        "FlaxRegNetForImageClassification",
        "FlaxRegNetModel",
        "FlaxRegNetPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_regnet import REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP, RegNetConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_regnet import (
            REGNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            RegNetForImageClassification,
            RegNetModel,
            RegNetPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_regnet import (
            TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFRegNetForImageClassification,
            TFRegNetModel,
            TFRegNetPreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_regnet import (
            FlaxRegNetForImageClassification,
            FlaxRegNetModel,
            FlaxRegNetPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
