# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Blueberry model configuration"""

from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule


_import_structure = {
    "configuration_blueberry": [
        "BLUEBERRY_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BlueberryConfig",
    ],
}

try:
    from ...utils import is_torch_available
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_blueberry"] = [
        "BlueberryForCausalLM",
        "BlueberryModel",
        "BlueberryPreTrainedModel",
        "BlueberryForSequenceClassification",
    ]


if TYPE_CHECKING:
    from .configuration_blueberry import BLUEBERRY_PRETRAINED_CONFIG_ARCHIVE_MAP, BlueberryConfig

    try:
        from ...utils import is_torch_available
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_blueberry import (
            BlueberryForCausalLM,
            BlueberryForSequenceClassification,
            BlueberryModel,
            BlueberryPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"BLUEBERRY_PRETRAINED_CONFIG_ARCHIVE_MAP": BLUEBERRY_PRETRAINED_CONFIG_ARCHIVE_MAP},
    )