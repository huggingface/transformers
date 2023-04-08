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
    is_torch_available,
)


_import_structure = {"configuration_umt5": ["UMT5Config", "UMT5OnnxConfig"]}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_umt5"] = [
        "UMT5EncoderModel",
        "UMT5ForConditionalGeneration",
        "UMT5Model",
        "UMT5PreTrainedModel",
        "UMT5Stack",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_umt5"] = ["UTFMT5EncoderModel", "UTFMT5ForConditionalGeneration", "UTFMT5Model"]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_umt5"] = [
        "FlaxUMT5EncoderModel",
        "FlaxUMT5ForConditionalGeneration",
        "FlaxUMT5Model",
    ]


if TYPE_CHECKING:
    from .configuration_umt5 import UMT5Config, UMT5OnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_umt5 import (
            UMT5EncoderModel,
            UMT5ForConditionalGeneration,
            UMT5Model,
            UMT5PreTrainedModel,
            UMT5Stack,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_umt5 import UTFMT5EncoderModel, UTFMT5ForConditionalGeneration, UTFMT5Model

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_umt5 import FlaxUMT5EncoderModel, FlaxUMT5ForConditionalGeneration, FlaxUMT5Model

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
