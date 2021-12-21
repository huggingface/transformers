# flake8: noqa
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

from ..file_utils import _LazyModule


_import_structure = {
    "config": [
        "EXTERNAL_DATA_FORMAT_SIZE_LIMIT",
        "OnnxConfig",
        "OnnxConfigWithPast",
        "OnnxSeq2SeqConfigWithPast",
        "PatchingSpec",
    ],
    "convert": ["export", "validate_model_outputs"],
    "utils": ["ParameterFormat", "compute_serialized_parameters_size"],
}


if TYPE_CHECKING:
    from .config import (
        EXTERNAL_DATA_FORMAT_SIZE_LIMIT,
        OnnxConfig,
        OnnxConfigWithPast,
        OnnxSeq2SeqConfigWithPast,
        PatchingSpec,
    )
    from .convert import export, validate_model_outputs
    from .utils import ParameterFormat, compute_serialized_parameters_size

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
