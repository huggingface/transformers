# Copyright 2024 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
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
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_emu3": ["Emu3Config", "Emu3VQVAEConfig", "Emu3TextConfig"],
    "processing_emu3": ["Emu3Processor"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_emu3"] = [
        "Emu3ForConditionalGeneration",
        "Emu3ForCausalLM",
        "Emu3TextModel",
        "Emu3PreTrainedModel",
        "Emu3VQVAE",
    ]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_emu3"] = ["Emu3ImageProcessor"]


if TYPE_CHECKING:
    from .configuration_emu3 import Emu3Config, Emu3TextConfig, Emu3VQVAEConfig
    from .processing_emu3 import Emu3Processor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_emu3 import (
            Emu3ForCausalLM,
            Emu3ForConditionalGeneration,
            Emu3PreTrainedModel,
            Emu3TextModel,
            Emu3VQVAE,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_emu3 import Emu3ImageProcessor


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
