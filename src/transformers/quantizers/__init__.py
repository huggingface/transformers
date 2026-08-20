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
import sys
from typing import TYPE_CHECKING

from ..utils import _LazyModule


if TYPE_CHECKING:
    from .auto import AutoHfQuantizer, AutoQuantizationConfig, register_quantization_config, register_quantizer
    from .base import HfQuantizer
    from .quantizers_utils import get_module_from_name


_import_structure = {
    "auto": ["AutoHfQuantizer", "AutoQuantizationConfig", "register_quantization_config", "register_quantizer"],
    "base": ["HfQuantizer"],
    "quantizers_utils": ["get_module_from_name"],
}
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
