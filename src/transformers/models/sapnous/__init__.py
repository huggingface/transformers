# coding=utf-8
# Copyright 2025-present, the HuggingFace Inc. Team and AIRAS Inc. Team. All rights reserved.
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
from transformers.utils import _LazyModule
from transformers.models.auto import CONFIG_MAPPING, MODEL_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.models.auto import AutoConfig, AutoModel, AutoModelForCausalLM

_import_structure = {
    "configuration_sapnous": ["SAPNOUS_PRETRAINED_CONFIG_ARCHIVE_MAP", "SapnousT1Config"],
    "modeling_sapnous": ["SapnousT1Model", "SapnousT1ForCausalLM"],
    "tokenization_sapnous": ["SapnousT1Tokenizer"],
}

if TYPE_CHECKING:
    from .configuration_sapnous import SAPNOUS_PRETRAINED_CONFIG_ARCHIVE_MAP, SapnousT1Config
    from .modeling_sapnous import SapnousT1Model, SapnousT1ForCausalLM
    from .tokenization_sapnous import SapnousT1Tokenizer
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)

# Import configuration and models
from .configuration_sapnous import SapnousT1Config
from .modeling_sapnous import SapnousT1Model, SapnousT1ForCausalLM

# Register model in auto classes
CONFIG_MAPPING["sapnous_t1"] = SapnousT1Config
MODEL_MAPPING["sapnous_t1"] = SapnousT1Model
MODEL_FOR_CAUSAL_LM_MAPPING["sapnous_t1"] = SapnousT1ForCausalLM