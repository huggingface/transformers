from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM

from .modeling_long_vita import LongVITAForCausalLM
from .modeling_long_vita import LongVITAModel
from .configuration_long_vita import LongVITAConfig

AutoConfig.register("long_vita", LongVITAConfig)
AutoModel.register(LongVITAConfig, LongVITAModel)
AutoModelForCausalLM.register(LongVITAConfig, LongVITAForCausalLM)
# AutoTokenizer.register(Qwen2Config, Qwen2Tokenizer)

LongVITAConfig.register_for_auto_class()
LongVITAModel.register_for_auto_class("AutoModel")
LongVITAForCausalLM.register_for_auto_class("AutoModelForCausalLM")

# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_long_vita import *
    from .modeling_long_vita import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
