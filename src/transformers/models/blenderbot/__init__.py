# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

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

from ...file_utils import _LazyModule, is_flax_available, is_tf_available, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_blenderbot": ["BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BlenderbotConfig"],
    "tokenization_blenderbot": ["BlenderbotTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_blenderbot_fast"] = ["BlenderbotTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_blenderbot"] = [
        "BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BlenderbotForCausalLM",
        "BlenderbotForConditionalGeneration",
        "BlenderbotModel",
        "BlenderbotPreTrainedModel",
    ]


if is_tf_available():
    _import_structure["modeling_tf_blenderbot"] = [
        "TFBlenderbotForConditionalGeneration",
        "TFBlenderbotModel",
        "TFBlenderbotPreTrainedModel",
    ]


if is_flax_available():
    _import_structure["modeling_flax_blenderbot"] = [
        "FlaxBlenderbotForConditionalGeneration",
        "FlaxBlenderbotModel",
        "FlaxBlenderbotPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_blenderbot import BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP, BlenderbotConfig
    from .tokenization_blenderbot import BlenderbotTokenizer

    if is_tokenizers_available():
        from .tokenization_blenderbot_fast import BlenderbotTokenizerFast

    if is_torch_available():
        from .modeling_blenderbot import (
            BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BlenderbotForCausalLM,
            BlenderbotForConditionalGeneration,
            BlenderbotModel,
            BlenderbotPreTrainedModel,
        )

    if is_tf_available():
        from .modeling_tf_blenderbot import (
            TFBlenderbotForConditionalGeneration,
            TFBlenderbotModel,
            TFBlenderbotPreTrainedModel,
        )

    if is_flax_available():
        from .modeling_flax_blenderbot import (
            FlaxBlenderbotForConditionalGeneration,
            FlaxBlenderbotModel,
            FlaxBlenderbotPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
