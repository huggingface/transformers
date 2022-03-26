# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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

from ...utils import _LazyModule, is_flax_available, is_tf_available, is_torch_available


_import_structure = {
    "configuration_gptj": ["GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTJConfig", "GPTJOnnxConfig"],
}

if is_torch_available():
    _import_structure["modeling_gptj"] = [
        "GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTJForCausalLM",
        "GPTJForQuestionAnswering",
        "GPTJForSequenceClassification",
        "GPTJModel",
        "GPTJPreTrainedModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_gptj"] = [
        "TFGPTJForCausalLM",
        "TFGPTJForQuestionAnswering",
        "TFGPTJForSequenceClassification",
        "TFGPTJModel",
        "TFGPTJPreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_gptj"] = [
        "FlaxGPTJForCausalLM",
        "FlaxGPTJModel",
        "FlaxGPTJPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_gptj import GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTJConfig, GPTJOnnxConfig

    if is_torch_available():
        from .modeling_gptj import (
            GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTJForCausalLM,
            GPTJForQuestionAnswering,
            GPTJForSequenceClassification,
            GPTJModel,
            GPTJPreTrainedModel,
        )

    if is_tf_available():
        from .modeling_tf_gptj import (
            TFGPTJForCausalLM,
            TFGPTJForQuestionAnswering,
            TFGPTJForSequenceClassification,
            TFGPTJModel,
            TFGPTJPreTrainedModel,
        )

    if is_flax_available():
        from .modeling_flax_gptj import FlaxGPTJForCausalLM, FlaxGPTJModel, FlaxGPTJPreTrainedModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
