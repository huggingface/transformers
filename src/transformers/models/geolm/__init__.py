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

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_geolm": ["GEOLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "GeoLMConfig"],
    "tokenization_geolm": ["GeoLMTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_geolm_fast"] = ["GeoLMTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_geolm"] = [
        "GEOLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GeoLMForMaskedLM",
        "GeoLMForCausalLM",
        "GeoLMForTokenClassification",
        "GeoLMLayer",
        "GeoLMModel",
        "GeoLMPreTrainedModel",
        "load_tf_weights_in_geolm",
    ]


if TYPE_CHECKING:
    from .configuration_geolm import GEOLM_PRETRAINED_CONFIG_ARCHIVE_MAP, GeoLMConfig
    from .tokenization_geolm import GeoLMTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_geolm_fast import GeoLMTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_geolm import (
            GEOLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            GeoLMForCausalLM,
            GeoLMForMaskedLM,
            GeoLMForTokenClassification,
            GeoLMLayer,
            GeoLMModel,
            GeoLMPreTrainedModel,
            load_tf_weights_in_geolm,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
