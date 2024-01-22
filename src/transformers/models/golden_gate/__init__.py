# Copyright 2024 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_golden_gate": ["GOLDEN_GATE_PRETRAINED_CONFIG_ARCHIVE_MAP", "GoldenGateConfig"],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_golden_gate"] = ["GoldenGateTokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_golden_gate_fast"] = ["GoldenGateTokenizerFast"]


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_golden_gate"] = [
        "GoldenGateForCausalLM",
        "GoldenGateModel",
        "GoldenGatePreTrainedModel",
        "GoldenGateForSequenceClassification",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_golden_gate"] = [
        "FlaxGoldenGateForCausalLM",
        "FlaxGoldenGateModel",
        "FlaxGoldenGatePreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_golden_gate import GOLDEN_GATE_PRETRAINED_CONFIG_ARCHIVE_MAP, GoldenGateConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_golden_gate import GoldenGateTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_golden_gate_fast import GoldenGateTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_golden_gate import (
            GoldenGateForCausalLM,
            GoldenGateForSequenceClassification,
            GoldenGateModel,
            GoldenGatePreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_golden_gate import (
            FlaxGoldenGateForCausalLM,
            FlaxGoldenGateModel,
            FlaxGoldenGatePreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
