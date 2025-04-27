# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team. All rights reserved.
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
"""ConvAI Innovations Hindi Causal LM model."""

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_torch_available,
)


_import_structure = {
    "configuration_hindi_causal_lm": ["HindiCausalLMConfig"],
}

# Tokenizer: real or dummy
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_hindi_causal_lm"] = ["HindiCausalLMTokenizer"]

# Model: real or dummy
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .dummy_pt_objects import (
        HindiCausalLMForCausalLM,
    )
    # Alias for auto-factory fallback
    HindiCausalLMHeadModel = HindiCausalLMForCausalLM
else:
    _import_structure["modeling_hindi_causal_lm"] = [
        "HindiCausalLMModel",
        "HindiCausalLMForCausalLM",
        "HindiCausalLMPreTrainedModel",
        "HindiCausalLMHeadModel",
    ]

if TYPE_CHECKING:

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        pass

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .dummy_pt_objects import (
            HindiCausalLMForCausalLM,
        )
        HindiCausalLMHeadModel = HindiCausalLMForCausalLM
    else:
        from .modeling_hindi_causal_lm import (
            HindiCausalLMForCausalLM,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
