# Copyright 2026 HuggingFace Inc.
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

"""Small tester defaults, shared by causal and multimodal tester bases.

Model testers declare a literal ``small_config_overrides`` dictionary for tools
that read test source without importing it. Python callers use
``Tester.get_small_config_defaults(**kwargs)`` for merged tester values without
instantiating a tester. These values still need the tester's ``get_config()`` to
construct a model config; they are not a checkpoint or quantization recipe.
Legacy constructor customizations still run normally but must be migrated into
``small_config_overrides`` to appear in the class-level defaults API.

For example, a model can declare ``small_config_overrides = {"hidden_size": 64}``
and callers can use ``Tester.get_small_config_defaults(hidden_size=128)``.
The caller's width wins, and any derived head size uses that final width.
"""

from copy import deepcopy


# Shared text-model defaults for CausalLMModelTester and MultiModalModelTester.
_TEXT_MODEL_TESTER_DEFAULTS = {
    "batch_size": 13,
    "seq_length": 7,
    "is_training": True,
    "use_input_mask": True,
    "use_labels": True,
    "vocab_size": 99,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
    "hidden_act": "gelu",
    "max_position_embeddings": 512,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "expert_interval": 1,
    "moe_layer_start_index": 0,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 36,
    "shared_expert_gate": True,
    "moe_num_shared_experts": 2,
    "num_experts_per_tok": 2,
    "num_experts": 8,
}


class ModelTesterConfigMixin:
    defaults = _TEXT_MODEL_TESTER_DEFAULTS
    small_config_overrides = {}

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        """Return fresh merged tester values, including explicit caller overrides.

        Precedence: caller kwargs > model overrides > modality/base defaults >
        shared text defaults. Tester bases declare ``defaults``; models declare
        literal ``small_config_overrides`` dictionaries. Inherited entries are retained.
        Derived values are resolved by the causal or modality tester afterwards.
        """
        values = {}
        for base in reversed(cls.__mro__):
            values.update(base.__dict__.get("defaults", {}))
        for base in reversed(cls.__mro__):
            values.update(base.__dict__.get("small_config_overrides", {}))
        values = deepcopy(values)
        values.update(kwargs)
        return values
