# coding=utf-8
# Copyright 2025 The GLM4 & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
#
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

from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import RopeParameters, rope_config_validation


class Glm4Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4Model`]. It is used to instantiate an Glm4
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Glm4-4-9b-chat.
    e.g. [THUDM/GLM-4-9B-0414](https://huggingface.co/THUDM/GLM-4-9B-0414)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
            vocab_size (`Optional`, *optional*, defaults to 151552): <fill_docstring>
            hidden_size (`Optional`, *optional*, defaults to 4096): <fill_docstring>
            intermediate_size (`Optional`, *optional*, defaults to 13696): <fill_docstring>
            num_hidden_layers (`Optional`, *optional*, defaults to 40): <fill_docstring>
            num_attention_heads (`Optional`, *optional*, defaults to 32): <fill_docstring>
            num_key_value_heads (`Optional`, *optional*, defaults to 2): <fill_docstring>
            partial_rotary_factor (`Optional`, *optional*, defaults to 0.5): <fill_docstring>
            head_dim (`Optional`, *optional*, defaults to 128): <fill_docstring>
            hidden_act (`Optional`, *optional*, defaults to `"silu"`): <fill_docstring>
            attention_dropout (`Optional`, *optional*, defaults to 0.0): <fill_docstring>
            max_position_embeddings (`Optional`, *optional*, defaults to 131072): <fill_docstring>
            initializer_range (`Optional`, *optional*, defaults to 0.02): <fill_docstring>
            rms_norm_eps (`Optional`, *optional*, defaults to 0.0): <fill_docstring>
            use_cache (`Optional`, *optional*, defaults to `True`): <fill_docstring>
            tie_word_embeddings (`Optional`, *optional*, defaults to `False`): <fill_docstring>
            rope_scaling (`Optional`, *optional*): <fill_docstring>
            pad_token_id (`Optional`, *optional*, defaults to 151329): <fill_docstring>
            eos_token_id (`Optional`, *optional*, defaults to `[151329, 151336, 151338]`): <fill_docstring>
            bos_token_id (`Optional`, *optional*): <fill_docstring>
            attention_bias (`Optional`, *optional*, defaults to `True`): <fill_docstring>
    ```python
    >>> from transformers import Glm4Model, Glm4Config
    >>> # Initializing a Glm4 glm4-4-9b-chat style configuration
    >>> configuration = Glm4Config()
    >>> # Initializing a model from the glm4-4-9b-chat style configuration
    >>> model = Glm4Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_up_proj": "colwise_rep",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_rep",  # we need to replicate here due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 151552,
        hidden_size: Optional[int] = 4096,
        intermediate_size: Optional[int] = 13696,
        num_hidden_layers: Optional[int] = 40,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = 2,
        partial_rotary_factor: Optional[float] = 0.5,
        head_dim: Optional[int] = 128,
        hidden_act: Optional[str] = "silu",
        attention_dropout: Optional[float] = 0.0,
        max_position_embeddings: Optional[int] = 131072,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 0.00000015625,
        use_cache: Optional[bool] = True,
        tie_word_embeddings: Optional[bool] = False,
        rope_scaling: Optional[RopeParameters] = None,
        pad_token_id: Optional[int] = 151329,
        eos_token_id: Optional[list[int]] = [151329, 151336, 151338],
        bos_token_id: Optional[int] = None,
        attention_bias: Optional[bool] = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.partial_rotary_factor = partial_rotary_factor
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Validate the correctness of rotary position embeddings parameters
        rope_theta = kwargs.get("rope_theta", 10000.0)
        if rope_scaling is None:
            rope_scaling = {"rope_type": "default", "rope_theta": rope_theta}
        else:
            # BC: if there is a 'type' field, copy it it to 'rope_type'.
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            rope_scaling.update({"rope_theta": rope_theta, "rope_type": rope_type})
        self.rope_scaling = rope_scaling
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Glm4Config"]
