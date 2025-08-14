# coding=utf-8
# Copyright 2023 Mixtral AI and the HuggingFace Inc. team. All rights reserved.
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
"""Mixtral model configuration"""

from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import RopeParameters, rope_config_validation
from ...utils import logging


logger = logging.get_logger(__name__)


class MixtralConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MixtralModel`]. It is used to instantiate an
    Mixtral model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mixtral-7B-v0.1 or Mixtral-7B-Instruct-v0.1.

    [mixtralai/Mixtral-8x7B](https://huggingface.co/mixtralai/Mixtral-8x7B)
    [mixtralai/Mixtral-7B-Instruct-v0.1](https://huggingface.co/mixtralai/Mixtral-7B-Instruct-v0.1)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
            vocab_size (`Optional`, *optional*, defaults to 32000): <fill_docstring>
            hidden_size (`Optional`, *optional*, defaults to 4096): <fill_docstring>
            intermediate_size (`Optional`, *optional*, defaults to 14336): <fill_docstring>
            num_hidden_layers (`Optional`, *optional*, defaults to 32): <fill_docstring>
            num_attention_heads (`Optional`, *optional*, defaults to 32): <fill_docstring>
            num_key_value_heads (`Optional`, *optional*, defaults to 8): <fill_docstring>
            head_dim (`Optional`, *optional*): <fill_docstring>
            hidden_act (`Optional`, *optional*, defaults to `"silu"`): <fill_docstring>
            max_position_embeddings (`Optional`, *optional*, defaults to 131072): <fill_docstring>
            initializer_range (`Optional`, *optional*, defaults to 0.02): <fill_docstring>
            rms_norm_eps (`Optional`, *optional*, defaults to 1e-05): <fill_docstring>
            use_cache (`Optional`, *optional*, defaults to `True`): <fill_docstring>
            pad_token_id (`Optional`, *optional*): <fill_docstring>
            bos_token_id (`Optional`, *optional*, defaults to 1): <fill_docstring>
            eos_token_id (`Optional`, *optional*, defaults to 2): <fill_docstring>
            tie_word_embeddings (`Optional`, *optional*, defaults to `False`): <fill_docstring>
            sliding_window (`Optional`, *optional*): <fill_docstring>
            attention_dropout (`Optional`, *optional*, defaults to 0.0): <fill_docstring>
            num_experts_per_tok (`Optional`, *optional*, defaults to 2): <fill_docstring>
            num_local_experts (`Optional`, *optional*, defaults to 8): <fill_docstring>
            output_router_logits (`Optional`, *optional*, defaults to `False`): <fill_docstring>
            router_aux_loss_coef (`Optional`, *optional*, defaults to 0.001): <fill_docstring>
            router_jitter_noise (`Optional`, *optional*, defaults to 0.0): <fill_docstring>
            rope_scaling (`Optional`, *optional*): <fill_docstring>

    ```python
    >>> from transformers import MixtralModel, MixtralConfig

    >>> # Initializing a Mixtral 7B style configuration
    >>> configuration = MixtralConfig()

    >>> # Initializing a model from the Mixtral 7B style configuration
    >>> model = MixtralModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mixtral"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.block_sparse_moe.gate": "colwise_rep",  # we need to replicate here to correctly route experts
        "layers.*.block_sparse_moe.experts.*.w1": "colwise",
        "layers.*.block_sparse_moe.experts.*.w2": "rowwise",
        "layers.*.block_sparse_moe.experts.*.w3": "colwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: Optional[int] = 32000,
        hidden_size: Optional[int] = 4096,
        intermediate_size: Optional[int] = 14336,
        num_hidden_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 32,
        num_key_value_heads: Optional[int] = 8,
        head_dim: Optional[int] = None,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 4096 * 32,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-5,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        tie_word_embeddings: Optional[bool] = False,
        sliding_window: Optional[int] = None,
        attention_dropout: Optional[float] = 0.0,
        num_experts_per_tok: Optional[int] = 2,
        num_local_experts: Optional[int] = 8,
        output_router_logits: Optional[bool] = False,
        router_aux_loss_coef: Optional[float] = 0.001,
        router_jitter_noise: Optional[float] = 0.0,
        rope_scaling: Optional[RopeParameters] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim

        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_jitter_noise = router_jitter_noise

        # Validate the correctness of rotary position embeddings parameters
        rope_theta = kwargs.get("rope_theta", 1000000.0)
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


__all__ = ["MixtralConfig"]
