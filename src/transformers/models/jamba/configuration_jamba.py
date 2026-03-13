# Copyright 2024 AI21 Labs Ltd. and the HuggingFace Inc. team. All rights reserved.
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
"""Jamba model configuration"""

import math

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="ai21labs/Jamba-v0.1")
class JambaConfig(PreTrainedConfig):
    r"""
    expert_layer_period (`int`, *optional*, defaults to 2):
        Once in this many layers, we will have an expert layer
    expert_layer_offset (`int`, *optional*, defaults to 1):
        The first layer index that contains an expert mlp layer
    attn_layer_period (`int`, *optional*, defaults to 8):
        Once in this many layers, we will have a vanilla attention layer
    attn_layer_offset (`int`, *optional*, defaults to 4):
        The first layer index that contains a vanilla attention mlp layer
    mamba_dt_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
        Rank of the mamba discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
    use_mamba_kernels (`bool`, *optional*, defaults to `True`):
        Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
        `causal-conv1d` are installed, and the mamba modules are running on a CUDA device. Raises ValueError if
        `True` and kernels are not available
    """

    model_type = "jamba"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "num_experts",
    }

    def __init__(
        self,
        vocab_size=65536,
        tie_word_embeddings=False,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=262144,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_experts=16,
        expert_layer_period=2,
        expert_layer_offset=1,
        attn_layer_period=8,
        attn_layer_offset=4,
        use_mamba_kernels=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.expert_layer_period = expert_layer_period
        self.expert_layer_offset = expert_layer_offset
        self.attn_layer_period = attn_layer_period
        self.attn_layer_offset = attn_layer_offset

        self._check_supported_offset("attention", self.attn_layer_period, self.attn_layer_offset)
        self._check_supported_offset("expert", self.expert_layer_period, self.expert_layer_offset)

        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)

    @property
    def layers_block_type(self):
        return [
            "attention" if i % self.attn_layer_period == self.attn_layer_offset else "mamba"
            for i in range(self.num_hidden_layers)
        ]

    @property
    def layers_num_experts(self):
        return [
            self.num_experts if i % self.expert_layer_period == self.expert_layer_offset else 1
            for i in range(self.num_hidden_layers)
        ]

    def _check_supported_offset(self, property_: str, period: int, offset: int):
        if offset >= period:
            raise ValueError(
                f"{property_} layer offset ({offset}) must be smaller than {property_} layer period ({period})"
            )


__all__ = ["JambaConfig"]
