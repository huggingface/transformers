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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="ai21labs/Jamba-v0.1")
@strict
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
    use_mamba_kernels (`bool`, *optional*, defaults to `True`):
        Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
        `causal-conv1d` are installed, and the mamba modules are running on a CUDA device. Raises ValueError if
        `True` and kernels are not available
    mamba_dt_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
        Rank of the mamba discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
    """

    model_type = "jamba"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "num_experts",
    }

    vocab_size: int = 65536
    tie_word_embeddings: bool = False
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    max_position_embeddings: int = 262144
    attention_dropout: float | int = 0.0
    num_experts_per_tok: int = 2
    num_experts: int = 16
    expert_layer_period: int = 2
    expert_layer_offset: int = 1
    attn_layer_period: int = 8
    attn_layer_offset: int = 4
    use_mamba_kernels: bool = True
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: int | str = "auto"
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if self.mamba_dt_rank == "auto" else self.mamba_dt_rank
        super().__post_init__(**kwargs)

    @property
    def layers_block_type(self):
        return [
            "attention" if i % self.attn_layer_period == self.attn_layer_offset else "mamba"
            for i in range(self.num_hidden_layers)
        ]

    @property
    def layer_types(self):
        # Follow the `layer_types` conventions
        layer_types = self.layers_block_type
        return ["full_attention" if x == "attention" else x for x in layer_types]

    @property
    def layers_num_experts(self):
        return [
            self.num_experts if i % self.expert_layer_period == self.expert_layer_offset else 1
            for i in range(self.num_hidden_layers)
        ]

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.attn_layer_offset >= self.attn_layer_period:
            raise ValueError(
                f"attention layer offset ({self.attn_layer_offset}) must be smaller than attention layer period ({self.attn_layer_period})"
            )

        if self.expert_layer_offset >= self.expert_layer_period:
            raise ValueError(
                f"expert layer offset ({self.expert_layer_offset}) must be smaller than expert layer period ({self.expert_layer_period})"
            )


__all__ = ["JambaConfig"]
