# Copyright 2024 Zyphra Technologies and the HuggingFace Inc. team. All rights reserved.
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
"""Zamba model configuration"""

import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="Zyphra/Zamba-7B-v1")
@strict
class ZambaConfig(PreTrainedConfig):
    r"""
    attention_hidden_size (`int`, *optional*):
        Dimension of the hidden representations of the inputs to the Attention layer.
    attention_head_dim (`int`, *optional*):
        Dimension of the attention head in the Transformer decoder.
    n_mamba_heads (`int`, *optional*, defaults to 2):
        Number of mamba heads for each mamba layer.
    hidden_mamba_act (`str` or `function`, *optional*, defaults to `"silu"`):
        The non-linear activation function (function or string) in the mamba layer.
    num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
        Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
        integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
        logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
        sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
        significantly.
    attn_layer_period (`int`, *optional*, defaults to 6):
        Once in this many layers, we will have a shared attention layer
    attn_layer_offset (`int`, *optional*, defaults to 4):
        Offset of the shared attention layer
    use_mamba_kernels (`bool`, *optional*, defaults to `True`):
        Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
        `causal-conv1d` are installed, and the mamba modules are running on a CUDA device. Raises ValueError if
        `True` and kernels are not available
    mamba_dt_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
        Rank of the mamba discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
    """

    model_type = "zamba"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"layer_types": "layers_block_type", "head_dim": "attention_head_dim"}

    vocab_size: int = 32000
    tie_word_embeddings: bool = True
    hidden_size: int = 3712
    attention_hidden_size: int | None = None
    intermediate_size: int = 14848
    num_hidden_layers: int = 76
    num_attention_heads: int = 16
    attention_head_dim: int | None = None
    num_key_value_heads: int = 16
    n_mamba_heads: int = 2
    hidden_act: str = "gelu"
    hidden_mamba_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    num_logits_to_keep: int = 1
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    max_position_embeddings: int = 4096
    attention_dropout: float | int = 0.0
    attn_layer_period: int = 6
    attn_layer_offset: int = 4
    use_mamba_kernels: bool = True
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: str | int = "auto"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    def __post_init__(self, **kwargs):
        self.attention_hidden_size = self.attention_hidden_size or 2 * self.hidden_size
        self.attention_head_dim = self.attention_head_dim or 2 * self.hidden_size // self.num_attention_heads
        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if self.mamba_dt_rank == "auto" else self.mamba_dt_rank
        self.layers_block_type = self._layers_block_type(
            self.num_hidden_layers, self.attn_layer_period, self.attn_layer_offset
        )
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if (self.mamba_expand * self.hidden_size) % self.n_mamba_heads != 0:
            raise ValueError("`intermediate_size` should be divisible by `n_mamba_heads`.")

    def _layers_block_type(self, num_hidden_layers, attn_layer_period, attn_layer_offset):
        layers = [
            "mamba",
            "mamba",
            "hybrid",
        ] + ["hybrid" if i % attn_layer_period == attn_layer_offset else "mamba" for i in range(num_hidden_layers - 3)]
        return layers


__all__ = ["ZambaConfig"]
