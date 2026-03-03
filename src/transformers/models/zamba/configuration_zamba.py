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
from dataclasses import dataclass

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig


@strict(accept_kwargs=True)
@dataclass(repr=False)
class ZambaConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ZambaModel`]. It is used to instantiate a
    Zamba model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Zamba-v0.1 model.

    [Zyphra/Zamba-7B-v1](https://huggingface.co/Zyphra/Zamba-7B-v1)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Zamba model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ZambaModel`]
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        hidden_size (`int`, *optional*, defaults to 3712):
            Dimension of the hidden representations.
        attention_hidden_size (`int`, *optional*):
            Dimension of the hidden representations of the inputs to the Attention layer.
        intermediate_size (`int`, *optional*, defaults to 14848):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 76):
            Number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        attention_head_dim (`int`, *optional*):
            Dimension of the attention head in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=None`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245).
        n_mamba_heads (`int`, *optional*, defaults to 2):
            Number of mamba heads for each mamba layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder.
        hidden_mamba_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the mamba layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
            integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
            logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
            sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
            significantly.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            This value doesn't have any real effect. The maximum sequence length that this model is intended to be
            used with. It can be used with longer sequences, but performance may degrade.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attn_layer_period (`int`, *optional*, defaults to 6):
            Once in this many layers, we will have a shared attention layer
        attn_layer_offset (`int`, *optional*, defaults to 4):
            Offset of the shared attention layer
        use_mamba_kernels (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
            `causal-conv1d` are installed, and the mamba modules are running on a CUDA device. Raises ValueError if
            `True` and kernels are not available
        mamba_d_state (`int`, *optional*, defaults to 16):
            The dimension the mamba state space latents
        mamba_d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel
        mamba_expand (`int`, *optional*, defaults to 2):
            Expanding factor (relative to hidden_size) used to determine the mamba intermediate size
        mamba_dt_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the mamba discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj_bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj_bias`.
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the mamba mixer block

    """

    model_type = "zamba"
    keys_to_ignore_at_inference = ["past_key_values"]

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
    eos_token_id: int | None = 2
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
