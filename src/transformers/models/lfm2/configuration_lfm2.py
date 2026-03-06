# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="LiquidAI/LFM2-1.2B")
class Lfm2Config(PreTrainedConfig):
    r"""
    conv_bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias in the conv layers.
    conv_L_cache (`int`, *optional*, defaults to 3):
        L_cache dim in the conv layers.
    block_multiple_of (`int`, *optional*, defaults to 256):
        Multiple for the `intermediate_size`.
    block_ffn_dim_multiplier (`float`, *optional*, defaults to 1.0):
        Multiplier for the `intermediate_size`.
    block_auto_adjust_ff_dim (`bool`, *optional*, defaults to `True`):
        Whether to adjust the dim of the `intermediate_size`.
    full_attn_idxs (`Optional`, *optional*):
        Index of the layers which use attention.


    ```python
    >>> from transformers import Lfm2Model, Lfm2Config

    >>> # Initializing a LFM2 model
    >>> configuration = Lfm2Config()

    >>> # Initializing a model from the LFM2-1.2B style configuration
    >>> model = Lfm2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "lfm2"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    def __init__(
        self,
        vocab_size: int | None = 65536,
        hidden_size: int | None = 2560,
        intermediate_size: int | None = 12288,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 8,
        max_position_embeddings: int | None = 128_000,
        initializer_range: float | None = 0.02,
        norm_eps: float | None = 0.00001,
        use_cache: bool | None = True,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = True,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        conv_bias: bool | None = False,
        conv_L_cache: int | None = 3,
        block_multiple_of: int | None = 256,
        block_ffn_dim_multiplier: float | None = 1.0,
        block_auto_adjust_ff_dim: bool | None = True,
        full_attn_idxs: list[int] | None = None,
        layer_types: list[str] | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range

        # attn operator config
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # custom operator config
        self.conv_bias = conv_bias
        self.conv_L_cache = conv_L_cache

        # MLP config
        self.intermediate_size = kwargs.get("block_ff_dim", intermediate_size)  # to fit original config keys
        self.block_multiple_of = block_multiple_of
        self.block_ffn_dim_multiplier = block_ffn_dim_multiplier
        self.block_auto_adjust_ff_dim = block_auto_adjust_ff_dim

        self.layer_types = layer_types
        if self.layer_types is None:
            full_attn_idxs = full_attn_idxs if full_attn_idxs is not None else list(range(num_hidden_layers))
            self.layer_types = ["full_attention" if i in full_attn_idxs else "conv" for i in range(num_hidden_layers)]

        self.rope_parameters = rope_parameters
        tie_word_embeddings = kwargs.get("tie_embedding", tie_word_embeddings)  # to fit original config keys
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        super().__init__(**kwargs)


__all__ = ["Lfm2Config"]
