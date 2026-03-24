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


from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="LiquidAI/LFM2-1.2B")
@strict
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
    ```
    """

    model_type = "lfm2"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 1000000.0

    vocab_size: int = 65536
    hidden_size: int = 2560
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 128_000
    initializer_range: float = 0.02
    norm_eps: float = 0.00001
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    conv_bias: bool = False
    conv_L_cache: int = 3
    block_multiple_of: int = 256
    block_ffn_dim_multiplier: float = 1.0
    block_auto_adjust_ff_dim: bool = True
    full_attn_idxs: list[int] | None = None
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.full_attn_idxs = (
                self.full_attn_idxs if self.full_attn_idxs is not None else list(range(self.num_hidden_layers))
            )
            self.layer_types = [
                "full_attention" if i in self.full_attn_idxs else "conv" for i in range(self.num_hidden_layers)
            ]

        self.tie_word_embeddings = kwargs.pop("tie_embedding", self.tie_word_embeddings)
        self.intermediate_size = kwargs.pop("block_ff_dim", self.intermediate_size)
        super().__post_init__(**kwargs)


__all__ = ["Lfm2Config"]
