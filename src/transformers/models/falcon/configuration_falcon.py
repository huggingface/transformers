# Copyright 2023 the Falcon authors and HuggingFace Inc. team.  All rights reserved.
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
"""Falcon configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="tiiuae/falcon-7b")
@strict
class FalconConfig(PreTrainedConfig):
    r"""
    num_ln_in_parallel_attn (`int`, *optional*):
        Set to 2 if separate layer norms are to be used for the MLP and the attention output when using parallel
        attention, otherwise, 1.
    alibi (`bool`, *optional*, defaults to `False`):
        Whether to use ALiBi positional biases during self-attention.
    new_decoder_architecture (`bool`, *optional*, defaults to `False`):
        Whether to use the new (Falcon-40B) decoder architecture. If `True`, the `multi_query` and `parallel_attn`
        arguments are ignored, as the new decoder always uses parallel attention.
    multi_query (`bool`, *optional*, defaults to `True`):
        Whether to use multi-query attention in the decoder. Ignored when `new_decoder_architecture` is `True`.
    parallel_attn (`bool`, *optional*, defaults to `True`):
        Whether to compute attention in parallel with the feedforward layer. If False, they are consecutive
        instead, as in the original Transformer architecture. Ignored when `new_decoder_architecture` is `True`.
    bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias on Linear layers.
    ffn_hidden_size (`int`, *optional*):
        The hidden size of the feedforward layer in the Transformer decoder.
        defaults to 4x hidden dim
    activation (`str`, *optional*, defaults to `"gelu"`):
        The activation function used in the feedforward layer.

    Example:

    ```python
    >>> from transformers import FalconModel, FalconConfig

    >>> # Initializing a small (2-layer) Falcon configuration
    >>> configuration = FalconConfig(num_hidden_layers=2)

    >>> # Initializing a model from the small configuration
    >>> model = FalconModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "falcon"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 65024
    hidden_size: int = 4544
    num_hidden_layers: int = 32
    num_attention_heads: int = 71
    num_ln_in_parallel_attn: int | None = None
    layer_norm_epsilon: float | None = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    hidden_dropout: float | int | None = 0.0
    attention_dropout: float | int | None = 0.0
    num_kv_heads: int | None = None
    alibi: bool | None = False
    new_decoder_architecture: bool | None = False
    multi_query: bool | None = True
    parallel_attn: bool | None = True
    bias: bool | None = False
    max_position_embeddings: int = 2048
    rope_parameters: RopeParameters | dict | None = None
    bos_token_id: int | None = 11
    eos_token_id: int | list[int] | None = 11
    pad_token_id: int | None = None
    ffn_hidden_size: int | None = None
    activation: str | None = "gelu"
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = self.hidden_size if n_embed is None else n_embed
        self.num_kv_heads = self.num_attention_heads if self.num_kv_heads is None else self.num_kv_heads
        if self.ffn_hidden_size is None:
            self.ffn_hidden_size = self.hidden_size * 4

        super().__post_init__(**kwargs)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary(self):
        return not self.alibi


__all__ = ["FalconConfig"]
