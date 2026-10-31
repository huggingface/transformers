# Copyright 2026 The RWKV team and The HuggingFace Inc. team. All rights reserved.
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
"""RWKV-7 (Goose) model configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="RWKV/RWKV7-1.5B-20260805")
@strict
class Rwkv7Config(PreTrainedConfig):
    r"""
    num_heads (`int`, *optional*, defaults to 32):
        Number of WKV heads, `hidden_size // head_dim`.
    decay_low_rank_dim (`int`, *optional*, defaults to 96):
        Rank of the decay (`w`) LoRA.
    a_low_rank_dim (`int`, *optional*, defaults to 96):
        Rank of the in-context learning rate (`a`) LoRA.
    v_low_rank_dim (`int`, *optional*, defaults to 64):
        Rank of the value-residual (`v`) LoRA.
    gate_low_rank_dim (`int`, *optional*, defaults to 256):
        Rank of the output gate (`g`) LoRA.
    norm_eps (`float`, *optional*, defaults to 1e-05):
        Epsilon of every LayerNorm/GroupNorm in the model.

    ```python
    >>> from transformers import Rwkv7Config, Rwkv7Model

    >>> configuration = Rwkv7Config()
    >>> model = Rwkv7Model(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "rwkv7"
    attribute_map = {"num_attention_heads": "num_heads"}
    # One cache slot per mixer: conv state = token shift, recurrent state = WKV matrix (attention slot only)
    number_of_conv_states = 2

    vocab_size: int = 65536
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    num_heads: int = 32
    head_dim: int = 64
    decay_low_rank_dim: int = 96
    a_low_rank_dim: int = 96
    v_low_rank_dim: int = 64
    gate_low_rank_dim: int = 256
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    use_cache: bool = True
    bos_token_id: int | None = 0
    eos_token_id: int | None = 0
    pad_token_id: int | None = 0

    @property
    def layer_types(self):
        return ["linear_attention"] * self.num_hidden_layers


__all__ = ["Rwkv7Config"]
