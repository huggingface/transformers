# Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Reformer model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/reformer-crime-and-punishment")
@strict
class ReformerConfig(PreTrainedConfig):
    r"""
    attention_head_size (`int`, *optional*, defaults to 64):
        Dimensionality of the projected key, query and value vectors
    attn_layers (`list[str]`, *optional*, defaults to `["local", "lsh", "local", "lsh", "local", "lsh"]`):
        List of attention layer types in ascending order. It can be chosen between a LSHSelfAttention layer
        (`"lsh"`) and a LocalSelfAttention layer (`"local"`).
        For more information on LSHSelfAttention layer, see [LSH Self Attention](reformer#lsh-self-attention). For
        more information on LocalSelfAttention layer, see [Local Self Attention](reformer#local-self-attention).
    axial_norm_std (`float`, *optional*, defaults to 1.0):
        The standard deviation of the normal_initializer for initializing the weight matrices of the axial
        positional encodings.
    axial_pos_embds (`bool`, *optional*, defaults to `True`):
        Whether or not to use axial position embeddings. For more information on how axial position embeddings
        work, see [Axial Position Encodings](reformer#axial-positional-encodings).
    axial_pos_shape (`list[int]`, *optional*, defaults to `[64, 64]`):
        The position dims of the axial position encodings. During training, the product of the position dims has to
        be equal to the sequence length.
        For more information on how axial position embeddings work, see [Axial Position
        Encodings](reformer#axial-positional-encodings).
    axial_pos_embds_dim (`list[int]`, *optional*, defaults to `[64, 192]`):
        The embedding dims of the axial position encodings. The sum of the embedding dims has to be equal to the
        hidden size.
        For more information on how axial position embeddings work, see [Axial Position
        Encodings](reformer#axial-positional-encodings).
    chunk_size_lm_head (`int`, *optional*, defaults to 0):
        The chunk size of the final language model feed forward head layer. A chunk size of 0 means that the feed
        forward layer is not chunked. A chunk size of n means that the feed forward layer processes n <
        sequence_length embeddings at a time.
        For more information on feed forward chunking, see [How does Feed Forward Chunking
        work?](../glossary#feed-forward-chunking).
    feed_forward_size (`int`, *optional*, defaults to 512):
        Dimensionality of the feed_forward layer in the residual attention block.
    hash_seed (`int`, *optional*):
        Seed that can be used to make local sensitive hashing in `LSHSelfAttention` deterministic. This should only
        be set for testing purposed. For evaluation and training purposes `hash_seed` should be left as `None` to
        ensure fully random rotations in local sensitive hashing scheme.
    local_num_chunks_before (`int`, *optional*, defaults to 1):
        Number of previous neighbouring chunks to attend to in `LocalSelfAttention` layer to itself.
    local_num_chunks_after (`int`, *optional*, defaults to 0):
        Number of following neighbouring chunks to attend to in `LocalSelfAttention` layer in addition to itself.
    local_attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities in `LocalSelfAttention`.
    local_attn_chunk_length (`int`, *optional*, defaults to 64):
        Length of each chunk in local attention layers.
    lsh_attn_chunk_length (`int`, *optional*, defaults to 64):
        Length of chunk which attends to itself in `LSHSelfAttention`. Chunking reduces memory complexity from
        sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk
        length (chunked self attention).
    lsh_attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities in `LSHSelfAttention`.
    lsh_num_chunks_before (`int`, *optional*, defaults to 1):
        Number of previous neighbouring chunks to attend to in `LSHSelfAttention` layer to itself.
    lsh_num_chunks_after (`int`, *optional*, defaults to 0):
        Number of following neighbouring chunks to attend to in `LSHSelfAttention` layer to itself.
    num_buckets (`int` or `list[int]`, *optional*):
        Number of buckets, the key query vectors can be "hashed into" using the locality sensitive hashing scheme.
        Each query key vector is hashed into a hash in `1, ..., num_buckets`. The number of buckets can also be
        factorized into a list for improved memory complexity. In this case, each query key vector is hashed into a
        hash in `1-1, 1-2, ..., num_buckets[0]-1, ..., num_buckets[0]-num_buckets[1]` if `num_buckets` is
        factorized into two factors. The number of buckets (or the product the factors) should approximately equal
        sequence length / lsh_chunk_length. If `num_buckets` not set, a good value is calculated on the fly.
    num_hashes (`int`, *optional*, defaults to 1):
        Number of hashing rounds (e.g., number of random rotations) in Local Sensitive Hashing scheme. The higher
        `num_hashes`, the more accurate the `LSHSelfAttention` becomes, but also the more memory and time intensive
        the hashing becomes.

    Examples:

    ```python
    >>> from transformers import ReformerConfig, ReformerModel

    >>> # Initializing a Reformer configuration
    >>> configuration = ReformerConfig()

    >>> # Initializing a Reformer model (with random weights)
    >>> model = ReformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "reformer"
    keys_to_ignore_at_inference = ["past_buckets_states"]
    attribute_map = {}

    attention_head_size: int = 64
    attn_layers: list[str] | tuple[str, ...] = ("local", "lsh", "local", "lsh", "local", "lsh")
    axial_norm_std: float = 1.0
    axial_pos_embds: bool = True
    axial_pos_shape: list[int] | tuple[int, ...] = (64, 64)
    axial_pos_embds_dim: list[int] | tuple[int, ...] = (64, 192)
    chunk_size_lm_head: int = 0
    eos_token_id: int | list[int] | None = 2
    feed_forward_size: int = 512
    hash_seed: int | None = None
    hidden_act: str = "relu"
    hidden_dropout_prob: float | int = 0.05
    hidden_size: int = 256
    initializer_range: float = 0.02
    is_decoder: bool = False
    layer_norm_eps: float = 1e-12
    local_num_chunks_before: int = 1
    local_num_chunks_after: int = 0
    local_attention_probs_dropout_prob: float | int = 0.05
    local_attn_chunk_length: int = 64
    lsh_attn_chunk_length: int | None = 64
    lsh_attention_probs_dropout_prob: float | None = 0.0
    lsh_num_chunks_before: int | None = 1
    lsh_num_chunks_after: int | None = 0
    max_position_embeddings: int = 4096
    num_attention_heads: int = 12
    num_buckets: int | list[int] | None = None
    num_hashes: int = 1
    vocab_size: int = 320
    tie_word_embeddings: bool = False
    use_cache: bool = True
    classifier_dropout: float | int | None = None
    bos_token_id: int | None = None
    pad_token_id: int | None = 0

    def __post_init__(self, **kwargs):
        self.num_hidden_layers = len(self.attn_layers)
        self.axial_pos_shape = tuple(self.axial_pos_shape)
        self.axial_pos_embds_dim = tuple(self.axial_pos_embds_dim)
        super().__post_init__(**kwargs)


__all__ = ["ReformerConfig"]
