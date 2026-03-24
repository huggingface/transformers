# Copyright 2023 HuggingFace Inc. team and MosaicML NLP team.
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
"""Mpt configuration"""

from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="mosaicml/mpt-7b")
@strict
class MptAttentionConfig(PreTrainedConfig):
    r"""
    attn_type (`str`, *optional*, defaults to `"multihead_attention"`):
        type of attention to use. Options: `"multihead_attention"`, `"multiquery_attention"`.
    attn_pdrop (`float`, *optional*, defaults to `0.0`):
        The dropout probability for the attention layers.
    attn_impl (`str`, *optional*, defaults to `"torch"`):
        The attention implementation to use. One of `"torch"`, `"flash"`, or `"triton"`.
    clip_qkv (`float`, *optional*):
        If not `None`, clip the queries, keys, and values in the attention layer to this value.
    softmax_scale (`float`, *optional*):
        If not `None`, scale the softmax in the attention layer by this value. If `None`, will default to
        `1/sqrt(hidden_size)`.
    prefix_lm (`bool`, *optional*, defaults to `False`):
        Whether the model should operate as a Prefix LM. This requires passing an extra `prefix_mask` argument
        which indicates which tokens belong to the prefix. Tokens in the prefix can attend to one another
        bi-directionally. Tokens outside the prefix use causal attention.
    qk_ln (`bool`, *optional*, defaults to `False`):
        Whether to apply layer normalization to the queries and keys in the attention layer.
    attn_uses_sequence_id (`bool`, *optional*, defaults to `False`):
        Whether to restrict attention to tokens that have the same token_type_ids. When the model is in `train`
        mode, this requires passing an extra *token_type_ids* argument which indicates which sub-sequence each
        token belongs to. Defaults to `False` meaning any provided *token_type_ids* will be ignored.
    alibi (`bool`, *optional*, defaults to `True`):
        Whether or not to use the alibi bias instead of positional embedding.
    alibi_bias_max (`int`, *optional*, defaults to 8):
        The maximum value of the alibi bias.
    """

    base_config_key = "attn_config"

    attn_type: Literal["multihead_attention", "multiquery_attention"] = "multihead_attention"
    attn_pdrop: int = 0
    attn_impl: str = "torch"
    clip_qkv: float | None = None
    softmax_scale: float | None = None
    prefix_lm: bool = False
    qk_ln: bool = False
    attn_uses_sequence_id: bool = False
    alibi: bool = True
    alibi_bias_max: int = 8


@auto_docstring(checkpoint="mosaicml/mpt-7b")
@strict
class MptConfig(PreTrainedConfig):
    r"""
    expansion_ratio (`int`, *optional*, defaults to 4):
        The ratio of the up/down scale in the MLP.
    max_seq_len (`int`, *optional*, defaults to 2048):
        The maximum sequence length of the model.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon to use in the layer normalization layers.
    learned_pos_emb (`bool`, *optional*, defaults to `True`):
        Whether to use learned positional embeddings.
    attn_config (`dict`, *optional*):
        A dictionary used to configure the model's attention module.
    init_device (`str`, *optional*, defaults to `"cpu"`):
        The device to use for parameter initialization. Defined for backward compatibility
    logit_scale (`float`, *optional*):
        If not None, scale the logits by this value.
    no_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in all linear layers.
    embedding_fraction (`float`, *optional*, defaults to 1.0):
        The fraction to scale the gradients of the embedding layer by.
    norm_type (`str`, *optional*, defaults to `"low_precision_layernorm"`):
        Type of layer norm to use. All MPT models uses the same layer norm implementation. Defined for backward
        compatibility.

    Example:

    ```python
    >>> from transformers import MptConfig, MptModel

    >>> # Initializing a Mpt configuration
    >>> configuration = MptConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MptModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "mpt"
    sub_configs = {"attn_config": MptAttentionConfig}
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
    }

    d_model: int = 2048
    n_heads: int = 16
    n_layers: int = 24
    expansion_ratio: int = 4
    max_seq_len: int = 2048
    vocab_size: int = 50368
    resid_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    emb_pdrop: float = 0.0
    learned_pos_emb: bool = True
    attn_config: dict | MptAttentionConfig | None = None
    init_device: str = "cpu"
    logit_scale: float | str | None = None
    no_bias: bool = True
    embedding_fraction: float = 1.0
    norm_type: str = "low_precision_layernorm"
    use_cache: bool = False
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.attn_config is None:
            self.attn_config = MptAttentionConfig()
        elif isinstance(self.attn_config, dict):
            self.attn_config = MptAttentionConfig(**self.attn_config)
        super().__post_init__(**kwargs)


__all__ = ["MptConfig"]
