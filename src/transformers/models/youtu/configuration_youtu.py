# Copyright 2025 Tencent Youtu Lab and the HuggingFace Inc. team. All rights reserved.

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
"""Youtu-LLM model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters


class YoutuConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`YoutuModel`]. It is used to instantiate an Youtu
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Youtu-LLM-2B.
    e.g. [tencent/Youtu-LLM-2B](https://huggingface.co/tencent/Youtu-LLM-2B)
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`YoutuModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            In MLA, num_key_value_heads=num_attention_heads.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the LoRA matrices for key and value projections.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the LoRA matrices for query projections.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of the query/key heads that use rotary position embeddings.
        v_head_dim (`int`, *optional*, defaults to 128):
            Dimension of the value heads.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            Dimension of the query/key heads that don't use rotary position embeddings.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to None):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices, except embedding matrices.
        embedding_initializer_range (`float`, *optional*, defaults to None):
            The standard deviation of the truncated_normal_initializer for initializing all embedding matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 128000):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 128001):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether to interleave the rotary position embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    ```python
    >>> from transformers import YoutuModel, YoutuConfig
    >>> # Initializing a Youtu-LLM-2B style configuration
    >>> configuration = YoutuConfig()
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "youtu"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 128256,
        hidden_size: int | None = 2048,
        intermediate_size: int | None = 6144,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = 16,
        kv_lora_rank: int | None = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 128,
        qk_nope_head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 131072,
        initializer_range: float | None = None,
        embedding_initializer_range: float | None = None,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 128000,
        eos_token_id: int | None = 128001,
        tie_word_embeddings: bool | None = True,
        rope_parameters: RopeParameters | dict[str, RopeParameters] = None,
        rope_interleave: bool | None = True,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.rope_interleave = rope_interleave

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.mlp_bias = False
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        # if initializer_range is None, set it to 2.0 / (5.0 * self.hidden_size) ** 0.5
        if self.hidden_size != 0:
            self.initializer_range = (
                (2.0 / (5.0 * self.hidden_size)) ** 0.5 if initializer_range is None else initializer_range
            )
        else:
            self.initializer_range = 0.02
        # if embedding_initializer_range is None, set it to 2.0 * self.initializer_range
        self.embedding_initializer_range = (
            self.initializer_range * 2.0 if embedding_initializer_range is None else embedding_initializer_range
        )
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["YoutuConfig"]
