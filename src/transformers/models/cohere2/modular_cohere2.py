# coding=utf-8
# Copyright 2024 Cohere Inc. HuggingFace Inc. team. All rights reserved.
#
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
from typing import Callable, Optional

import torch
import torch.nn as nn

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PretrainedConfig, layer_type_validation
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.deprecation import deprecate_kwarg
from ..cohere.modeling_cohere import (
    CohereAttention,
    CohereDecoderLayer,
    CohereForCausalLM,
    CohereLayerNorm,
    CoherePreTrainedModel,
    CohereRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..gemma2.modeling_gemma2 import Gemma2Model


logger = logging.get_logger(__name__)


class Cohere2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CohereModel`]. It is used to instantiate an Cohere
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01) model.


    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Cohere model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CohereModel`]
        hidden_size (`int`, *optional*, defaults to 8192):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22528):
            Dimension of the MLP representations.
        logit_scale (`float`, *optional*, defaults to 0.0625):
            The scaling factor for the output logits.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 5):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 255001):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        sliding_window (`int`, *optional*, defaults to 4096):
            Size of the sliding window attention context.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.

    ```python
    >>> from transformers import Cohere2Model, Cohere2Config

    >>> # Initializing a Cohere Nextmodel configuration
    >>> configuration = Cohere2Config()

    >>> # Initializing a model from the Cohere2 configuration
    >>> model = Cohere2Model(configuration) # doctest: +SKIP

    >>> # Accessing the model configuration
    >>> configuration = model.config # doctest: +SKIP
    ```
    """

    model_type = "cohere2"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
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
        vocab_size=256000,
        hidden_size=8192,
        intermediate_size=22528,
        logit_scale=0.0625,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=5,
        eos_token_id=255001,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=4096,
        layer_types=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.logit_scale = logit_scale
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        # Need to specify head_dim in the config so it can be used in the attention forward functions
        self.head_dim = hidden_size // num_attention_heads

        # Validate the correctness of rotary position embeddings parameters
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
        self._sliding_window_pattern = kwargs.get("sliding_window_pattern", 4)

        if self.layer_types is None:
            # BC -> the pattern used to be a simple int, and it's still present in configs on the Hub
            self._sliding_window_pattern = getattr(self, "sliding_window_pattern", 4)
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)


class Cohere2RotaryEmbedding(CohereRotaryEmbedding):
    pass


class Cohere2LayerNorm(CohereLayerNorm):
    pass


class Cohere2Attention(CohereAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Cohere2Config, layer_idx: Optional[int] = None):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.sliding_window is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Cohere2DecoderLayer(CohereDecoderLayer):
    def __init__(self, config: Cohere2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states_attention, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states_mlp = self.mlp(hidden_states)
        hidden_states = residual + hidden_states_attention + hidden_states_mlp
        return hidden_states


class Cohere2PreTrainedModel(CoherePreTrainedModel):
    config: Cohere2Config


class Cohere2Model(Gemma2Model):
    def __init__(self, config: Cohere2Config):
        super().__init__(config)
        self.norm = Cohere2LayerNorm(hidden_size=(config.hidden_size), eps=config.layer_norm_eps)
        self.rotary_emb = Cohere2RotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Cohere2ForCausalLM(CohereForCausalLM):
    pass


__all__ = ["Cohere2Config", "Cohere2ForCausalLM", "Cohere2Model", "Cohere2PreTrainedModel"]
