# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Callable, Sequence
from typing import List, Literal, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...cache_utils import Cache, HybridCache, StaticCache
from ...configuration_utils import PretrainedConfig
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
)
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...utils import (
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from ..gemma2.modeling_gemma2 import (
    Gemma2ForCausalLM,
    Gemma2MLP,
    Gemma2Model,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from ..paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from ..siglip import SiglipVisionConfig


_CHECKPOINT_FOR_DOC = "google/gemma-3-4b"
_CONFIG_FOR_DOC = "Gemma3Config"

logger = logging.get_logger(__name__)

GEMMA3_INPUTS_DOCSTRING = ""

ATTENTION_TYPE_GLOBAL = "global"
ATTENTION_TYPE_LOCAL = "local_sliding"
AttentionType = Literal["global", "local_sliding"]
AttentionPattern = Sequence[AttentionType]
DEFAULT_ATTENION_PATTERN = cast(
    AttentionPattern,
    (
        ATTENTION_TYPE_LOCAL,
        ATTENTION_TYPE_LOCAL,
        ATTENTION_TYPE_LOCAL,
        ATTENTION_TYPE_LOCAL,
        ATTENTION_TYPE_LOCAL,
        ATTENTION_TYPE_GLOBAL,
    ),
)


class Gemma3TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Gemma3Model`]. It is used to instantiate a Gemma3
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Gemma3-7B.
    e.g. [google/gemma-3-4b](https://huggingface.co/google/gemma-3-4b)
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Gemma3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Gemma3Model`]
        num_hidden_layers (`int`, *optional*, defaults to 26):
            Number of hidden layers in the Transformer decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        hidden_size (`int`, *optional*, defaults to 2304):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 9216):
            Dimension of the MLP representations.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_activation (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the decoder. Will default to
            `"gelu_pytorch_tanh"` if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"`
            activation function.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_local_base_freq (float, *optional*, defaults to `rope_theta`):
            The base period of the RoPE embeddings for local attention.
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
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_pattern (Sequence[AttentionTypes], defaults to (5 * local, global)):
            The attention pattern to apply
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        query_pre_attn_scalar (`float`, *optional*, defaults to None):
            The scaling factor used on the attention scores, not that
        sliding_window (`int`, *optional*, defaults to 4096): in Gemma3, every other layer uses sliding window
            attention. This is the size of the sliding window.
        attn_logit_softcapping (`float`, *optional*, defaults to 50.0): scaling factor when applying tanh soft-capping
            on the attention scorexs.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        cache_implementation (`str`, *optional*, defaults to `"hybrid"`): the cache type to be used with `generate`.

    ```python
    >>> from transformers import Gemma3Model, Gemma3Config
    >>> # Initializing a Gemma3 gemma3-7b style configuration
    >>> configuration = Gemma3Config()
    >>> # Initializing a model from the gemma3-7b style configuration
    >>> model = Gemma3Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gemma3_text"

    def __init__(
        self,
        # Config parameters found in all implementations, name differences noted
        vocab_size: int = 262_144,  # num_embed in FLAX
        hidden_size: int = 2304,  # embed_dim in FLAX
        intermediate_size: int = 9216,  # hidden_dim in FLAX
        num_hidden_layers: int = 26,  # num_layers in FLAX
        num_attention_heads: int = 8,  # num_heads in FLAX
        num_key_value_heads: int = 4,  # num_kv_heads in FLAX
        head_dim: int = 256,
        sliding_window: int = 4096,  # sliding_window_size in FLAX
        query_pre_attn_scalar: Optional[float] = None,
        attention_pattern: AttentionPattern = DEFAULT_ATTENION_PATTERN,
        rope_theta: float = 1_000_000.0,
        rope_scaling = None,
        rope_local_base_freq: float = 10_000.0,
        rms_norm_eps: float = 1e-6,
        hidden_activation: str = "gelu_pytorch_tanh",
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        max_position_embeddings: int = 131_072,
        initializer_range: float = 0.02,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        cache_implementation: str = "hybrid",
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_local_base_freq = rope_local_base_freq
        self.attention_pattern = attention_pattern
        # For configuring HybridCache to work with 5:1 attention pattern
        self.sliding_window_pattern = len(self.attention_pattern)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_activation = hidden_activation
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logit_softcapping = attn_logit_softcapping
        self.cache_implementation = cache_implementation
        rope_config_validation(self)


class Gemma3Config(PretrainedConfig):
    model_type = "gemma3"
    sub_configs = {
        "text_config": Gemma3TextConfig,
        "vision_config": SiglipVisionConfig,
    }

    def __init__(
        self,
        text_config: Optional[Gemma3TextConfig] = None,
        vision_config: Optional[SiglipVisionConfig] = None,
        mm_tokens_per_image: int = 256,
        boi_token_index: int = 255_999,
        eoi_token_index: int = 256_000,
        image_token_index: int = 262_144,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        if text_config is None:
            text_config = Gemma3TextConfig()
            logger.info("text_config is None, using default Gemma3TextConfig vision config.")
        elif isinstance(text_config, dict):
            text_config = Gemma3TextConfig(**text_config)

        if isinstance(vision_config, dict):
            vision_config = SiglipVisionConfig(**vision_config)
        else:
            vision_config = SiglipVisionConfig()
            logger.info(
                "vision_config is None or incompatible with Gemma3VisionConfig intialization. Gemma3 will be limited "
                "to text tasks."
            )

        self.text_config = text_config
        self.vision_config = vision_config
        self.mm_tokens_per_image = mm_tokens_per_image
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


class Gemma3MLP(Gemma2MLP):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)


class Gemma3RMSNorm(Gemma2RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()


class Gemma3RotaryEmbedding(Gemma2RotaryEmbedding):
    def __init__(self, config: Gemma3TextConfig, device=None):
        super().__init__(config)


def create_sliding_window_mask(
    position_ids: torch.LongTensor,
    cache_position: int,
    cache_len: int,
    sliding_window_size: int,
) -> torch.Tensor:
    """Creates mask for sliding window attention."""
    total_tokens = cache_position + position_ids.shape[1]  # cached + processing tokens

    def _reconstruct_rotated_cache_positions():
        cache_positions = torch.arange(cache_len) + total_tokens - cache_len
        rotated_cache_positions = torch.zeros_like(cache_positions)
        rotated_cache_positions[cache_positions % cache_len] = cache_positions
        return rotated_cache_positions

    # Reconstruct position_ids for cached kv.
    if total_tokens <= cache_len:
        cache_positions = torch.arange(cache_len)
    else:
        cache_positions = _reconstruct_rotated_cache_positions()

    cache_positions = cache_positions.unsqueeze(0).unsqueeze(0).to(position_ids.device)  # [1, 1, cache_len]
    position_ids = position_ids.unsqueeze(-1)  # [B, seq_len, 1]
    sliding_mask = cache_positions > position_ids - sliding_window_size
    sliding_mask *= cache_positions < position_ids + sliding_window_size
    return sliding_mask.unsqueeze(1)


def create_sliding_window_mask(
    sliding_window_size: int,
    q_pos: torch.Tensor,
    kv_pos: torch.Tensor,
) -> torch.Tensor:
    """Creates mask for sliding window attention."""
    return q_pos < kv_pos + sliding_window_size


def eager_attention_forward(
    module: "Gemma3Attention",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.attention_dropout = config.attention_dropout
        self.attention_type: AttentionType = config.attention_pattern[layer_idx % len(config.attention_pattern)]
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.is_causal = True
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar
        self.is_sliding = self.attention_type == ATTENTION_TYPE_LOCAL
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if self.attention_type == ATTENTION_TYPE_GLOBAL:
            cos, sin = position_embeddings_global
        else:
            cos, sin = position_embeddings_local

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # Here we need to slice as we use a static cache by default, but FA2 does not support it
            if attention_mask is not None and self.config._attn_implementation == "flash_attention_2":
                seq_len = attention_mask.shape[-1]
                key_states, value_states = key_states[:, :, :seq_len, :], value_states[:, :, :seq_len, :]

        if self.is_sliding and key_states.shape[-2] > self.sliding_window:
            assert self.sliding_window is not None
            if query_states.shape[-2] == key_states.shape[-2]:
                sliding_window_mask = create_sliding_window_mask(
                    sliding_window_size=self.sliding_window,
                    q_pos=torch.arange(query_states.shape[-2]).unsqueeze(-1),
                    kv_pos=torch.arange(key_states.shape[-2]).unsqueeze(-2),
                )
                attention_mask = torch.logical_and(attention_mask, sliding_window_mask)
            else:
                raise ValueError()

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "
                    "Falling back to eager attention. This warning can be removed using the argument "
                    '`attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask.to(query_states),
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.is_sliding = self.self_attn.is_sliding
        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: int = 0,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if not isinstance(past_key_value, HybridCache):
            raise ValueError("Gemma 3 only supports a HybridCache, required for local vs global attention")

        if self.is_sliding and attention_mask is not None:  # efficient SDPA and no padding
            # In prefill, we may be larger than sliding window
            effective_seq_len = max(cache_position.shape[0], self.sliding_window)
            # For FA2, the mask is 2D and is of shape [bs, processed_tokens] (not [bs, max_cache_len]),
            # thus we must slice from the right (at most `effective_seq_len` elements)
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask[:, -effective_seq_len:]
            # Otherwise, the mask is 4D of shape [bs, 1, query_len, max_cache_len] thus we must slice
            # from the left, with an offset if we are beyond the sliding window
            else:
                min_dtype = torch.finfo(attention_mask.dtype).min
                sliding_window_mask = torch.tril(
                    torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.sliding_window
                )
                attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
                # In case we are beyond the sliding window, we need to correctly offset the mask slicing
                # `last_cache_position` is equivalent to `cache_position[-1]` but without breaking dynamo
                offset = last_cache_position - effective_seq_len
                # Should only be used when beyond the sliding window (i.e. offset > 0)
                offset = max(0, offset)
                attention_mask = attention_mask[:, :, :, offset : offset + effective_seq_len]

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Gemma3Model(Gemma2Model):
    config_class = Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)

        # TODO: raushan fix this after RoPE refactor. For now we hack it by reassigning thetas
        # when we want to create a local RoPE layer. Config defaults should hold values for global RoPE
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: Optional[int] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            # normalized
            # Gemma3 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
            # See https://github.com/huggingface/transformers/pull/29402
            normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=inputs_embeds.dtype)
            inputs_embeds = inputs_embeds * normalizer

        if use_cache and past_key_values is None and not self.training:
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = HybridCache(
                self.config,
                max_batch_size=batch_size,
                max_cache_len=seq_len,
                dtype=inputs_embeds.dtype,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # This is needed to correctly slice the mask without data-dependent slicing later on if using dynamo tracing
        # (retrieving the same value from `cache_position` later on would crash dynamo)
        if last_cache_position is None:
            last_cache_position = 0
            if attention_mask is not None:
                # In case a 4d mask is passed directly without using `generate`, we have to rely on cache_position
                # It will break dynamo tracing but there are no way around it (and it should never happen in practice)
                last_cache_position = (
                    attention_mask.shape[-1] if attention_mask.dim() == 2 else cache_position[-1].item()
                )
        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings_global,
                    position_embeddings_local,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    last_cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    last_cache_position=last_cache_position,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class Gemma3ForCausalLM(Gemma2ForCausalLM):
    config_class = Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)


class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        b, _, l = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(b, l, self.patches_per_image, self.patches_per_image)
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.einsum("btm,md->btd", normed_vision_outputs, self.mm_input_projection_weight)
        return projected_vision_outputs.type_as(vision_outputs)

# The only diff on forward is that Gemma3 relies on `input_ids` when creating causal mask, while Paligemma
# passes input embeds. Can be removed when we enable good token type ids
class Gemma3ForConditionalGeneration(PaliGemmaForConditionalGeneration):
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features

    @add_start_docstrings_to_model_forward(GEMMA3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Gemma3CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/Gemma3-test-224px-hf")
        >>> processor = AutoProcessor.from_pretrained("google/Gemma3-test-224px-hf")

        >>> prompt = "answer en Where is the cow standing?"
        >>> url = "https://huggingface.co/gv-hf/Gemma3-test-224px-hf/resolve/main/cow_beach_1.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "answer en Where is the cow standing?\nbeach"
        ```"""

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None

        if inputs_embeds is None:
            special_image_mask = (input_ids == self.config.image_token_index)
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)
            # normalized
            # Gemma3 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
            # See https://github.com/huggingface/transformers/pull/29402
            normalizer = torch.tensor(self.config.text_config.hidden_size**0.5, dtype=inputs_embeds.dtype)
            inputs_embeds = inputs_embeds * normalizer

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Gemma3 positions are 1-indexed

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            special_image_mask = special_image_mask.unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, input_ids, is_training
        )
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        logits = outputs.logits
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        input_tensor,
        is_training: bool = False,
    ):
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted
            # form and requires no inversion or slicing.
            return attention_mask

        min_dtype = torch.finfo(self.dtype).min
        batch_size, sequence_length = input_tensor.shape[:2]
        if isinstance(past_key_values, (HybridCache, StaticCache)):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        causal_mask = torch.ones((sequence_length, target_length), dtype=torch.bool)
        causal_mask = torch.tril(causal_mask)
        causal_mask = causal_mask.to(self.device)

        attention_mask = attention_mask.unsqueeze(-2).to(self.device)
        causal_mask = causal_mask.unsqueeze(0).repeat(attention_mask.shape[0], 1, 1)
        combined_mask = attention_mask * causal_mask[:, :, :sequence_length]

        image_token_mask = input_tensor == self.config.image_token_index
        image_token_mask.to(self.device)
        # logger.warning("image_token_mask shape = %s", image_token_mask.shape)
        padded_mask = nn.functional.pad(image_token_mask, (1, 0), value=0)
        padded_mask = padded_mask.to(self.device)
        # logger.warning("padded_mask shape = %s", padded_mask.shape)
        boundary = padded_mask[:, 1:] > padded_mask[:, :-1]
        boundary = boundary.to(self.device)
        numbered_boundary = torch.cumsum(boundary, dim=-1)
        numbered_boundary = numbered_boundary.to(self.device)
        q_block_indices = image_token_mask * numbered_boundary
        q_block_indices = q_block_indices.to(self.device)
        kv_block_indices = q_block_indices
        # logger.warning("q_block_indices/kv_block_indices shape = %s", q_block_indices.shape)
        bidirectional_mask = torch.logical_and(
            kv_block_indices[:, None, :] == q_block_indices.unsqueeze(-1),
            q_block_indices.unsqueeze(-1) > 0,
        )
        bidirectional_mask.to(self.device)
        attention_mask = torch.logical_or(combined_mask.unsqueeze(1), bidirectional_mask.unsqueeze(1)).to(self.device)
        full_attention_mask = torch.zeros((batch_size, 1, sequence_length, target_length)).to(self.device, torch.bool)
        full_attention_mask[:, :, :, :sequence_length] = attention_mask
        attention_mask = torch.where(full_attention_mask, 0, min_dtype).to(self.device)

        return attention_mask


__all__ = [
    "Gemma3Config",
    "Gemma3TextConfig",
    "Gemma3PreTrainedModel",  # noqa: F822
    "Gemma3Model",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
]
