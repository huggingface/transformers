# coding=utf-8
# Copyright 2024 Jingze Shi and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on the Wonderful Matrices paper implementation.
# The Doge family of small language models is trained by Jingze Shi.
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
"""PyTorch Doge model."""

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from ...activations import ACT2FN
from ...cache_utils import Cache, StaticCache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    is_torch_flex_attn_available,
    logging,
)


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import flex_attention


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DogeConfig"


class DogeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DogeModel`]. It is used to instantiate an Doge
    model according to the specified arguments, defining the model architecture like [SmallDoge/Doge-20M](https://huggingface.co/SmallDoge/Doge-20M).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the Doge model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [`DogeModel`]
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        hidden_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the hidden layers.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for each sequence transformation and state transformation module.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        pad_token_id (`int`, *optional*, defaults to 2):
            Padding token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
            NOTE: if you apply new rope type and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value accordingly.
            Doge family of small models use `{ 'rope_type': 'dynamic', 'factor': 4.0, 'original_max_position_embeddings': 2048 }` as the default value.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope', 'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings.
                    In most scaling types, a `factor` of x will enable the model to handle sequences of length x * original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'.
                    The original max position embeddings used during pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation.
                    If unspecified, it defaults to value recommended by the implementation, using the `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<`original_max_position_embeddings`).
                    Must be a list of numbers with the same length as the hidden size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<`original_max_position_embeddings`).
                    Must be a list of numbers with the same length as the hidden size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention.
            If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used.
            When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group.
            For more details checkout [this paper](https://arxiv.org/pdf/2305.13245.pdf).
            If it is not specified, will default to `num_attention_heads`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        dynamic_mask_ratio (`float`, *optional*, defaults to 0.0):
            The ratio to control the proportion of the dynamic mask filled with the minimum value. For more details checkout [this paper](https://arxiv.org/pdf/2412.11834).
        is_moe (`bool`, *optional*, defaults to `False`):
            Whether to use the Cross Domain Mixture of Experts, if `True`, the MoE will inherit the MLP to initialize. For more details checkout [this paper](https://arxiv.org/pdf/2412.11834).
        num_experts (`int`, *optional*, defaults to 2048):
            Number of Experts for the Cross Domain Mixture of Experts.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts to route per-token.
        expert_retrieval_size (`int`, *optional*, defaults to 64):
            Dimension of the Expert retrieval states for calculating the dot product of query and key to determine the expert index.

    ```python
    >>> from transformers import DogeConfig, DogeModel

    >>> # Initializing a Doge-320M style configuration
    >>> configuration = DogeConfig()

    >>> # Initializing a model from the Doge-320M style configuration
    >>> model = DogeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "doge"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `DogeModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.dt_proj": "rowwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=32,
        hidden_bias=False,
        hidden_dropout=0.0,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        tie_word_embeddings=False,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_scaling=None,
        num_attention_heads=8,
        num_key_value_heads=None,
        attention_dropout=0.0,
        dynamic_mask_ratio=0.0,
        is_moe=False,
        num_experts=2048,
        num_experts_per_tok=8,
        expert_retrieval_size=64,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        self.hidden_bias = hidden_bias
        self.hidden_dropout = hidden_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_dropout = attention_dropout
        self.dynamic_mask_ratio = dynamic_mask_ratio
        self.is_moe = is_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_retrieval_size = expert_retrieval_size

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # for backward compatibility
        if num_key_value_heads is None:
            self.num_key_value_heads = num_attention_heads

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class DogeRMSNorm(LlamaRMSNorm):
    pass


ALL_LAYERNORM_LAYERS.append(DogeRMSNorm)


class DogeResidual(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, residual_states, hidden_states):
        return self.weight * residual_states + hidden_states

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}"


class DogeRotaryEmbedding(LlamaRotaryEmbedding):
    pass


def flex_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    if is_causal is None:
        is_causal = causal_mask is None and query.shape[2] > 1

    def causal_mod(score, b, h, q_idx, kv_idx):
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        if causal_mask is not None:
            score = score + causal_mask[b][h][q_idx][kv_idx]
        if head_mask is not None:
            score = score + head_mask[b][h][0][0]
        return score

    attn_output, attention_weights = flex_attention(
        query=query,
        key=key,
        value=value,
        score_mod=causal_mod,
        enable_gqa=True,
        scale=scaling,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse regardless.
        # For simplification, we thus always return it as no additional computations are introduced.
        return_lse=True,
    )
    # lse is returned in float32
    attention_weights = attention_weights.to(value.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights


ALL_ATTENTION_FUNCTIONS.update({"flex_attention": flex_attention_forward})


class DogeDynamicMaskAttention(nn.Module):
    """Dynamic Mask Attention from 'Wonderful Matrices' paper."""

    def __init__(self, config: DogeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.dynamic_mask_ratio = config.dynamic_mask_ratio

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.hidden_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.hidden_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.hidden_bias
        )
        # dynamic mask for the QK^T attention weights matrix
        self.A = nn.Parameter(torch.zeros(config.num_attention_heads))
        self.dt_proj = nn.Linear(
            config.num_key_value_heads * self.head_dim, config.num_attention_heads, bias=config.hidden_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.hidden_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # calculate dynamic mask from value_states
        dt_states = self.dt_proj(
            value_states.transpose(1, 2).reshape(value_states.shape[0], value_states.shape[-2], -1)
        )
        dynamic_mask = torch.exp(self.A * F.softplus(dt_states)).transpose(-1, -2)
        attn_mask = self.prepare_dynamic_mask(
            hidden_states=hidden_states,
            dynamic_mask=dynamic_mask,
            dynamic_mask_ratio=self.dynamic_mask_ratio,
            attention_mask=attention_mask,
        )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attn_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def prepare_dynamic_mask(
        self,
        hidden_states: torch.Tensor,
        dynamic_mask: torch.Tensor,
        dynamic_mask_ratio: float = 0.0,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Combine `dynamic_mask` with `attention_mask` to generate the final `attn_mask`.

        Args:
            hidden_states (`torch.Tensor`): The input hidden_states, used to determine the minimum value of the current input precision.
            dynamic_mask (`torch.Tensor`): dynamic mask of shape `(batch_size, num_heads, key_sequence_length)`.
            dynamic_mask_ratio (`float`, *optional*): Ratio from 0.0 to 1.0 used to control the proportion of the dynamic mask filled with the minimum value.
            attention_mask (`torch.Tensor`, *optional*): attention mask of shape `(batch_size, 1, query_sequence_length, key_sequence_length)`.
        """
        attn_mask = dynamic_mask[:, :, None, :]
        if 0.0 < dynamic_mask_ratio < 1.0:
            min_type = torch.finfo(hidden_states.dtype).min
            num_dynamic_mask = int(attn_mask.shape[-1] * dynamic_mask_ratio)
            if num_dynamic_mask > 0:
                rate_value = torch.kthvalue(attn_mask, num_dynamic_mask, dim=-1, keepdim=True).values
                attn_mask = attn_mask.masked_fill(attn_mask < rate_value, min_type)
        else:
            ValueError("`dynamic_mask_ratio` should be in the range (0.0, 1.0)")
        if attention_mask is not None:
            attn_mask = attn_mask + attention_mask[:, :, :, : attn_mask.shape[-1]]

        return attn_mask


class DogeMLP(LlamaMLP):
    def __init__(self, config: DogeConfig):
        super().__init__()

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.hidden_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.hidden_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.hidden_bias)


class DogeCDMoE(DogeMLP):
    """Cross Domain Mixture of Experts from 'Wonderful Matrices' paper."""

    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.hidden_dim = config.hidden_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.expert_retrieval_dim = config.expert_retrieval_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.num_keys = int(math.sqrt(self.num_experts))

        # queries and keys for retrieval experts
        self.queries_proj = nn.Linear(self.hidden_dim, self.expert_retrieval_dim, bias=False)
        self.keys = nn.Parameter(torch.zeros(2, self.expert_retrieval_dim // 2, self.num_keys))

        # experts
        self.down_embed = nn.Embedding(self.num_experts, self.hidden_dim)
        self.up_embed = nn.Embedding(self.num_experts, self.hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        # get routing weights with queries and keys
        queries = self.queries_proj(hidden_states).view(2, bsz * seq_len, -1)
        routing_weights = torch.bmm(queries, self.keys).view(2, bsz * seq_len, self.num_keys)

        # get experts with the highest routing weights
        (scores_x, scores_y), (indices_x, indices_y) = routing_weights.topk(self.top_k, dim=-1)
        all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
        all_scores = all_scores.view(*scores_x.shape[:-1], -1)
        all_indices = (indices_x.unsqueeze(-1) * self.num_keys) + indices_y.unsqueeze(-2)
        all_indices = all_indices.view(*indices_x.shape[:-1], -1)
        scores, pk_indices = all_scores.topk(self.top_k, dim=-1)
        indices = all_indices.gather(-1, pk_indices)
        down_embed = self.down_embed(indices).transpose(1, 2)
        up_embed = self.up_embed(indices)

        # mix experts states with cross domain states
        experts_weights = torch.bmm(hidden_states.view(bsz * seq_len, 1, -1), down_embed).view(bsz * seq_len, -1)
        experts_weights = self.act_fn(experts_weights) * scores.softmax(dim=-1)
        experts_states = torch.bmm(experts_weights.view(bsz * seq_len, 1, -1), up_embed).view(bsz, seq_len, -1)
        hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = hidden_states + experts_states
        return hidden_states


class DogeDecoderLayer(nn.Module):
    def __init__(self, config: DogeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_dropout = config.hidden_dropout

        self.pre_layernorm = DogeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DogeDynamicMaskAttention(config=config, layer_idx=layer_idx)
        self.pre_residual = DogeResidual(config.hidden_size)

        self.post_layernorm = DogeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = DogeMLP(config) if not config.is_moe else DogeCDMoE(config)
        self.post_residual = DogeResidual(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # sequence transformation
        residual = hidden_states
        hidden_states = self.pre_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        self_attn_weights = None
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        hidden_states = self.pre_residual(residual, hidden_states)

        # state transformation
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        hidden_states = self.post_residual(residual, hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class DogePreTrainedModel(LlamaPreTrainedModel):
    config_class = DogeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DogeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True


class DogeModel(DogePreTrainedModel, LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DogeDecoderLayer`]

    Args:
        config: DogeConfig
    """

    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [DogeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DogeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DogeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # We have to provide attention_mask for dynamic mask computation
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class DogeForCausalLM(DogePreTrainedModel, GenerationMixin, LlamaForCausalLM):
    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.config = config
        self.model = DogeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class DogeForSequenceClassification(LlamaForSequenceClassification):
    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.model = DogeModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


__all__ = [
    "DogeConfig",
    "DogeForCausalLM",
    "DogeModel",
    "DogePreTrainedModel",
    "DogeForSequenceClassification",
]
