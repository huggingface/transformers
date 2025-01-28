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
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import PretrainedConfig, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    LossKwargs,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from ...utils.deprecation import deprecate_kwarg


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import flex_attention


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DogeConfig"


class DogeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DogeModel`]. It is used to instantiate an Doge
    model according to the specified arguments, defining the model architecture like [JingzeShi/Doge-20M](https://huggingface.co/JingzeShi/Doge-20M).

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
        rope_scaling (`Dict`, *optional*, defaults to `{'rope_type': 'dynamic', 'factor': 4.0, 'original_max_position_embeddings': 2048}`):
            Dictionary containing the scaling configuration for the RoPE embeddings.
            NOTE: if you apply new rope type and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value accordingly.
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
        num_cdmoe_experts (`int`, *optional*, defaults to 16348):
            Number of Experts for the Cross Domain Mixture of Experts.
        num_cdmoe_heads (`int`, *optional*, defaults to 4):
            Number of retrieval heads, used to mix multi-head experts.
        num_cdmoe_experts_per_head (`int`, *optional*, defaults to 8):
            Number of Experts per retrieval head, used to mix multi-head experts.
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
        "layers.*.self_attn.dt_proj": "colwise",
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
        rope_scaling={
            "rope_type": "dynamic",
            "factor": 4.0,
            "original_max_position_embeddings": 2048,
        },
        num_attention_heads=8,
        num_key_value_heads=None,
        attention_dropout=0.0,
        dynamic_mask_ratio=0.0,
        is_moe=False,
        num_cdmoe_experts=16348,
        num_cdmoe_heads=4,
        num_cdmoe_experts_per_head=8,
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
        self.num_cdmoe_experts = num_cdmoe_experts
        self.num_cdmoe_heads = num_cdmoe_heads
        self.num_cdmoe_experts_per_head = num_cdmoe_experts_per_head
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


class RMSNorm(LlamaRMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size, eps)


class Residual(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, residual_states, hidden_states):
        return self.weight * residual_states + hidden_states

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}"


class RotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: Optional[DogeConfig] = None, device=None):
        super().__init__(config, device)


def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k.
            For example, note that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim].
            Then, if q and k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k.
            Similarly, if q and k have the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class DogeDynamicMaskAttention(nn.Module):
    """Dynamic Mask Attention from 'Wonderful Matrices' paper."""

    def __init__(self, config: DogeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.dynamic_mask_ratio = config.dynamic_mask_ratio

        self.ALL_ATTENTION_FUNCTIONS = {
            "eager": self.eager_attention_forward,
            "flex_attention": self.flex_attention_forward,
            "sdpa": self.sdpa_attention_forward,
        }

        # Q K V O projections
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.hidden_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.hidden_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.hidden_bias
        )
        # dynamic mask for the QK^T attention score matrix
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
    ) -> Tuple[torch.Tensor, Optional[Cache]]:
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

        attention_interface: Callable = self.eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = self.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output = attention_interface(
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
        return attn_output

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
        attn_mask = None
        if dynamic_mask is not None:
            attn_mask = dynamic_mask[:, :, None, :]
            if 0.0 < dynamic_mask_ratio < 1.0:
                min_type = torch.finfo(hidden_states.dtype).min
                num_dynamic_mask = int(attn_mask.shape[-1] * dynamic_mask_ratio)
                if num_dynamic_mask > 0:
                    rate_value = torch.kthvalue(attn_mask, num_dynamic_mask, dim=-1, keepdim=True).values
                    attn_mask = attn_mask.masked_fill(attn_mask < rate_value, min_type)
            if attention_mask is not None:
                attn_mask = attn_mask + attention_mask[:, :, :, : attn_mask.shape[-1]]
        else:
            attn_mask = attention_mask

        return attn_mask

    def eager_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        key_states = repeat_kv(key, self.num_key_value_groups)
        value_states = repeat_kv(value, self.num_key_value_groups)

        # compute attention scores matrix
        attn_weights = torch.matmul(query, key_states.transpose(-1, -2)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention scores to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)

        # apply attention scores to value states
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    def sdpa_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]

        # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # NOTE: As of pytorch 2.5.1, cuDNN's SDPA backward pass is still incorrect, so we disable cuDNN SDPA (see https://github.com/pytorch/pytorch/issues/138581)
        torch.backends.cuda.enable_cudnn_sdp(False)
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask,
            dropout_p=dropout,
            scale=scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    def flex_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key.shape[-2]]

        # TODO: flex_attention: As of pytorch 2.5.1, captured buffers that require grad are not yet supported.
        # NOTE: So we only use flex_attention in inference mode.
        def causal_mod(score, batch, head, q_idx, kv_idx):
            score = score + causal_mask[batch][0][q_idx][kv_idx]
            return score

        def dynamic_mod(score, batch, head, q_idx, kv_idx):
            score = score + causal_mask[batch][head][q_idx][kv_idx]
            return score

        mask_mod = causal_mod if self.is_causal else dynamic_mod

        attn_output = flex_attention(
            query,
            key,
            value,
            score_mod=mask_mod,
            scale=scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output


class DogeMLP(nn.Module):
    def __init__(self, config: DogeConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=config.hidden_bias)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=config.hidden_bias)
        self.down_proj = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=config.hidden_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return hidden_states


class DogeCDMoE(DogeMLP):
    """Cross Domain Mixture of Experts from 'Wonderful Matrices' paper."""

    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.hidden_dim = config.hidden_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.expert_retrieval_dim = config.expert_retrieval_size
        self.num_cdmoe_experts = config.num_cdmoe_experts
        self.num_cdmoe_heads = config.num_cdmoe_heads
        self.num_cdmoe_experts_per_head = config.num_cdmoe_experts_per_head
        self.num_keys = int(math.sqrt(self.num_cdmoe_experts))

        # queries and keys for retrieval experts
        self.queries = nn.Linear(self.hidden_dim, self.num_cdmoe_heads * self.expert_retrieval_dim, bias=False)
        self.keys = nn.Parameter(torch.zeros(self.num_cdmoe_heads, self.num_keys, 2, self.expert_retrieval_dim // 2))

        # experts
        self.down_embed = nn.Embedding(self.num_cdmoe_experts, self.hidden_dim)
        self.up_embed = nn.Embedding(self.num_cdmoe_experts, self.hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        # get similarity with queries and keys
        queries = self.queries(hidden_states)
        queries = queries.view(bsz, seq_len, 2, self.num_cdmoe_heads, -1).permute(2, 0, 1, 3, 4)
        sim = torch.einsum("p b t h n, h k p n -> p b t h k", queries, self.keys)

        # get experts with the highest similarity
        (scores_x, scores_y), (indices_x, indices_y) = sim.topk(self.num_cdmoe_experts_per_head, dim=-1)
        all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
        all_scores = all_scores.view(*scores_x.shape[:-1], -1)
        all_indices = (indices_x.unsqueeze(-1) * self.num_keys) + indices_y.unsqueeze(-2)
        all_indices = all_indices.view(*indices_x.shape[:-1], -1)
        scores, pk_indices = all_scores.topk(self.num_cdmoe_experts_per_head, dim=-1)
        indices = all_indices.gather(-1, pk_indices)
        down_embed = self.down_embed(indices)
        up_embed = self.up_embed(indices)

        # mix experts states with cross domain states
        experts_weights = torch.einsum("b t d, b t h k d -> b t h k", hidden_states, down_embed)
        experts_weights = self.act_fn(experts_weights) * scores.softmax(dim=-1)
        experts_states = torch.einsum("b t h k, b t h k d -> b t d", experts_weights, up_embed)
        hidden_states = self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = hidden_states + experts_states
        return hidden_states


class DogeDecoderLayer(nn.Module):
    def __init__(self, config: DogeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_dropout = config.hidden_dropout

        self.pre_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DogeDynamicMaskAttention(config=config, layer_idx=layer_idx)
        self.pre_residual = Residual(config.hidden_size)

        self.post_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = DogeMLP(config) if not config.is_moe else DogeCDMoE(config)
        self.post_residual = Residual(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # sequence transformation
        residual = hidden_states
        hidden_states = self.pre_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
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


DOGE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DogeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Doge Model outputting raw hidden-states without any specific head on top.",
    DOGE_START_DOCSTRING,
)
class DogePreTrainedModel(PreTrainedModel):
    config_class = DogeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DogeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


DOGE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Doge Model outputting raw hidden-states without any specific head on top.",
    DOGE_START_DOCSTRING,
)
class DogeModel(DogePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DogeDecoderLayer`]

    Args:
        config: DogeConfig
    """

    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.word_embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.rotary_emb = RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [DogeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embed

    def set_input_embeddings(self, value):
        self.word_embed = value

    @add_start_docstrings_to_model_forward(DOGE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.word_embed(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

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
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
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

        # in case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask=attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor = None,
        sequence_length: int = None,
        target_length: int = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
        cache_position: torch.Tensor = None,
        batch_size: int = None,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(LossKwargs): ...


class DogeForCausalLM(DogePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.config = config
        self.model = DogeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embed

    def set_input_embeddings(self, value):
        self.model.word_embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(DOGE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
         >>> from transformers import AutoTokenizer, AutoModelForCausalLM

        >>> model = AutoModelForCausalLM.from_pretrained("JingzeShi/Doge-20M")
        >>> tokenizer = AutoTokenizer.from_pretrained("JingzeShi/Doge-20M")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder output consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Doge Model transformer with a sequence classification head on top (linear layer).

    [`DogeForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    DOGE_START_DOCSTRING,
)
class DogeForSequenceClassification(LlamaForSequenceClassification):
    def __init__(self, config: DogeConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        self.model = DogeModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embed

    def set_input_embeddings(self, value):
        self.model.word_embed = value


__all__ = [
    "DogeConfig",
    "DogeForCausalLM",
    "DogeModel",
    "DogePreTrainedModel",
    "DogeForSequenceClassification",
]
