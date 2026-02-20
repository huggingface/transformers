# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...modeling_rope_utils import RopeParameters, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import logging
from ...utils.generic import is_flash_attention_requested, maybe_autocast
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    eager_attention_forward,
)
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts


logger = logging.get_logger(__name__)


class DeepseekV2Config(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV2Model`]. It is used to instantiate a DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of DeepSeek-V2-Lite" [deepseek-ai/DeepSeek-V2-Lite"](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite").
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the DeepSeek model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`DeepseekV2Model`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            The number of key-value heads used to implement Grouped Query Attention (GQA). If
            `num_key_value_heads=num_attention_heads`, the model will use Multi-Head Attention (MHA). If
            `num_key_value_heads=1`, the model will use Multi-Query Attention (MQA). Otherwise, GQA is used.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon value used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/value attentions (useful for inference optimization).
        pad_token_id (`int`, *optional*):
            Padding token ID.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning-of-sequence token ID.
        eos_token_id (`int`, *optional*, defaults to 2):
            End-of-sequence token ID.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value, and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to attention weights.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias term in the MLP layers.
        first_k_dense_replace (`int`, *optional*, defaults to 0):
            Number of dense layers in the shallow layers before switching to MoE layers.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the LoRA decomposition for key-value projections.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the LoRA decomposition for query projections.
            Specifically, it determines the dimensionality to which the query (q) vectors are compressed before being expanded back to their original size.
            It reduces computational overhead while maintaining model performance.
        n_group (`int`, *optional*):
            Number of groups for routed experts.
        n_routed_experts (`int`, *optional*, defaults to 64):
            Number of routed experts (None indicates a dense model).
        n_shared_experts (`int`, *optional*, defaults to 2):
            Number of shared experts (None indicates a dense model).
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            The head dimension for the QK (query-key) projections when using NOPE (Neural Operator Position Encoding).
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            The head dimension for QK projections when using RoPE.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for routed experts in MoE models.
        topk_group (`int`, *optional*):
            Number of selected groups per token for expert selection.
        topk_method (`str`, *optional*, defaults to `"greedy"`):
            The method used for selecting top-k experts in the routed gate mechanism.
        norm_topk_prob (`bool`, *optional*, defaults to `False`):
            Whether to renormalize the router probabilities when `top_k > 1`. This flag is kept for backward
            compatibility with previously released checkpoints and runtimes relying on the legacy DeepSeek config.
        v_head_dim (`int`, *optional*, defaults to 128):
            The dimension of value projections in the attention layers.
        num_experts_per_tok (`int`, *optional*):
            The number of experts selected per token. If `None`, the model behaves as a dense Transformer.
        moe_intermediate_size (`int`, *optional*, defaults to 1407):
            Dimension of the MoE (Mixture of Experts) representations.

    ```python
    >>> from transformers import DeepseekV2Model, DeepseekV2Config
    >>> # Initializing a DeepSeek-V2 style configuration
    >>> configuration = DeepseekV2Config()
    >>> # Accessing the model configuration
    >>> model = DeepseekV2Model(configuration)
    >>> print(model.config)
    ```
    """

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.q_a_proj": "colwise",
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
    }

    model_type = "deepseek_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int | None = 32000,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 11008,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 2048,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        mlp_bias: bool | None = False,
        first_k_dense_replace: int | None = 0,
        kv_lora_rank: int | None = 512,
        q_lora_rank: int | None = 1536,
        n_group: int | None = None,
        n_routed_experts: int | None = 64,
        n_shared_experts: int | None = 2,
        qk_nope_head_dim: int | None = 128,
        qk_rope_head_dim: int | None = 64,
        routed_scaling_factor: float | None = 1.0,
        topk_group: int | None = None,
        topk_method: str | None = "greedy",
        norm_topk_prob: bool | None = False,
        v_head_dim: int | None = 128,
        num_experts_per_tok: int | None = None,
        moe_intermediate_size: int | None = 1407,
        **kwargs,
    ):
        self.first_k_dense_replace = first_k_dense_replace
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.n_group = n_group
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.v_head_dim = v_head_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        super().__init__(**kwargs)

        self.head_dim = qk_rope_head_dim
        del self.pretraining_tp


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast to [1, 1, seq_len, dim // 2]
    freqs_cis = freqs_cis.unsqueeze(1).to(xq_.device)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)
    return xq_out, xk_out


class DeepseekV2Experts(Qwen2MoeExperts):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.n_routed_experts


class DeepseekV2Moe(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.experts = DeepseekV2Experts(config)
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(config=config, intermediate_size=intermediate_size)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk_method = config.topk_method
        self.num_group = config.n_group
        self.top_k = config.num_experts_per_tok
        self.topk_group = config.topk_group

    def route_tokens_to_experts(self, router_logits):
        batch_size, seq_len, hidden_dim = router_logits.shape
        router_logits = router_logits.view(-1, hidden_dim)
        router_logits = router_logits.softmax(dim=-1, dtype=torch.float32)
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(router_logits, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            group_scores = router_logits.view(batch_size * seq_len, self.num_group, -1).max(dim=-1).values
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(batch_size * seq_len, self.num_group, self.num_experts // self.num_group)
                .reshape(batch_size * seq_len, -1)
            )
            tmp_scores = router_logits.masked_fill(~score_mask.bool(), 0.0)
            topk_weight, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)

        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = nn.functional.linear(hidden_states.type(torch.float32), self.gate.weight.type(torch.float32))
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class DeepseekV2MLP(LlamaMLP):
    def __init__(self, config: DeepseekV2Config, hidden_size=None, intermediate_size=None):
        super().__init__(config)
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size


class DeepseekV2RMSNorm(LlamaRMSNorm):
    pass


class DeepseekV2RotaryEmbedding(LlamaRotaryEmbedding):
    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Convert to complex representation
            freqs_cis = freqs_cis * self.attention_scaling

        return freqs_cis


class DeepseekV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.qk_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(query_shape).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_nope, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_nope = self.kv_b_proj(self.kv_a_layernorm(k_nope)).view(key_shape).transpose(1, 2)
        k_nope, value_states = torch.split(k_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, position_embeddings.to(q_pe.device))

        k_pe = k_pe.expand(*k_nope.shape[:-1], -1)
        query_states = torch.cat((q_nope, q_pe), dim=-1)
        key_states = torch.cat((k_nope, k_pe), dim=-1)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DeepseekV2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.self_attn = DeepseekV2Attention(config=config, layer_idx=layer_idx)
        self.mlp = DeepseekV2Moe(config) if layer_idx >= config.first_k_dense_replace else DeepseekV2MLP(config)

        self.input_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DeepseekV2PreTrainedModel(LlamaPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, DeepseekV2Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


class DeepseekV2Model(LlamaModel):
    pass


class DeepseekV2ForCausalLM(LlamaForCausalLM):
    pass


class DeepseekV2ForSequenceClassification(LlamaForSequenceClassification):
    pass


__all__ = [
    "DeepseekV2PreTrainedModel",
    "DeepseekV2Model",
    "DeepseekV2ForCausalLM",
    "DeepseekV2ForSequenceClassification",
    "DeepseekV2Config",
]
