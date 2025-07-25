# coding=utf-8
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

import warnings
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...utils import (
    logging,
)
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    eager_attention_forward,
)
from ..llama4.modeling_llama4 import Llama4TextRotaryEmbedding


logger = logging.get_logger(__name__)


class DeepseekV2Config(LlamaConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV2Model`]. It is used to instantiate a DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of DeepSeek-V2-Lite" [deepseek-ai/DeepSeek-V2-Lite"](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite").
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

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
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the Rotary Position Embeddings (RoPE).
        rope_scaling (`Dict`, *optional*):
            Configuration for scaling RoPE embeddings. Supports `linear` and `dynamic` scaling strategies.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value, and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability applied to attention weights.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias term in the MLP layers.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Weight coefficient for auxiliary loss in Mixture of Experts (MoE) models.
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
        seq_aux (`bool`, *optional*, defaults to `True`):
            Whether to compute the auxiliary loss for each individual sequence.
        topk_group (`int`, *optional*):
            Number of selected groups per token for expert selection.
        topk_method (`str`, *optional*, defaults to `"greedy"`):
            The method used for selecting top-k experts in the routed gate mechanism.
        v_head_dim (`int`, *optional*, defaults to 128):
            The dimension of value projections in the attention layers.
        num_experts_per_tok (`int`, *optional*):
            The number of experts selected per token. If `None`, the model behaves as a dense Transformer.
        norm_topk_prob (`bool`, *optional*, defaults to `False`):
            Whether to normalize the probability distribution over top-k selected experts.
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
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    model_type = "deepseek_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        aux_loss_alpha=0.001,
        first_k_dense_replace=0,
        kv_lora_rank=512,
        q_lora_rank=1536,
        n_group=None,
        n_routed_experts=64,
        n_shared_experts=2,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        routed_scaling_factor=1.0,
        seq_aux=True,
        topk_group=None,
        topk_method="greedy",
        v_head_dim=128,
        num_experts_per_tok=None,
        norm_topk_prob=False,
        moe_intermediate_size=1407,
        **kwargs,
    ):
        super().__init__(**kwargs)

        del self.pretraining_tp
        self.aux_loss_alpha = aux_loss_alpha
        self.first_k_dense_replace = first_k_dense_replace
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.n_group = n_group
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.routed_scaling_factor = routed_scaling_factor
        self.seq_aux = seq_aux
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.v_head_dim = v_head_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.moe_intermediate_size = moe_intermediate_size
        self.head_dim = qk_rope_head_dim


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


class DeepseekV2MoEGate(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.num_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, hidden_dim)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        # select top-k experts
        # greedy method is used for DeepSeek-V2-Lite
        # group_limited_greedy for DeepSeek-V2 and DeepSeek-V2-Chat
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            group_scores = scores.view(batch_size * seq_len, self.num_group, -1).max(dim=-1).values  # [n, num_group]
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, num_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, num_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(batch_size * seq_len, self.num_group, self.num_experts // self.num_group)
                .reshape(batch_size * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)

        topk_weight = topk_weight * self.routed_scaling_factor
        ### expert-level computation auxiliary loss
        return topk_idx, topk_weight


class DeepseekV2MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = nn.ModuleList(
            [
                (DeepseekV2MLP(config, intermediate_size=config.moe_intermediate_size))
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = DeepseekV2MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(config=config, intermediate_size=intermediate_size)
        self.ep_rank = 0
        self.experts_per_rank = config.n_routed_experts

    def moe(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        indicies = topk_ids.view(-1).argsort()
        sorted_tokens = hidden_states[indicies // topk_ids.shape[1]]

        # Process experts
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            if num_tokens == 0:
                continue
            end_idx = start_idx + num_tokens
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)

        # Reorder and combine outputs
        new_x = torch.empty_like(outs)
        new_x[indicies] = outs
        hidden_states = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class DeepseekV2MLP(LlamaMLP):
    def __init__(self, config: DeepseekV2Config, hidden_size=None, intermediate_size=None):
        super().__init__(config)
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size


class DeepseekV2RMSNorm(LlamaRMSNorm):
    pass


class DeepseekV2RotaryEmbedding(Llama4TextRotaryEmbedding):
    def __init__(self, config: DeepseekV2Config, device=None):
        super().__init__(config=config, device=device)
        # BC: "rope_type" was originally "type"
        self.rope_type = (
            config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            if config.rope_scaling is not None
            else "default"
        )


class DeepseekV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
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
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
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

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

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
            **kwargs,
        )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DeepseekV2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.self_attn = DeepseekV2Attention(config=config, layer_idx=layer_idx)
        self.mlp = DeepseekV2MoE(config) if layer_idx >= config.first_k_dense_replace else DeepseekV2MLP(config)

        self.input_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DeepseekV2PreTrainedModel(LlamaPreTrainedModel):
    def _init_weights(self, module):
        LlamaPreTrainedModel._init_weights(module)
        if isinstance(module, DeepseekV2MoEGate):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)


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
