# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from typing import Callable, Optional, Tuple

import torch
from packaging import version
from torch import nn
from torch.nn import functional as F

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaPreTrainedModel,
    LlamaMLP,
    LlamaModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3MoE,
    apply_rotary_pos_emb_interleave,
    yarn_get_mscale,
)
from transformers.models.mixtral.modeling_mixtral import MixtralExperts
from transformers.models.phi.modeling_phi import PhiRotaryEmbedding

from ...cache_utils import Cache, DynamicCache
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.import_utils import get_torch_version
from ...utils.generic import check_model_inputs
from .configuration_openpangu_v2 import OpenPanguV2Config


logger = logging.get_logger(__name__)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    query_multi_head = query
    key_multi_head = key_states
    value_multi_head = value_states
    attn_output_list = []
    num_heads = query.shape[1]


    for i in range(num_heads):
        query = query_multi_head[:, i:i+1, :, :]
        key_states = key_multi_head[:, i:i+1, :, :]
        value_states = value_multi_head[:, i:i+1, :, :]
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        del attn_weights
        attn_output_list.append(attn_output)
        
    attn_output = torch.cat(attn_output_list, dim=1)
    
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


class DsaIndexer(nn.Module):
    """
    Dynamic Sparse Attention (DSA) indexer for selecting top-k tokens.

    The Indexer has its own lightweight projections (wq_b, wk) separate from the
    main MLA attention. It uses non-interleaved (NeoX/Llama) RoPE, unlike the main attention
    which uses interleaved RoPE.

    **Cache strategy**: The Indexer manages its own key cache (`_cached_keys`) separately
    from the DynamicCache used by MLA attention, since DynamicCache is sized for exactly
    `num_hidden_layers` attention layers. Keys are concatenated along the sequence dimension
    during autoregressive decode.
    """

    def __init__(self, config: OpenPanguV2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank

        # Named to match checkpoint: wq_b, wk, k_norm
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = OpenPanguV2RMSNorm(self.head_dim, eps=1e-6)
        # Named to match checkpoint: weights_proj
        # In the reference, this is fp32; the HF FP8 checkpoint stores a bf16 tensor.
        # Keeping it as a plain Linear prevents FP8 conversion (see `_keep_in_fp32_modules`).
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

        # Indexer maintains its own key cache (not in DynamicCache, which is sized for attention layers only)
        self._cached_keys: torch.Tensor | None = None

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, hidden]
        q_resid: torch.Tensor,  # [B, S, q_lora_rank]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        use_cache: bool = False,
    ) -> torch.LongTensor:
        """
        Computes top-k token indices for sparse attention (DSA).

        This is the bf16 equivalent of the reference Indexer which uses `rotate_activation` (Hadamard transform)
        and `fp8_index` (FP8 quantized scoring kernel). Since the Hadamard transform is orthogonal (dot products
        are preserved: Hq·Hk = q·k), and FP8 quantization is a precision optimization, we skip both and compute
        scores directly in bf16/fp32.

        The scoring logic computes:
            index_score[b,s,t] = Σ_h (weight[b,s,h] · softmax_scale · q[b,s,h,:] · k[b,t,:])

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            position_embeddings: `(cos, sin)` from RotaryEmbedding.
            attention_mask: Causal mask, broadcastable to `[B, S, T]`.
            use_cache: Whether to store/update the indexer's own key cache for autoregressive decode.

        Returns:
            `torch.LongTensor`: Top-k token indices of shape `[B, S, topk]`.
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # === Queries and Keys ===
        q = self.wq_b(q_resid)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
       
        k = self.k_norm(self.wk(hidden_states))  # [B, S, D]
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2)
        k_pe = k_pe.squeeze(2)
        
        q = torch.cat([q_pe, q_nope], dim=-1)  # [B, S, H, D]
        k = torch.cat([k_pe, k_nope], dim=-1)  # [B, S, D]

        # === Key cache (managed by the indexer, not DynamicCache) ===
        # Reset cache on prefill (new prompt) to avoid stale keys / batch-size mismatch
        if seq_len > 1:
            self._cached_keys = None

        if use_cache:
            if self._cached_keys is not None:
                k_cached = torch.cat([self._cached_keys, k], dim=1)  # [B, T, D]
            else:
                k_cached = k
            self._cached_keys = k_cached
        else:
            k_cached = k

        # === Scoring ===
        # Reference: weights = weights_proj(x.float()) * n_heads^(-0.5)
        # Reference: weights = weights.unsqueeze(-1) * q_scale * softmax_scale
        # Reference: index_score = fp8_index(q_fp8, weights, k_cache, k_scale_cache)
        #
        # In bf16 mode (no FP8), q_scale = 1. The fp8_index kernel computes:
        #   score[b,s,t] = sum_h(weights[b,s,h] * dot(q[b,s,h,:], k[b,t,:]))
        # where weights already absorbs n_heads^(-0.5) and softmax_scale.

        # Don't force fp32 inputs here: the checkpoint stores `weights_proj.weight` in bf16.
        # Use native dtype for matmul, then upcast the result for scoring stability.
        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)  # [B, S, H]

        # q·k^T per head: [B, S, H, D] @ [B, T, D]^T → [B, S, H, T]
        num_heads = q.shape[2]
        q_multi_heads = q # [B, S, H, D]
        weights_multi_heads = weights # [B, S, H]
        index_scores_list = []
        for i in range(num_heads):
            q = q_multi_heads[:,:,i:i+1,:] # [B, S, 1, D]
            weights = weights_multi_heads[:,:,i:i+1] # [B, S, 1]
            scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale # [B, S, 1, t]
        
            scores = torch.nn.functional.relu(scores) # [B, S, 1, t]
            # Weight per head and sum across heads → [B, S, T]
            index_scores = torch.einsum("bsht,bsh->bst", scores, weights) # [B, S, T]
            index_scores_list.append(index_scores)
            del scores
        index_scores = sum(index_scores_list)

        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        total_len = index_scores.shape[-1]
        topk = min(self.index_topk, total_len)
        topk_indices = index_scores.topk(topk, dim=-1).indices  # [B, S, topk]
       
        return topk_indices


class FastGELU(nn.Module):
    """
    Applies GELU approximation
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        abs_value = torch.abs(input)
        return input * torch.sigmoid(1.702 * abs_value) * torch.exp(0.851 * (input - abs_value))


class mHCModule(nn.Module):
    def __init__(
        self,
        config: OpenPanguV2Config,
        merge_layer_only_pre=False,
    ):
        super().__init__()
        self.num_stream = config.mhc_num_stream
        self.hidden_size = config.hidden_size
        self.merge_layer_only_pre = merge_layer_only_pre
        
        if not self.merge_layer_only_pre:
            phi_output_hidden_size = (self.num_stream + 2) * self.num_stream
            self.branch_alpha = nn.Parameter(torch.empty(3, dtype=torch.bfloat16))
            self.branch_beta = nn.Parameter(torch.empty(self.num_stream*(self.num_stream+2), dtype=torch.bfloat16))
            # self.branch_alpha_post = nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
            # self.branch_alpha_res = nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
            # self.branch_beta_post = nn.Parameter(torch.empty(self.num_stream, dtype=torch.bfloat16))
            # self.branch_beta_res = nn.Parameter(torch.empty(self.num_stream * self.num_stream, dtype=torch.bfloat16))
        else:
            phi_output_hidden_size = self.num_stream
            self.branch_alpha_pre = nn.Parameter(torch.empty(1, dtype=torch.bfloat16))
            self.branch_beta_pre = nn.Parameter(torch.empty(self.num_stream, dtype=torch.bfloat16))
        self.phi = nn.Linear(
            self.hidden_size * self.num_stream,
            phi_output_hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.mhc_use_gamma = config.mhc_use_gamma
        self.hc_eps = 1e-6
        self.norm_eps = config.rms_norm_eps
        self.mhc_recur_norm = config.mhc_recur_norm
        if self.mhc_use_gamma:
            self.norm_gamma = nn.Parameter(torch.empty(self.hidden_size * self.num_stream, dtype=torch.bfloat16))
            
    def hc_pre(self, x: torch.Tensor):
        """
        x: (B, S, n * H)
        """
        dtype = x.dtype
        # x = x.float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        if self.mhc_use_gamma:
            weight = self.phi((x * rsqrt * self.norm_gamma.unsqueeze(0)).to(torch.bfloat16))
        else:
            weight = self.phi(x) * rsqrt

        # (B,S,n), (B,S,n), (B,S,n,n)
        h_pre, h_post, h_res = self.hc_split_sinkhorn_torch(weight)

        # (B,S,H)
        y = torch.sum(h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(self.num_stream, -1)), dim=-2)
        return y.to(dtype), h_post, h_res
    
    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor, h_res: torch.Tensor):
        """
        x: (B, S, H)
        residual: (B, S, n * H)
        h_post: (B, S, n)
        h_res: (B, S, n, n)
        """
        if self.merge_layer_only_pre:
            return x

        # B, S, _ = x.shape
        # n = self.num_stream
        # res_shared = residual.view(B, S, n, -1)
        # y = torch.matmul(h_res.transpose(-1, -2), res_shared.to(h_res.dtype))
        # y += h_post.unsqueeze(-1) * x.unsqueeze(-2)

        y = h_post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            h_res.unsqueeze(-1) * residual.unflatten(dim=-1, sizes=(self.num_stream, -1)).unsqueeze(-2), dim=-3
        )
        return y.view(residual.shape).type_as(x)

    def hc_split_sinkhorn_torch(self, weight):
        if not self.merge_layer_only_pre:
            h_pre, h_post, h_res = weight.split(
                [self.num_stream, self.num_stream, self.num_stream * self.num_stream], dim=-1
            )
            alpha_pre, alpha_post, alpha_res = self.branch_alpha.view(-1).split([1,1,1])
            beta_pre, beta_post, beta_res = self.branch_beta.view(-1).split(
                [self.num_stream, self.num_stream, self.num_stream * self.num_stream]
            )
            
            h_post = 2 * torch.sigmoid(h_post * alpha_post + beta_post)
            h_res = h_res.unflatten(-1, (self.num_stream, self.num_stream))
            h_res = h_res * alpha_res + beta_res.view(self.num_stream, self.num_stream)
            h_res = self.sinkhorn_knopps(h_res, self.mhc_recur_norm, self.hc_eps)
        else:
            h_pre = weight
            h_post = None
            h_res = None
            alpha_pre = self.branch_alpha_pre
            beta_pre = self.branch_beta_pre
        h_pre = torch.sigmoid(h_pre * alpha_pre + beta_pre) + self.hc_eps
        return h_pre, h_post, h_res

    def sinkhorn_knopps(self, h_res, sinkhorn_iters, eps):
        h_res = h_res.softmax(-1) + eps
        col_sum = h_res.sum(-2, keepdim=True)
        h_res = h_res / (col_sum + eps)
        for _ in range(sinkhorn_iters - 1):
            row_sum = h_res.sum(-1, keepdim=True)
            h_res = h_res / (row_sum + eps)
            col_sum = h_res.sum(-2, keepdim=True)
            h_res = h_res / (col_sum + eps)
        return h_res


class WindowBuffer:
    def __init__(self, win_size, aggregate_fn):
        self.win_size = win_size
        self.aggregate_fn = aggregate_fn
        self.cache_len = win_size - 1
        self.cache = None

    def _update_cache(self, hidden_states):
        if self.cache_len <= 0:
            return None
        
        if self.cache is None:
            B, S, H = hidden_states.shape
            if S < self.cache_len:
                padding = torch.zeros((B, self.cache_len - S, H), 
                                      device=hidden_states.device, 
                                      dtype=hidden_states.dtype)
                self.cache = torch.cat([padding, hidden_states], dim=1)
            else:
                self.cache = hidden_states[:, -self.cache_len:, :]
        else:
            combined = torch.cat([self.cache, hidden_states], dim=1)
            self.cache = combined[:, -self.cache_len:, :]
        
        return self.cache

    def reset(self):
        self.cache = None

    def __call__(self, hidden_states, conv_mask=None, use_cache=True):
        """
        hidden_states: (B, S, H)
        """
        B, S, H = hidden_states.shape
        
        if not use_cache:
            padding = torch.zeros((B, self.cache_len, H), 
                                  device=hidden_states.device, 
                                  dtype=hidden_states.dtype)
            conv_input = torch.cat([padding, hidden_states], dim=1)
        else:
            if S > 1:
                self.reset()
            current_cache = self.cache if self.cache is not None else \
                            torch.zeros((B, self.cache_len, H), 
                                        device=hidden_states.device, 
                                        dtype=hidden_states.dtype)
            
            conv_input = torch.cat([current_cache, hidden_states], dim=1)
            self._update_cache(hidden_states)

        # (B, cache_len+S, H) -> (B, H, cache_len+S)
        x = conv_input.permute(0, 2, 1)
        output = self.aggregate_fn(x)
        if conv_mask is not None:
            output = output * conv_mask.unsqueeze(-1).permute(0, 2, 1)
        
        # (B, H, S) -> (B, S, H)
        return output.permute(0, 2, 1)


if version.parse(get_torch_version()) >= version.parse("2.3.0"):

    class OpenPanguV2RMSNorm(nn.RMSNorm):
        def __init__(self, hidden_size, eps: float = 1e-6) -> None:
            super().__init__(normalized_shape=hidden_size, eps=eps, elementwise_affine=True)

else:
    @use_kernel_forward_from_hub("RMSNorm")
    class OpenPanguV2RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps: float = 1e-6) -> None:
            """
            OpenPanguV2RMSNorm is equivalent to T5LayerNorm
            """
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)

        def extra_repr(self):
            return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class OpenPanguV2RotaryEmbedding(PhiRotaryEmbedding):
    pass


class OpenPanguV2MLP(LlamaMLP):
    def __init__(self, config, intermediate_size=None):
        super().__init__(config)
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class OpenPanguV2Attention(nn.Module):
    def __init__(self, config: OpenPanguV2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.is_causal = True
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
            self.q_a_layernorm = OpenPanguV2RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = OpenPanguV2RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_parameters.get("rope_type", "default") != "default":
            mscale_all_dim = self.config.rope_parameters.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_parameters["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        self.param_sink_number = config.param_sink_number
        if self.param_sink_number > 0:
            self.param_sink_k_pe = torch.nn.Parameter(
                torch.empty(
                    (self.param_sink_number, self.qk_rope_head_dim),
                    dtype=config.torch_dtype,
                )
            )
            self.param_sink_compressed_kv = torch.nn.Parameter(
                torch.empty(
                    (self.param_sink_number, self.kv_lora_rank),
                    dtype=config.torch_dtype,
                )
            )

        self.use_mome = config.router_sliding_window > 0
        if self.use_mome:
            self.qa_conv = torch.nn.Conv1d(
                self.q_lora_rank,
                self.q_lora_rank,
                config.router_sliding_window,
                groups=self.q_lora_rank,
                bias=False,
            )
            self.compresskv_conv = torch.nn.Conv1d(
                self.kv_lora_rank,
                self.kv_lora_rank,
                config.router_sliding_window,
                groups=self.kv_lora_rank,
                bias=False,
            )
            self.o_conv = torch.nn.Conv1d(
                self.num_heads * self.v_head_dim,
                self.num_heads * self.v_head_dim,
                config.router_sliding_window,
                groups=self.num_heads * self.v_head_dim,
                bias=False,
            )
            self.qa_buffer = WindowBuffer(config.router_sliding_window, self.qa_conv.forward)
            self.compresskv_buffer = WindowBuffer(config.router_sliding_window, self.compresskv_conv.forward)
            self.o_buffer = WindowBuffer(config.router_sliding_window, self.o_conv.forward)
            self.router_sliding_window = config.router_sliding_window
        
        self.use_dsa = config.index_topk is not None and config.index_topk > 0 and layer_idx in config.dsa_layers
        if self.use_dsa:
            self.indexer = DsaIndexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
            q_resid = None
        else:
            q_states = self.q_a_proj(hidden_states)
            if self.use_mome:
                if seq_length >= 2:
                    conv_mask = (attention_mask == 0).any(dim=1).any(dim=1)
                    zeros_prefix = torch.zeros((batch_size, self.router_sliding_window - 1), dtype=torch.bool, device=attention_mask.device)
                    conv_mask = torch.cat([zeros_prefix, conv_mask[:, :-self.router_sliding_window + 1]], dim=1)
                else:
                    conv_mask = None
                q_states = self.qa_buffer(q_states, conv_mask) + q_states
            q_resid = self.q_a_layernorm(q_states)
            q_states = self.q_b_proj(q_resid)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        if self.use_mome:
            k_pass = self.compresskv_buffer(k_pass, conv_mask) + k_pass

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        if self.config.rope_interleave:  # support using interleaved weights for efficiency
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.use_dsa:
            indexer_mask = (
                attention_mask[:, 0, :, :]
                if attention_mask is not None and attention_mask.dim() == 4
                else attention_mask.unsqueeze(1)
                if attention_mask is not None
                else None
            )
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                indexer_mask,
                use_cache=past_key_values is not None,
            )

            total_len = key_states.shape[2]
            index_mask = torch.full(
                (batch_size, seq_length, total_len),
                float("-inf"),
                device=hidden_states.device,
                dtype=query_states.dtype,
            )
            index_mask.scatter_(-1, topk_indices, 0.0)
            index_mask = index_mask.unsqueeze(1)
            if attention_mask is not None and attention_mask.dim() == 4:
                causal_mask = attention_mask[..., :total_len]
                combined_mask = index_mask + causal_mask
            else:
                combined_mask = (
                    attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
                    if attention_mask is not None
                    else index_mask
                )
            attention_mask = combined_mask
        
        if self.param_sink_number > 0:
            # [b, n, s, d]
            batch_size, kv_seq_len = key_states.shape[0], key_states.shape[2]
            param_sink_kv_c_normed = self.kv_a_layernorm(self.param_sink_compressed_kv)
            param_sink_kv = self.kv_b_proj(param_sink_kv_c_normed).view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            param_sink_k_nope, param_sink_value = param_sink_kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            param_sink_k_pe = self.param_sink_k_pe.unsqueeze(1).expand(-1, self.num_heads, -1)
            param_sink_key = torch.cat((param_sink_k_nope, param_sink_k_pe), dim=-1)

            param_sink_key = param_sink_key.permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1).to(key_states.device)
            param_sink_value = param_sink_value.permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1).to(value_states.device)
            key_states = torch.cat([param_sink_key, key_states], dim=2)
            value_states = torch.cat([param_sink_value, value_states], dim=2)
            kv_seq_len += self.param_sink_number

            attention_mask = torch.nn.functional.pad(attention_mask, (self.param_sink_number, 0), value=0.0)

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

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.use_mome:
            attn_output = self.o_buffer(attn_output, conv_mask) + attn_output
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights



class OpenPanguV2Experts(MixtralExperts):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.n_routed_experts
        self.intermediate_dim = config.moe_intermediate_size


class OpenPanguV2TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        
    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits


class OpenPanguV2SparseMoeBlock(DeepseekV3MoE):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = OpenPanguV2Experts(config)
        self.gate = OpenPanguV2TopkRouter(config)
        self.shared_experts = OpenPanguV2MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = 1
        self.topk_group = 1
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


class OpenPanguV2DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: OpenPanguV2Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = OpenPanguV2Attention(config=config, layer_idx=layer_idx)
        self.attention_type = config.layer_types[layer_idx]

        if config.first_k_dense_replace > 0 and layer_idx >= config.first_k_dense_replace:
            self.mlp = OpenPanguV2SparseMoeBlock(config)
        else:
            self.mlp = OpenPanguV2MLP(config)

        self.input_layernorm = OpenPanguV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = OpenPanguV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.sandwich_norm = getattr(config, "sandwich_norm", False)
        if self.sandwich_norm:
            self.pre_mlp_layernorm = OpenPanguV2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_mlp_layernorm = OpenPanguV2RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        # self.use_mome = (layer_idx == 0 or layer_idx == config.num_hidden_layers - 1) and config.router_sliding_window > 0
        # if self.use_mome:
        #     self.merge_conv = torch.nn.Conv1d(
        #         config.hidden_size,
        #         config.hidden_size,
        #         config.router_sliding_window,
        #         groups=config.hidden_size,
        #         bias=False,
        #     )
        #     self.window_buffer = WindowBuffer(
        #         config.router_sliding_window, self.merge_conv.forward
        #     )

        block_post_layernorm_hidden_size = config.hidden_size
        self.use_mhc = config.use_mhc
        if self.use_mhc:
            self.attn_mhc_module = mHCModule(config)
            self.mlp_mhc_module = mHCModule(config)
            block_post_layernorm_hidden_size *= config.mhc_num_stream
        
        self.has_block_post_layernorm = config.block_post_layernorm_idx is not None and layer_idx in config.block_post_layernorm_idx
        if self.has_block_post_layernorm:
            self.block_post_layernorm = OpenPanguV2RMSNorm(block_post_layernorm_hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        if self.use_mhc:
            hidden_states, h_post, h_res = self.attn_mhc_module.hc_pre(hidden_states)
        
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if self.sandwich_norm:
            hidden_states = self.post_attention_layernorm(hidden_states)

        if self.use_mhc:
            hidden_states = self.attn_mhc_module.hc_post(hidden_states, residual, h_post, h_res)
        else:
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        # if self.use_mome:
        #     hidden_states = self.window_buffer(hidden_states)

        if self.use_mhc:
            hidden_states, h_post, h_res = self.mlp_mhc_module.hc_pre(hidden_states)

        if self.sandwich_norm:
            hidden_states = self.pre_mlp_layernorm(hidden_states)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)

        if self.use_mhc:
            hidden_states = self.mlp_mhc_module.hc_post(hidden_states, residual, h_post, h_res)
        else:
            hidden_states = residual + hidden_states

        if self.has_block_post_layernorm:
            hidden_states = self.block_post_layernorm(hidden_states)
        return hidden_states


class OpenPanguV2PreTrainedModel(LlamaPreTrainedModel):
    pass


class OpenPanguV2Model(LlamaModel):
    def __init__(self, config: OpenPanguV2Config):
        super().__init__(config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.use_mhc = config.use_mhc
        if self.use_mhc:
            self.num_stream = config.mhc_num_stream
            self.merge_mhc_module = mHCModule(
                config=config,
                merge_layer_only_pre=True,
            )

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if self.use_mhc:
            hidden_states = hidden_states.repeat(1, 1, self.num_stream)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        if self.use_mhc:
            hidden_states, _, _ = self.merge_mhc_module.hc_pre(hidden_states)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class OpenPanguV2ForCausalLM(LlamaForCausalLM):
    pass


__all__ = [
    "OpenPanguV2PreTrainedModel",
    "OpenPanguV2Model",
    "OpenPanguV2ForCausalLM",
]
