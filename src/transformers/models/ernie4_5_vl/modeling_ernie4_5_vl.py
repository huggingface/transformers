# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
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

"""Ernie VL model"""
import itertools
import math
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import logging

from .configuration_ernie4_5_vl import (
    DFNRopeVisionTransformerConfig,
    Ernie4_5_MoEConfig,
    Ernie4_5_VLMoEConfig,
)


logger = logging.get_logger(__name__)


class TokenType:
    """token type definition"""

    text = 0
    image = 1
    video = 2


class UniqueNameGuard:
    """name guard"""

    def __init__(self, prefix=""):
        self.prefix = prefix
        self.counter = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_unique_name(self, name):
        """get unique name"""
        if name not in self.counter:
            self.counter[name] = 0
        else:
            self.counter[name] += 1
        return f"{self.prefix}{name}_{self.counter[name]}"


class RopeEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation for transformer models.

    RoPE encodes absolute positional information with rotation matrices and
    naturally incorporates relative position information in self-attention.

    Args:
        head_dim (int): Dimension size of each attention head
        compression_ratio (float, optional): Sequence length compression ratio. Defaults to 1.0.
        base (int, optional): Base value for frequency calculation. Defaults to 10000.

    Attributes:
        head_dim (int): Dimension size of each attention head
        compression_ratio (float): Sequence length compression factor
        base (int): Base value for frequency calculation
    """

    def __init__(self, head_dim, compression_ratio=1.0, base=10000, freq_allocation=0):
        """
        Initialize RoPE embedding layer.

        Args:
            head_dim: Dimension of each attention head
            compression_ratio: Scaling factor for position indices
            base: Base value for frequency calculation
        """
        super().__init__()
        self.head_dim = head_dim
        self.inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim))

        # num of freq allocated to time
        self.freq_allocation = freq_allocation

    #"""
    def forward(self, seq_length, position_ids=None):
        indices = self.inv_freq
        if position_ids is None:
            position_ids = torch.arange(
                0, seq_length, 1, dtype=torch.float32
            ).unsqueeze(1)
            sinusoid_inp = position_ids * indices.unsqueeze(0)
        else:
            seq_length = position_ids.shape[-1]
            sinusoid_inp = position_ids.unsqueeze(-1).to(
                torch.float32
            ) * indices.unsqueeze(0)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        pos_emb = pos_emb.view(-1, 1, seq_length, self.head_dim)
        pos_emb = pos_emb.detach()
        return pos_emb#"""

    def apply_rotary_3d(self, rp, q, k, position_ids):
        """
        rope 3d rotary

        args:
            rp: [1, max_seqlen, 1, head_dim]
            q: [bsz, seqlen, head, head_dim]
            k: [bsz, seqlen, head, head_dim]
            position_ids: [bsz, seqlen, 3]
        """
        #current_device = q.device
        #sin, cos = torch.chunk(rp, 2, axis=-1)
        assert position_ids.shape[:1] == q.shape[:1]
        #batch_indices = torch.arange(end=position_ids.shape[0])
        #batch_indices = batch_indices[..., None]
        #sin = sin.tile(position_ids.shape[0], 1, 1, 1).to(device=position_ids.device)
        #cos = cos.tile(position_ids.shape[0], 1, 1, 1).to(device=position_ids.device)

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[0], -1, 1).to(position_ids.device)
        position_ids_expanded = position_ids.permute(2, 0, 1)[:, :, None, :].float()  # shape (3, bs, 1, positions)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        cos_2 = freqs.cos()
        sin_2 = freqs.sin()

        sin_2_t = sin_2.split([44, 20], dim=-1)[-1][0].unsqueeze(-2)
        sin_2_h = sin_2.split([44, 20], dim=-1)[0][1][..., 0::2].unsqueeze(-2)
        sin_2_w = sin_2.split([44, 20], dim=-1)[0][2][..., 1::2].unsqueeze(-2)

        cos_2_t = cos_2.split([44, 20], dim=-1)[-1][0].unsqueeze(-2)
        cos_2_h = cos_2.split([44, 20], dim=-1)[0][1][..., 0::2].unsqueeze(-2)
        cos_2_w = cos_2.split([44, 20], dim=-1)[0][2][..., 1::2].unsqueeze(-2)

        assert self.freq_allocation != 0
        """sin_t = sin[batch_indices, position_ids[..., 0], :, -self.freq_allocation :]
        sin_h = sin[
            batch_indices,
            position_ids[..., 1],
            :,
            : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        sin_w = sin[
            batch_indices,
            position_ids[..., 2],
            :,
            1 : self.head_dim // 2 - self.freq_allocation : 2,
        ]

        cos_t = cos[batch_indices, position_ids[..., 0], :, -self.freq_allocation :]
        cos_h = cos[
            batch_indices,
            position_ids[..., 1],
            :,
            : self.head_dim // 2 - self.freq_allocation : 2,
        ]
        cos_w = cos[
            batch_indices,
            position_ids[..., 2],
            :,
            1 : self.head_dim // 2 - self.freq_allocation : 2,
        ]

        print(torch.allclose(sin_2_t, sin_t))
        print(torch.allclose(sin_2_h, sin_h))
        print(torch.allclose(sin_2_w, sin_w))
        print(torch.allclose(cos_2_t, cos_t))
        print(torch.allclose(cos_2_h, cos_h))
        print(torch.allclose(cos_2_w, cos_w))
        print()"""

        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_hw = torch.stack([sin_2_h, sin_2_w], dim=-1).reshape(
            sin_2_h.shape[:-1] + (sin_2_h.shape[-1] * 2,)
        )
        sin_thw = torch.cat([sin_hw, sin_2_t], dim=-1)
        sin_pos = sin_thw.repeat_interleave(2, dim=-1)

        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_hw = torch.stack([cos_2_h, cos_2_w], dim=-1).reshape(
            cos_2_h.shape[:-1] + (cos_2_h.shape[-1] * 2,)
        )
        cos_thw = torch.cat([cos_hw, cos_2_t], dim=-1)
        cos_pos = cos_thw.repeat_interleave(2, dim=-1)

        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_q = torch.stack(
            [-q[:, :, :, 1::2], q[:, :, :, 0::2]], dim=-1
        ).reshape(q.shape)
        query = (q.to(torch.float32) * cos_pos) + (
            rotate_half_q.to(torch.float32) * sin_pos
        )
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_k = torch.stack(
            [-k[:, :, :, 1::2], k[:, :, :, 0::2]], dim=-1
        ).reshape(k.shape)
        key = (k.to(torch.float32) * cos_pos) + (
            rotate_half_k.to(torch.float32) * sin_pos
        )
        return query, key


# copy Llama
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# copy Llama
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs#: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Ernie4_5_Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.use_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.use_bias
        )

        # TODO: rope to be moved outside
        self.freq_allocation = getattr(config, "freq_allocation", 0)
        self.rotary_emb = RopeEmbedding(
            self.head_dim,
            compression_ratio=config.compression_ratio,
            base=config.rope_theta,
            freq_allocation=self.freq_allocation,
        )

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert position_ids is not None, "rope3d requires pos-id"

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        # rope
        query_states_dtype = query_states.dtype
        kv_seq_len = position_ids.max() + 1
        offset = 0
        if past_key_value is not None:
            offset = position_ids.max()
            kv_seq_len = position_ids.max() + 1
            position_ids = position_ids[:, -1:, :]

        cos_sin = self.rotary_emb(kv_seq_len).permute([0, 2, 1, 3])
        if offset > 0 and position_ids is None:
            cos_sin = cos_sin[:, offset:]
        query_states, key_states = self.rotary_emb.apply_rotary_3d(
            cos_sin, query_states, key_states, position_ids
        )
        query_states = query_states.to(query_states_dtype)
        key_states = key_states.to(query_states_dtype)

        # cache
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        past_key_value = [key_states, value_states] if use_cache else None

        # core attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        #attention_interface: Callable = eager_attention_forward
        #if self.config._attn_implementation != "eager":
        #    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]  # forcing sdpa for now

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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


# Copy LlamaRMSNorm
class Ernie4_5_MoERMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Ernie4_5_MoERMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copy Ernie4_5_MoE
class Ernie4_5_MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def masked_fill(x, mask, value):
    """
    Fills elements of the input tensor with a given value where mask is True.
    """
    return torch.where(mask, torch.full_like(x, value), x)


def _squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    """Computes 0.5 * sum(x^2)"""
    return 0.5 * torch.sum(x * x)


@torch.no_grad()
def compute_optimal_transport(M, r, c, lam=1.0, epsilon=1e-8, max_iters: int = 10):
    """
    Computes optimal transport matrix and Sinkhorn distance using Sinkhorn-Knopp algorithm.
    """
    n, _ = M.shape
    P = F.softmax(-M / lam, dim=1)  # Applying softmax over columns
    u = torch.zeros(n, dtype=torch.float32, device=M.device)

    for _ in range(max_iters):
        P_sum_1 = P.sum(1)
        if (u - P_sum_1).abs().max() < epsilon:
            break
        u = P_sum_1
        P *= (r / (u + 1e-8)).unsqueeze(1)
        P *= (c / (P.sum(0) + 1e-8)).unsqueeze(0)

    P = torch.where(~P.isnan(), P, torch.zeros_like(P))
    return P, _


# TODO: seems like only topk is used
class Top2Gate(nn.Module):
    """
    Gate module implementing Top2Gating as described in Gshard paper.
    """

    def __init__(self, config, layer_idx: int, group=None, gate_weight=None) -> None:
        """
        Initialize the MoE (Mixture of Experts) layer.

        Args:
            config: Model configuration containing MoE parameters
            layer_idx: Index of this layer in the model
            group: Distributed communication group
            gate_weight: Optional pre-existing gate weight tensor
        """
        super().__init__()
        self.config = config

        self.model_dim = config.hidden_size
        self.num_experts = config.moe_num_experts
        self.num_experts_tensor = (
            sum(config.moe_num_experts)
            if config.multimodel_experts
            else config.moe_num_experts
        )

        self.cap = config.moe_capacity
        self.group = group

        self.layer_idx = layer_idx

        self.sinkhorn_2gate = config.sinkhorn_2gate
        self.sinkhorn_temp = config.sinkhorn_temp
        self.use_correction_bias = config.moe_use_aux_free  # true
        self.use_token_type_bias = config.get("moe_use_token_type_bias", False)

        self.act = partial(F.softmax, dim=-1)  # [S,E]

        self.no_jitter = True
        self.expert_drop = False
        self.eye_matrix = None
        self.eye_matrix_size = None
        self.norm_gate_logits = config.moe_norm_gate_logits  # true
        self.one = torch.ones([], dtype=torch.float32)

        self.moe_aux_loss_lambda = torch.tensor(config.moe_aux_loss_lambda).to(
            dtype=torch.float32
        )
        self.moe_z_loss_lambda = torch.tensor(config.moe_z_loss_lambda).to(
            dtype=torch.float32
        )
        self.moe_orthogonal_loss_lambda = torch.tensor(
            config.moe_orthogonal_loss_lambda
        ).to(dtype=torch.float32)

        if self.moe_aux_loss_lambda.ndim == 0:
            self.moe_aux_loss_lambda = self.moe_aux_loss_lambda.unsqueeze(0)
        if self.moe_z_loss_lambda.ndim == 0:
            self.moe_z_loss_lambda = self.moe_z_loss_lambda.unsqueeze(0)
        if self.moe_orthogonal_loss_lambda.ndim == 0:
            self.moe_orthogonal_loss_lambda = self.moe_orthogonal_loss_lambda.unsqueeze(
                0
            )

        self.experts_type_ids = None

        self.eps = torch.tensor([1e-12]).to(dtype=torch.float32)
        if config.multimodel_experts:
            if config.get("moe_use_hard_gate", False):
                self.num_experts_list = []
                self.experts_type_mask = []
                # hard-gate + group_experts 需要对gate_logits不同部分分开计算
                experts_ids = torch.zeros(
                    [sum(self.num_experts)], dtype=torch.int64
                ).reshape((1, -1))
                offset = 0
                for i, expert_num in enumerate(self.num_experts):
                    experts_ids[:, offset : offset + expert_num] = i
                    offset += expert_num
                self.experts_type_ids = experts_ids.reshape([-1])
                logger.info(
                    f"use moe_use_hard_gate, experts_ids: {self.experts_type_ids}"
                )
                for i, expert_num in enumerate(self.num_experts):
                    self.experts_type_mask.append(
                        self.experts_type_ids == i,
                    )
                    self.num_experts_list.append(expert_num)
            else:
                # 非group_experts, 依赖token_type_bias实现hard-gate能力。
                assert (
                    not config.moe_group_experts
                ), "group_experts must use hard_gate when multimodel_experts is True"
        else:
            self.num_experts_list = [self.num_experts]

        if gate_weight is not None:
            self.weight = gate_weight

            assert (
                not self.config.moe_use_token_type_bias
            ), "gate_weights is from outside, token_type_bias can't be used"
            logger.info("moe use gate_weight from outside")
            # use fp32 pecison in amp
            self._cast_to_low_precision = False
            self._cast_to_low_precison = False
        else:
            self._create_gate_parameter()
        logger.info(
            f"{config.moe_gate}: w/ capacity: {self.cap} experts:{self.num_experts} "
            f"use_token_type_bias:{self.use_token_type_bias} "
            f"gate_act:{config.moe_gate_act} "
            f"norm_gate_logits={self.norm_gate_logits} use_correction_bias={self.use_correction_bias}"
        )

    def _create_gate_parameter(self):
        """
        Create gate weight parameter.
        """
        if self.config.multimodel_experts:
            # support setting lambda for each expert group
            self.moe_z_loss_lambda = self.moe_z_loss_lambda.expand(
                len(self.num_experts)
            )
            self.moe_aux_loss_lambda = self.moe_aux_loss_lambda.expand(
                len(self.num_experts)
            )
            self.moe_orthogonal_loss_lambda = self.moe_orthogonal_loss_lambda.expand(
                len(self.num_experts)
            )

            for i, num_experts in enumerate(self.num_experts):
                if i == 1:
                    with UniqueNameGuard(f"mm_gate_{self.layer_idx}_"):
                        p = nn.Parameter(
                            torch.empty(
                                self.model_dim,
                                num_experts,
                                dtype=torch.float32,
                                device="cpu",
                            )
                        )
                        nn.init.xavier_uniform_(p)  # Common initialization
                else:
                    p = nn.Parameter(
                        torch.empty(
                            self.model_dim,
                            num_experts,
                            dtype=torch.float32,
                            device="cpu",
                        )
                    )
                    nn.init.xavier_uniform_(p)  # Common initialization
                self.register_parameter(
                    "weight" if i == 0 else f"weight_{i}",
                    p,
                )
        else:
            self.weight = nn.Parameter(
                torch.empty(self.model_dim, self.num_experts, dtype=torch.float32)
            )
            nn.init.xavier_uniform_(self.weight)  # Common initialization
        # use fp32 pecison in amp
        self._cast_to_low_precision = False
        self._cast_to_low_precison = False

    def get_gate_weight(self, transform_weight, is_multimodel=True):
        """
        在`multimodel_experts` 的情况下，将多个 weights merge 成一个整体
        transform_weight: bool, 按照 local-expert id 将 多模态 weight 交叠
        """
        if not is_multimodel or not self.config.multimodel_experts:
            return self.weight
        else:
            return torch.cat(
                [
                    getattr(self, "weight" if i == 0 else f"weight_{i}")
                    for i in range(len(self.num_experts))
                ],
                -1,
            )

    def forward(
        self,
        input: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        transform_weight: bool = True,
        correction_bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the gate.

        Args:
            input: Input tensor of shape [Seq, Dim]
            token_type_ids: Token type IDs tensor of shape [Seq]
            transform_weight: Whether to transform weights for multimodal experts
            correction_bias: Bias tensor for correction

        Returns:
            tuple: (capacity, dispatch_mask, combine_weights, scatter_index, router_loss, logits)
        """
        orig_dtype = input.dtype
        current_device = input.device
        weight = self.get_gate_weight(transform_weight)

        logits = F.linear(
            input.to(dtype=torch.float32, device=current_device),
            weight.T.to(dtype=torch.float32, device=current_device),
        )

        (
            capacity,
            dispatch_mask,
            combine_weights,
            scatter_index,
            l_aux,
            l_zloss,
        ) = self.top2_gating(
            logits,
            correction_bias=(
                correction_bias.to(device=current_device)
                if correction_bias is not None
                else None
            ),
        )

        combine_weights = combine_weights.to(orig_dtype)
        return capacity, dispatch_mask, combine_weights, scatter_index, None, logits

    def get_capacity(self, num_tokens, cap_factor=None, is_multimodel=True):
        """
        Calculate capacity based on number of tokens.

        Args:
            num_tokens: Number of input tokens
            cap_factor: Optional capacity factor override

        Returns:
            int: Calculated capacity
        """
        if is_multimodel and self.config.multimodel_experts:
            num_experts = sum(self.num_experts_list)
        elif isinstance(self.num_experts, (list, tuple)):
            num_experts = self.num_experts[0]
        else:
            num_experts = self.num_experts
        if cap_factor is not None:
            cap = cap_factor
        else:
            if self.training:
                cap = self.cap[0]
            elif num_tokens < num_experts:  # seqlen < num_expert
                cap = self.cap[2]
            else:
                cap = self.cap[1]
        # capacity = 2S/E
        capacity = int(cap * num_tokens // num_experts)
        assert (
            capacity > 0
        ), f"requires capacity to >= 0. cap={cap}, num_tokens={num_tokens}"
        return capacity

    def top2_gating(self, logits, cap=None, correction_bias=None):
        """
        Implement Top2 gating mechanism.

        Args:
            logits: Input logits tensor
            cap: Optional capacity override
            correction_bias: Bias tensor for correction

        Returns:
            tuple: (capacity, dispatch_masks, combine_weights, scatter_indexes, loss_aux, loss_z)

        Note:
        capacity: The maximum number that each token can be dispatched.
        dispatch_masks: Masks used for dispatching. The first element is the mask for the first
        type of tokens; the second element is the mask for the second type of tokens.
        combine_weights: Weights used for combining. The first element is the weight for the first
        type of tokens; the second element is the weight for the second type of tokens.
        scatter_indexes: Indexes used for scattering. The first element is the index for the first
        type of tokens; the second element is the index for the second type of tokens.
        loss_aux: Auxiliary loss.
        loss_z: Z loss.
        """
        gates = self.act(logits)

        # gates has shape of SE
        assert logits.ndim == 2, logits.shape
        num_tokens = gates.shape[0]
        num_experts = gates.shape[1]
        # capacity = 2S/E
        capacity = self.get_capacity(logits.shape[0], cap)
        current_device = logits.device

        # Create a mask for 1st's expert per token
        score_for_argmax = (
            gates + correction_bias.unsqueeze(0)
            if correction_bias is not None
            else gates
        )
        indices1_s = torch.argmax(score_for_argmax, dim=1)
        mask1 = F.one_hot(indices1_s, num_classes=num_experts).to(
            dtype=torch.int64, device=current_device
        )  # [0,1]

        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        if self.training and not self.no_jitter:
            gumbels = (
                -torch.empty_like(
                    logits,
                    device=current_device,
                )
                .exponential_()
                .log()
            )  # ~Gumbel(0,1)
            logits_w_noise = logits + gumbels
        else:
            logits_w_noise = logits

        logits_except1 = masked_fill(
            logits_w_noise,
            mask1.to(dtype=torch.bool, device=current_device),
            float("-inf"),
        )
        score_for_argmax = (
            self.act(logits_except1) + correction_bias.unsqueeze(0)
            if correction_bias is not None
            else logits_except1
        )
        indices2_s_original = torch.argmax(score_for_argmax, dim=1)

        if self.training and self.sinkhorn_2gate:
            r = (
                torch.ones(num_tokens, dtype=torch.float32, device=current_device)
                / num_tokens
            )
            c_mask_sum = mask1.to(dtype=torch.float32, device=current_device).sum(0)
            c = capacity - c_mask_sum
            c = torch.maximum(c, torch.zeros_like(c, device=current_device))
            c_sum = c.sum()
            if c_sum > 0:
                c = c / c_sum
            else:  # Avoid division by zero if all experts are full from top-1
                c = torch.ones_like(c, device=current_device) / num_experts

            pi, _ = compute_optimal_transport(
                -logits_except1.to(dtype=torch.float32, device=current_device).detach(),
                r,
                c,
                lam=self.sinkhorn_temp,
            )
            pi = masked_fill(
                pi, mask1.to(dtype=torch.bool, device=current_device), float("-inf")
            )
            indices2_s = torch.argmax(pi, dim=1)
        else:
            indices2_s = indices2_s_original

        mask2 = F.one_hot(indices2_s, num_classes=self.num_experts).to(
            dtype=torch.int64, device=current_device
        )

        # Compute locations in capacity buffer
        locations1 = (
            torch.cumsum(mask1, dim=0) - 1
        )  # [0,1,1,0,1,0,0] -> [0,0,0,0,1,1,1,]
        locations2 = torch.cumsum(mask2, dim=0) - 1
        # Update 2nd's location by accounting for locations of 1st
        locations2 += torch.sum(mask1, dim=0, keepdim=True)

        # Remove locations outside capacity from mask
        mask1 = mask1 * (locations1 < capacity).to(
            dtype=torch.int64, device=current_device
        )  # [0,1,1,0,0,0,0]
        mask2 = mask2 * (locations2 < capacity).to(
            dtype=torch.int64, device=current_device
        )

        # Store the capacity location for each token
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        locations2_s = torch.sum(locations2 * mask2, dim=1)

        # Normalize gate probabilities
        mask1_float = mask1.to(dtype=torch.float32, device=current_device)
        mask2_float = mask2.to(dtype=torch.float32, device=current_device)
        gates1_s = (gates * mask1_float).sum(dim=-1)
        gates2_s = (gates * mask2_float).sum(dim=-1)
        # logger.info(f'gates1_s:{gates1_s} gates2_s:{gates2_s} logits:{logits}')

        if self.norm_gate_logits:
            denom_s = gates1_s + gates2_s  # [0.2, 0.3]
            # Avoid divide-by-zero
            denom_s = torch.clamp(denom_s, min=1e-6)
            gates1_s /= denom_s
            gates2_s /= denom_s
        if self.training and self.expert_drop:
            # log.debug(gates2_s)
            gates2_s = torch.where(
                2 * gates2_s < torch.rand_like(gates2_s, device=current_device),
                torch.zeros_like(gates2_s, device=current_device),
                gates2_s,
            )

        # Calculate combine_weights and dispatch_mask
        gates1 = gates1_s.unsqueeze(1) * mask1_float
        gates2 = gates2_s.unsqueeze(1) * mask2_float

        combine1_weight, expert1_index = torch.max(gates1, dim=-1, keepdim=True)
        scatter1_index = expert1_index.squeeze(-1) * capacity + locations1_s
        scatter1_index = scatter1_index.to(dtype=torch.int64, device=current_device)
        dispatch1_mask = combine1_weight.to(
            dtype=torch.bool, device=current_device
        ).detach()

        combine2_weight, expert2_index = torch.max(gates2, dim=-1, keepdim=True)
        scatter2_index = expert2_index.squeeze(-1) * capacity + locations2_s
        scatter2_index = scatter2_index.to(dtype=torch.int64, device=current_device)
        dispatch2_mask = combine2_weight.to(
            dtype=torch.bool, device=current_device
        ).detach()
        # logger.info(f'expert-id: {expert1_index} vs {expert2_index}, mask:{mask1_float} vs {mask2_float}')

        return (
            capacity,
            torch.cat((dispatch1_mask, dispatch2_mask), 1),
            torch.cat((combine1_weight, combine2_weight), 1),
            torch.stack((scatter1_index, scatter2_index), 1),
            None,
            None,
        )

    def _cal_orthogonal_loss_opt_each_weight(self, weight, use_group):
        """
        Calculate optimized orthogonal loss for each weight.

        Args:
            weight: Weight tensor
            use_group: Whether to use expert groups

        Returns:
            Tensor: Calculated orthogonal loss
        """
        if weight.dtype != torch.float32:
            weight = weight.to(torch.float32)

        wnorm = torch.norm(weight, p=2, dim=1)
        weight = weight / torch.maximum(wnorm, self.eps.to(weight.device)).unsqueeze(1)

        if use_group:
            weight = weight.reshape(
                [self.config.moe_k, -1, weight.shape[1]]
            )  # [K, E/K, H]
            eye_matrix = torch.eye(
                weight.shape[1], dtype=weight.dtype, device=weight.device
            ).unsqueeze(0)
        else:
            eye_matrix = torch.eye(
                weight.shape[0], dtype=weight.dtype, device=weight.device
            )

        weight_matmul = torch.matmul(weight, weight.T)

        orthogonal_loss = weight_matmul - eye_matrix
        orthogonal_loss = _squared_l2_norm(orthogonal_loss) / (
            orthogonal_loss.size(0) * orthogonal_loss.size(1)
        )
        return orthogonal_loss


class TopKGate(Top2Gate):
    """
    Fused version of TopK gate for improved performance.
    """

    def forward(
        self,
        input: torch.Tensor,
        token_type_ids=None,
        transform_weight=True,
        is_multimodel=True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for fused gate.

        Args:
            input: Input tensor
            token_type_ids: Token type IDs
            transform_weight: Whether to transform weights

        Returns:
            tuple: (logits, capacity, router_loss)
        """
        current_device = input.device
        weight = self.get_gate_weight(transform_weight, is_multimodel=is_multimodel)

        logits = F.linear(
            input.to(dtype=torch.float32, device=current_device),
            weight.T.to(dtype=torch.float32, device=current_device),
        )
        if self.use_token_type_bias:
            assert token_type_ids is not None
            assert (
                token_type_ids.max() < self.bias.shape[0]
            ), f"token_type_ids {token_type_ids.max()} >= bias shape {self.bias.shape[0]}"
            bias = self.bias[token_type_ids]  # [seq]
            logits = logits + bias

        return logits


gate_class = dict(
    top2=Top2Gate,
    topk=TopKGate,
)


def get_gate(
    config: Ernie4_5_MoEConfig,
    expert: nn.Module,
    layer_idx: int,
) -> Tuple[nn.Module, nn.ModuleList]:
    """Initialize and distribute MoE (Mixture of Experts) components.

    Creates gate layer and distributed expert network for MoE architecture.

    Args:
        config (Ernie4_5_MoEConfig): Configuration for MoE architecture
        expert (nn.Module): Prototype expert network to be replicated
        layer_idx (int): Index of current layer in transformer stack

    Returns:
        Tuple[nn.Module, nn.ModuleList]:
            - gate: Initialized gate layer for routing
            - experts: ModuleList containing expert networks
    """
    moe_num_experts = (
        sum(config.moe_num_experts)
        if config.multimodel_experts
        else config.moe_num_experts
    )
    experts = nn.ModuleList([])

    for expert_id, (experts_num, fc) in enumerate(expert):
        experts_to_append = []
        if not hasattr(fc, "__len__"):  # run this
            experts_to_append.append(fc)
            if expert_id == 1:
                with UniqueNameGuard("_mm_deepcopy"):
                    for _ in range(experts_num - 1):
                        experts_to_append.append(deepcopy(fc))
            else:
                for _ in range(experts_num - 1):
                    experts_to_append.append(deepcopy(fc))
        else:
            experts_to_append = fc
        for ex in experts_to_append:
            for p in ex.parameters():
                p.expert_type = f"expert_type_{expert_id}"  # Different `expert_type` can have different intermediate-size
        index = 0
        for i in range(experts_num):
            if i // experts_num == 0:
                experts.append(experts_to_append[index])
                index += 1
            else:
                experts.append(None)

    assert (
        len(experts) == moe_num_experts
    ), f"experts.len={len(experts)} != experts_num={experts_num}"
    logger.info(f"MOE-GATE:-{config.moe_gate}")

    gate = gate_class[config.moe_gate.lower()](config, layer_idx=layer_idx)

    if config.multimodel_experts and config.moe_use_hard_gate and moe_num_experts > 2:
        lm_experts = experts[: config.moe_num_experts[0]]
        lm_gate = gate
    else:
        if config.multimodel_experts and config.moe_use_hard_gate:
            lm_gate, lm_experts = gate, experts
        else:
            lm_gate, lm_experts = None, None

    logger.info(f"LM-experts-{lm_experts} -- experts-{experts}")

    return gate, experts, lm_gate, lm_experts


class MoEStatics(nn.Module):
    """
    Stores MoE (Mixture of Experts) statistics
    and expert usage information.
    """

    def __init__(self, config, layer_idx):
        """
        Initialize MoE statistics tracking.

        Args:
            config: Model configuration containing MoE parameters
            layer_idx: Index of the MoE layer in the model
        """
        super().__init__()
        self._cast_to_low_precision = False
        self._cast_to_low_precison = False
        num_experts = (
            config.moe_num_experts[0]
            if config.multimodel_experts
            else config.moe_num_experts
        )
        if config.multimodel_experts:
            assert (
                len(set(config.moe_num_experts)) == 1
            ), "assume expert group has same size, got: {config.moe_num_experts}"

        with UniqueNameGuard(f"mm_layer_{layer_idx}_"):
            num_experts_groups = (
                len(config.moe_num_experts) if config.multimodel_experts else 1
            )
            p = nn.Parameter(
                torch.zeros(num_experts_groups, num_experts, dtype=torch.float32),
                requires_grad=False,
            )
            self.e_score_correction_bias = p
            p = torch.zeros(num_experts_groups, num_experts, dtype=torch.int64)
            self.expert_usage = p


def dispatching(x, dispatch_mask, scatter_index, num_experts, capacity):
    """
    Reorders input tensor based on gate results with capacity truncation and padding.

    Args:
        x (Tensor): Input tensor of shape [Seq, Dim]
        dispatch_mask (Tensor): Dispatching mask of shape [Seq, 2]
        scatter_index (Tensor): Scatter indices of shape [Seq, 2]
        num_experts (int): Number of experts
        capacity (int): Capacity per expert

    Returns:
        Tensor: Dispatched output tensor of shape [Expert*Capacity, Dim]
    """
    output = None
    orig_dtype = x.dtype
    scatter_index_unbound = [scatter_index[:, 0], scatter_index[:, 1]]
    dispatch_mask_unbound = [dispatch_mask[:, 0], dispatch_mask[:, 1]]

    for i_scatter_index, i_dispatch_mask in zip(
        scatter_index_unbound, dispatch_mask_unbound
    ):
        updates = x * i_dispatch_mask.unsqueeze(-1).to(orig_dtype)  # [seq, dim]
        init_output = torch.zeros(
            num_experts * capacity, x.shape[-1], dtype=orig_dtype, device=x.device
        )

        index = i_scatter_index.unsqueeze(-1).expand(-1, x.shape[-1])  # [seq, dim]
        if output is None:
            output = init_output.scatter_add(0, index, updates)
        else:
            output = output + init_output.scatter_add(0, index, updates)
    if output.dtype != orig_dtype:
        output = output.to(orig_dtype)
    return output


def combining(x, combine_weights, scatter_index):
    """
    Combines and aggregates input matrix using combination weights.

    Args:
        x (Tensor): Input tensor of shape [num_experts * capacity, dim]
        combine_weights (Tensor): Combination weights of shape [seq, 2]
        scatter_index (Tensor): Scatter indices of shape [seq, 2]

    Returns:
        Tensor: Combined output tensor of shape [seq, dim]
    """
    dim = x.shape[-1]

    current_device = scatter_index.device
    x = x.to(current_device)
    scatter_index = scatter_index.reshape([-1])
    num_k = combine_weights.shape[-1]

    combine_weights = combine_weights.unsqueeze(1).to(current_device)

    x = x[scatter_index].reshape([-1, num_k, dim])  # [seq, 2, dim]

    return torch.matmul(combine_weights, x).squeeze(
        1
    )  # [seq, 1, 2] @ [seq, 2, dim] -> [seq, 1, dim]


class MOELayer(nn.Module):
    """
    Mixture of Experts layer implementation based on GShard paper.
    """

    def __init__(
        self,
        gate: nn.Module,
        experts: List[nn.Module],
        layer_idx: int,
        shared_experts: Optional[List[nn.Module]] = None,
        group=None,
        recompute: bool = False,
        k: int = 2,
        all_to_all_dropout: float = 0,
        group_experts: bool = False,
        moe_statics=None,
        moe_num_experts=None,
    ):
        """
        Initialize MoE layer.

        Args:
            gate: Gate network for expert selection
            experts: List of expert networks
            layer_idx: Index of this layer in the model
            group: Distributed communication group
            recompute: Whether to enable recomputation
            k: Number of experts to select per token
            all_to_all_dropout: Dropout rate for all-to-all communication
            group_experts: Whether to group experts
            moe_statics: MoE statistics tracking object
        """
        super().__init__()
        self.gate = gate
        self.layer_idx = layer_idx

        if isinstance(experts, nn.ModuleList):
            self.experts = experts
        else:
            logger.info(f"using fused experts, type={type(experts)}")
            self.experts = experts
        self.shared_experts = shared_experts

        self.group = group
        self.k = k
        self.all_to_all_dropout = all_to_all_dropout
        self.use_correction_bias = moe_statics is not None
        self.moe_statics = moe_statics
        if self.use_correction_bias:
            logger.info(
                f"using correction bias, aux-coef:{self.gate.config.moe_aux_loss_lambda}"
            )
            assert self.gate.config.moe_use_aux_free

        self.world_size = 1
        self.rank = 0

        self.multimodal_experts = (
            isinstance(moe_num_experts, (tuple, list)) and len(moe_num_experts) > 1
        )
        self.num_local_experts = len(self.experts) // self.world_size
        if self.multimodal_experts:
            self.num_local_multimodal_experts = [
                num // self.world_size for num in moe_num_experts
            ]
            self.multimodal_expert_index = [0] + list(
                itertools.accumulate(moe_num_experts)
            )

        self.input_preprocess = self.output_postprocess = None
        self.group_experts = group_experts
        self.config = self.gate.config
        self.zero = torch.tensor(0).to(dtype=torch.float32)

    def forward_experts(self, dispatched_input):
        """
        Forward pass through experts sequentially.

        Args:
            dispatched_input: Input tensor of shape [num_experts, capacity, dim]

        Returns:
            Tensor: Expert outputs of shape [num_experts, capacity, dim]
        """

        if not self.multimodal_experts:
            true_experts = self.experts[
                self.rank
                * self.num_local_experts : (self.rank + 1)
                * self.num_local_experts
            ]
        else:
            true_experts = []
            for i, num in enumerate(self.num_local_multimodal_experts):
                current_modal_experts = self.experts[
                    self.multimodal_expert_index[i] : self.multimodal_expert_index[
                        i + 1
                    ]
                ]
                true_experts.extend(
                    current_modal_experts[self.rank * num : (self.rank + 1) * num]
                )

        dispatched_input = dispatched_input.reshape(
            [self.world_size, self.num_local_experts, -1, dispatched_input.shape[-1]]
        )
        current_device = dispatched_input.device
        expert_outputs = []
        if isinstance(self.experts, nn.ModuleList):
            chunks = dispatched_input.permute(1, 0, 2, 3).contiguous().unbind(0)
            assert len(chunks) == len(
                true_experts
            ), f"{len(chunks)}, {len(true_experts)}"
            for chunk, expert in zip(chunks, true_experts):
                expert_outputs.append(expert(chunk))
        else:
            dispatched_input = dispatched_input.permute(1, 0, 2, 3).contiguous()
            orig_shape = dispatched_input.shape
            chunks = dispatched_input.reshape(orig_shape[0], -1, orig_shape[-1])
            chunks = self.experts(chunks)
            chunks = chunks.reshape(orig_shape[:-1] + (chunks.shape[-1],)).unbind(0)
            expert_outputs.extend(chunks)

        for i, expert_output in enumerate(expert_outputs):
            expert_outputs[i] = expert_output.to(current_device)
        expert_output = torch.stack(expert_outputs, dim=1)
        return expert_output

    def moe_gate_dispatch(
        self,
        x: torch.Tensor,  # [S, H]   float16 / float32 / bfloat16
        gate_logits: torch.Tensor,  # [S, E]   float32
        k: int,
        capacity: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """dispatch input to experts based on gate logits"""

        S, H = x.shape
        E = gate_logits.shape[1]
        device = x.device
        if self.use_correction_bias:
            _, topk_idx = torch.topk(gate_logits + self.moe_statics.e_score_correction_bias[0].detach().to(gate_logits.device), k, dim=-1)
            topk_prob = torch.gather(gate_logits, dim=1, index=topk_idx) #  [Seq, k]
        else:
            topk_prob, topk_idx = torch.topk(gate_logits, k, dim=-1)  # [S, k]
        combine_weights = topk_prob  # [S, k]
        expert_id = topk_idx  # [S, k]
        y = x.new_zeros((E, capacity, H))  # [E, C, H]
        scatter_index = x.new_full((k, S), -1, dtype=torch.int32)  # [k, S]
        # per-expert slot counters
        slot_counter = torch.zeros(E, dtype=torch.int32, device=device)

        for tok in range(S):
            for route in range(k):
                e = expert_id[tok, route].item()
                slot = slot_counter[e].item()
                if slot >= capacity:  # expert is full -> drop
                    combine_weights[tok, route] = 0.0
                    continue
                # record mapping & dispatch activation
                scatter_index[route, tok] = e * capacity + slot
                y[e, slot] = x[tok]
                slot_counter[e] += 1

        expert_offset = torch.cumsum(slot_counter, 0, dtype=torch.int64)

        return y, combine_weights, scatter_index, expert_offset, expert_id

    def gate_and_dispatch(self, input, token_type_ids=None, is_multimodel=True):
        """
        Calculate gate and dispatch inputs.

        Args:
            input: Input tensor of shape [seq, dim]

        Returns:
            tuple: (dispatched_input, combine_weights, dispatch_mask,
            scatter_index, router_loss, gate_logits, gate_prob)
        """
        d_model = input.shape[1]
        if isinstance(self.gate, (TopKGate)):
            capacity = self.gate.get_capacity(
                input.shape[0], is_multimodel=is_multimodel
            )
            if token_type_ids is not None:
                token_type_ids = token_type_ids.reshape([-1])
            gate_logits = self.gate(
                input, token_type_ids=token_type_ids, is_multimodel=is_multimodel
            )
            prob = self.gate.act(gate_logits)
            (
                dispatched_input,
                combine_weights_unnorm,
                scatter_index,
                dispatch_mask,
                _,
            ) = self.moe_gate_dispatch(input, prob, k=self.k, capacity=capacity)
            dispatch_mask = torch.diff(F.pad(dispatch_mask, (1, 0)))

            scatter_index.detach()
            dispatch_mask.detach()

            scatter_index = scatter_index.transpose(0, 1)  # [k, s] -> [s, k]
            combine_weights = combine_weights_unnorm / torch.clamp(
                combine_weights_unnorm.sum(dim=-1, keepdim=True), min=1e-12
            )
            combine_weights = combine_weights.to(dtype=dispatched_input.dtype)

        else:
            (
                capacity,
                dispatch_mask,
                combine_weights,
                scatter_index,
                router_loss,
                gate_logits,
            ) = self.gate(
                input,
            )
            prob = None
            dispatched_input = dispatching(
                input,
                dispatch_mask,
                scatter_index,
                num_experts=self.world_size * self.num_local_experts,
                capacity=capacity,
            )

        dispatched_input = dispatched_input.reshape(
            [self.world_size * self.num_local_experts, capacity, d_model]
        )

        dispatch_mask = dispatch_mask.detach()
        scatter_index = scatter_index.detach()
        return (
            dispatched_input,
            combine_weights,
            dispatch_mask,
            scatter_index,
            None,
            gate_logits,
            prob,
        )

    def combine_expert_output(self, expert_output, combine_weights, scatter_index):
        """
        Combine expert outputs using combination weights.

        Args:
            expert_output: Expert outputs [num_experts, capacity, dim]
            combine_weights: Combination weights
            scatter_index: Scatter indices

        Returns:
            Tensor: Combined output [seqlen, dim]
        """
        expert_output = expert_output.reshape(
            [-1, expert_output.shape[-1]]
        )  # [e*1,c,m]

        combined_output = combining(expert_output, combine_weights, scatter_index)

        if self.output_postprocess is not None:
            combined_output = self.output_postprocess(combined_output)

        return combined_output

    def forward(
        self,
        input: torch.Tensor,
        token_type_ids=None,
        is_multimodel=True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            input: Input tensor of shape [s, d]

        Returns:
            tuple: (output, combine_weights, router_loss, gate_logits)
        """
        if input.dim() == 3:
            orig_shape = input.shape
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None
        assert (
            input.dim() == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        if token_type_ids is not None:
            token_type_ids = token_type_ids.clone()[:, :-1]

        assert self.gate is not None

        gate_input = input

        (
            dispatched_input,
            combine_weights,
            dispatch_mask,
            scatter_index,
            router_loss,
            gate_logits,
            gate_prob,
        ) = self.gate_and_dispatch(
            gate_input, token_type_ids, is_multimodel=is_multimodel
        )

        if self.shared_experts is not None:
            shared_out = self.shared_experts(input)

        expert_out = self.forward_experts(dispatched_input)

        combined_output = self.combine_expert_output(
            expert_out, combine_weights, scatter_index
        )

        if self.shared_experts is not None:
            combined_output += shared_out

        if orig_shape:
            combined_output = combined_output.clone().reshape(
                orig_shape[:-1] + (combined_output.shape[-1],)
            )
        return combined_output, combine_weights, None, gate_logits


class MOEAllGatherLayerV2(MOELayer):
    """
    MoE Layer with allgather implement.
    """

    def __init__(
        self,
        gate: nn.Module,
        experts: List[nn.Module],
        layer_idx,
        shared_experts: Optional[List[nn.Module]] = None,
        group=None,
        recompute=False,
        k=2,
        enable_reverse_token_drop=False,
        all_to_all_dropout=0,
        group_experts=False,
        use_expert_out_alltoall=True,
        use_expert_alltoall_overlap=False,
        use_padding=True,
        dense_token_type=3,  # considerd as dense tokens (no moe)
        moe_statics=None,
        moe_num_experts=None,
    ):
        super().__init__(
            gate,
            experts,
            layer_idx,
            shared_experts,
            group,
            recompute,
            k,
            all_to_all_dropout,
            group_experts,
            moe_statics,
            moe_num_experts,
        )
        self.enable_reverse_token_drop = enable_reverse_token_drop
        self.is_allgather_moe_layer = True
        self.use_padding = use_padding

        self.send_rank = None
        self.local_expert_id = None
        self.dense_experts = None
        self.dense_token_type = dense_token_type
        self.capacity_tensor = None
        logger.info(
            f"uisng MOEAllGatherLayerV2, use_expert_out_alltoall={use_expert_out_alltoall}, "  # false
            f"use_padding={use_padding}, use_expert_alltoall_overlap={use_expert_alltoall_overlap} "  # true false
            f"enable_reverse_token_drop={self.enable_reverse_token_drop}"  # false
        )
        self.two = torch.tensor(2).to(dtype=torch.float32)
        self.zero = torch.tensor(0).to(dtype=torch.float32)

    def forward(
        self,
        input: torch.Tensor,
        token_type_ids=None,
        use_dense_expert=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implements forward pass for Mixture-of-Experts (MoE) layer with distributed communication.

        Core Functionality:
          - Processes input through gating network to determine expert assignments
          - Combines expert outputs and calculates routing loss

        Key Features:
          1. Supports both dense and sparse expert computation modes
          2. Implements fused gating and dispatch for performance optimization
          3. Handles sequence length padding/unpadding for irregular inputs
          4. Enables communication-computation overlap through asynchronous operations

        Args:
            input (Tensor): Input tensor of shape [seq_len, hidden_dim]
            token_type_ids: Optional segmentation markers for heterogeneous inputs
            use_dense_expert: Flag to enable dense expert computation bypass

        Returns:
            tuple: (
                combined_output: Aggregated expert outputs [seq_len, hidden_dim],
                combine_weights: Expert combination coefficients,
            )
        """
        use_fuse = isinstance(self.gate, (TopKGate))
        assert use_fuse
        if input.ndim == 3:
            orig_shape = input.shape
            input = input.reshape([-1, input.shape[-1]])
        else:
            orig_shape = None

        assert (
            len(input.shape) == 2
        ), f"input Tensor must have dimensions: (s)equence, (d)im, got:{input.shape}"
        dispatch_token_type_ids = None
        global_dense_expert_mask = None
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, :-1].reshape([-1])
            dispatch_token_type_ids = token_type_ids
            if use_dense_expert:
                global_dense_expert_mask = (
                    dispatch_token_type_ids == self.dense_token_type
                )

        assert self.gate is not None

        (
            dispatched_input,
            global_hidden_states,
            local_combine_weights,
            expert_num_global_no_token_drop,
            expert_num_global,
            expert_num_global_list,
            local_scatter_index,
            scatter_index_rev,
            router_loss,
            (gate_logits, gate_prob),
            (gate_logits_mm, gate_prob_mm),
            expert_num_local,
        ) = self.fused_gate_and_dispatch(
            input, token_type_ids, global_dense_expert_mask
        )

        seqlen_this_mp = input.shape[0]
        if len(scatter_index_rev):
            recv_rank_local = scatter_index_rev // seqlen_this_mp
        else:
            recv_rank_local = scatter_index_rev

        if self.send_rank is None:
            capacity = self.gate.get_capacity(input.shape[0])
            self.send_rank = (
                torch.arange(1)
                .repeat_interleave(capacity * self.num_local_experts)
                .to(torch.int32)  # cap
            )
            self.local_expert_id = (
                torch.arange(self.num_local_experts)
                .repeat_interleave(capacity)
                .repeat(1)
                .to(self.send_rank.dtype)
            )
        send_rank = self.send_rank
        local_expert_id = self.local_expert_id

        expert_outs = self.forward_experts(*dispatched_input)
        for e in expert_outs:
            if e is not None:
                current_device = e.device
                break
        expert_outs = torch.cat(
            [e.to(current_device) for e in expert_outs if e is not None], dim=0
        )  # [e*c,m]

        # global -> local
        combined_output = self.combine_expert_output(
            expert_outs, local_combine_weights, local_scatter_index
        )

        if self.shared_experts is not None:
            shared_out = self.shared_experts(input).to(combined_output.device)
            combined_output += shared_out

        if orig_shape:
            combined_output = combined_output.reshape(
                *orig_shape[:-1], combined_output.shape[-1]
            )

        return combined_output, local_combine_weights, None, gate_logits

    def _expand_modality_expert_id(
        self,
        expert_id: torch.Tensor,  # (seqlen, k)
        seqlen: int,
        k: int,
        num_expert_per_modality: int,
        group_size: int,
        modality_offset: int,
        is_group_expert: bool,
    ) -> torch.Tensor:
        """
        expert_id: tensor of shape (seqlen, k), containing expert ids
        Returns: tensor of same shape, with updated expert ids
        """
        device = expert_id.device
        expert_id = expert_id.clone()

        if is_group_expert:
            # idx % k * group_size
            offsets = (torch.arange(k, device=device) * group_size).view(
                1, k
            )  # shape (1, k)
            expert_id += offsets

        if num_expert_per_modality <= 0:
            return expert_id

        # Compute rank and local expert id
        rank = expert_id // num_expert_per_modality
        expert_id_in_rank = expert_id % num_expert_per_modality

        # Compute new expert id with modality-aware adjustment
        expert_id_out = (
            rank * (num_expert_per_modality * 2)  # 2 modalities assumed
            + expert_id_in_rank
            + modality_offset * num_expert_per_modality
        )

        return expert_id_out

    def expand_modality_expert_id(
        self,
        expert_id,
        num_expert_per_modality,
        group_size,
        modality_offset,
        is_group_expert,
    ):
        """expand expert id for modality aware moe layer"""
        seq_len, k = expert_id.shape

        return self._expand_modality_expert_id(
            expert_id,
            seq_len,
            k,
            num_expert_per_modality,
            group_size,
            modality_offset,
            is_group_expert,
        )

    def fused_gate_logits_process_fused(
        self, gate_logits_lm, gate_logits_mm=None, token_type_ids=None
    ):
        """Process gating logits for expert selection in Mixture-of-Experts (MoE) layers.

        Core Functionality:
        - Transforms raw gating logits into expert selection weights and IDs
        - Supports both grouped and standard expert selection modes
        - Handles bias correction for improved expert load balancing

        Args:
            gate_logits_lm (Tensor): Raw gating scores of shape [batch_size, total_experts]

        Returns:
            tuple: (
                lm_weight_and_expert_id: Combined tensor containing selection weights
                       and expert IDs [batch_size, 2*top_k],
                prob_flat: Flattened expert probabilities [batch_size, total_experts]
            )
        """
        top_k = self.k
        num_expert_per_rank_per_modality = gate_logits_lm.shape[-1]
        group_size = gate_logits_lm.shape[-1] // top_k
        if self.group_experts:
            assert not self.use_correction_bias
            gate_logits_lm = gate_logits_lm.reshape(
                [gate_logits_lm.shape[0], top_k, -1]
            )
            prob_lm = self.gate.act(gate_logits_lm)
            prob_lm_ = prob_lm
            weight_lm, expert_id_lm = prob_lm_.topk(k=1, dim=-1)
            weight_lm = weight_lm.reshape([gate_logits_lm.shape[0], -1])
            group_size = gate_logits_lm.shape[-1]
            expert_id_lm = expert_id_lm.squeeze(-1)
        else:
            prob_lm = self.gate.act(gate_logits_lm)
            if self.use_correction_bias:
                prob_lm_ = prob_lm + self.moe_statics.e_score_correction_bias[
                    0
                ].detach().to(prob_lm.device)
            else:
                prob_lm_ = prob_lm
            weight_lm, expert_id_lm = prob_lm_.topk(k=top_k, dim=-1)

        if self.use_correction_bias:
            batch_idx = (
                torch.arange(prob_lm_.shape[0]).unsqueeze(-1).expand_as(expert_id_lm)
            )
            weight_lm = prob_lm[batch_idx, expert_id_lm]  # use correct bias

        expert_id_lm = self.expand_modality_expert_id(
            expert_id_lm,
            num_expert_per_modality=(
                num_expert_per_rank_per_modality if token_type_ids is not None else 0
            ),
            group_size=group_size,
            modality_offset=0,
            is_group_expert=self.group_experts,
        )
        expert_id_lm = expert_id_lm.reshape(weight_lm.shape)
        lm_weight_and_expert_id = torch.cat(
            [weight_lm, expert_id_lm.to(torch.float32)], -1
        )

        if token_type_ids is None or gate_logits_mm is None:
            return (
                lm_weight_and_expert_id,
                prob_lm.reshape([prob_lm.shape[0], -1]),
                None,
            )

        prob_mm = self.gate.act(gate_logits_mm)
        if self.use_correction_bias:
            prob_mm_ = prob_mm + self.moe_statics.e_score_correction_bias[
                1
            ].detach().to(prob_lm.device)
        else:
            prob_mm_ = prob_mm
        weight_mm, expert_id_mm = prob_mm_.topk(k=top_k, dim=-1)
        if self.use_correction_bias:
            batch_idx = (
                torch.arange(prob_lm_.shape[0]).unsqueeze(-1).expand_as(expert_id_lm)
            )
            weight_mm = prob_mm[batch_idx, expert_id_mm]  # use correct bias

        expert_id_mm = self.expand_modality_expert_id(
            expert_id_mm,
            num_expert_per_modality=num_expert_per_rank_per_modality,
            group_size=group_size,
            modality_offset=1,
            is_group_expert=False,
        )
        expert_id_mm = expert_id_mm.reshape(weight_mm.shape)
        mm_weight_and_expert_id = torch.cat(
            [weight_mm, expert_id_mm.to(torch.float32)], -1
        )
        weight_and_expert = torch.where(
            (token_type_ids == 0).unsqueeze(-1),
            lm_weight_and_expert_id.to(token_type_ids.device),
            mm_weight_and_expert_id.to(token_type_ids.device),
        )
        return weight_and_expert, prob_lm.reshape([prob_lm.shape[0], -1]), prob_mm

    def moe_gate_dispatch_partial_nosoftmaxtopk(
        self,
        x,
        combine_weights,
        expert_id,
        k,
        num_experts,
    ):
        """
        MoE Gate Dispatch kernel
        """
        device = x.device
        dtype = x.dtype
        num_rows, hidden_size = x.shape
        k = expert_id.shape[1]
        expert_ids_flat = expert_id.reshape(-1)  # [num_rows * k]
        combine_weights_flat = combine_weights.reshape(-1)  # [num_rows * k]

        expanded_token_ids = torch.arange(num_rows * k, device=device)  # [num_rows * k]

        sorted_expert_ids, sorted_indices = torch.sort(expert_ids_flat, stable=True)
        sorted_indices = sorted_indices.to(expanded_token_ids.device)

        sorted_expanded_token_ids = expanded_token_ids[sorted_indices]

        expert_nums_local = torch.zeros(num_experts, dtype=torch.int64, device=device)

        for expert_idx in range(num_experts):
            count = (sorted_expert_ids == expert_idx).sum().item()
            expert_nums_local[expert_idx] = count

        total_dispatched_tokens = torch.cumsum(expert_nums_local, dim=0)[-1].item()

        y = x[sorted_indices // k]  # [total_dispatched_tokens, hidden_size]

        scatter_index = torch.full((k, num_rows), -1, dtype=torch.int32, device=device)

        for i, (expanded_idx, sorted_pos) in enumerate(
            zip(sorted_expanded_token_ids, range(total_dispatched_tokens))
        ):
            token_idx = expanded_idx // k
            k_idx = expanded_idx % k
            scatter_index[k_idx, token_idx] = sorted_pos

        scatter_index_rev = sorted_indices // k

        combine_weights_out = combine_weights.clone()

        return (
            y,  # [total_dispatched_tokens, hidden_size]
            combine_weights_out,  # [num_rows, k]
            scatter_index,  # [k, num_rows]
            scatter_index_rev,  # [total_dispatched_tokens]
            expert_nums_local,  # [num_experts]
            expert_nums_local,  # [num_experts]
        )

    def fused_gate_and_dispatch(
        self, input, token_type_ids=None, global_dense_expert_mask=None
    ):
        """Implements fused expert gating and token dispatch logic for Mixture-of-Experts (MoE) layers.

        Core Functionality:
          - Computes expert selection probabilities and routing weights
          - Performs distributed token-to-expert assignment
          - Handles communication and synchronization in model-parallel environments

        Args:
            input (Tensor): Input tensor of shape [seq_len, hidden_dim]

        Returns:
            tuple: (
                dispatched_input: Expert-assigned tokens [num_experts, capacity, hidden_dim],
                global_hidden_states: Full sequence representations,
                local_combine_weights: Local expert combination weights,
                expert_num_global_notrunc: Global expert token counts (without capacity truncation),
                expert_num_global: Actual expert token counts,
                expert_num_global_list: Per-expert token counts,
                local_scatter_index: Local token reorganization indices,
                scatter_index_rev: Reverse scattering indices,
                router_loss: Calculated routing loss,
                gate_outputs: Raw gating network outputs,
                expert_num_local: Local expert utilization counts
            )
        """
        seqlen, d_model = input.shape
        args = ()
        if token_type_ids is not None:
            token_type_ids = token_type_ids.reshape([-1])
            args = (token_type_ids,)

        router_loss = torch.zeros([1], dtype=torch.float32)
        top_k = self.k

        def build_weights_and_expert_id(input):
            nonlocal token_type_ids, args
            logits = self.gate(input, *args, transform_weight=False)
            if self.config.multimodel_experts:
                gate_logits_lm, gate_logits_mm = logits.chunk(2, dim=-1)
            else:
                gate_logits_lm, gate_logits_mm = logits, None

            weigth_and_expert, gate_prob_lm, gate_prob_mm = (
                self.fused_gate_logits_process_fused(
                    gate_logits_lm,
                    gate_logits_mm,
                    token_type_ids if global_dense_expert_mask is None else None,
                )
            )
            return (
                weigth_and_expert,
                gate_logits_lm,
                gate_logits_mm,
                gate_prob_lm,
                gate_prob_mm,
            )

        capacity = self.gate.get_capacity(input.shape[0]) * self.world_size
        global_hidden_states = input
        (
            combine_weights_and_expert_id,
            gate_logits_lm,
            gate_logits_mm,
            gate_prob_lm,
            gate_prob_mm,
        ) = build_weights_and_expert_id(input)

        combine_weights_unnorm, expert_id = combine_weights_and_expert_id.chunk(
            2, dim=-1
        )
        expert_id = expert_id.to(torch.int32)
        num_experts = (
            sum(self.config.moe_num_experts)
            if isinstance(self.config.moe_num_experts, (tuple, list))
            else self.config.moe_num_experts
        )
        if global_dense_expert_mask is not None:
            combine_weights_unnorm[global_dense_expert_mask] = 0.0
            expert_id[global_dense_expert_mask] = num_experts
            num_experts += 1

        (
            dispatched_input,
            combine_weights_unnorm,
            scatter_index,  # input -> dispatched_input
            scatter_index_rev,  # dispatch-input -> input
            expert_num_global,
            expert_num_local,
        ) = self.moe_gate_dispatch_partial_nosoftmaxtopk(
            global_hidden_states,
            combine_weights_unnorm,
            expert_id,
            top_k,
            num_experts,
        )

        if self.use_correction_bias:
            if self.gate.config.multimodel_experts:
                # MLLM
                for i in range(len(self.moe_statics.expert_usage)):
                    self.moe_statics.expert_usage[i] += (
                        expert_num_local[self.gate.experts_type_mask[i]]
                        .detach()
                        .to(self.moe_statics.expert_usage.device)
                    )
            else:
                # LLM
                self.moe_statics.expert_usage[0] += expert_num_local.detach().to(
                    self.moe_statics.expert_usage.device
                )

        # When use unpad , `moe_ops_partial` output likes `scatter_index_rev==[]`.
        if scatter_index_rev.ndim == 0:
            assert not self.use_padding
            scatter_index_rev = torch.empty([0], dtype=scatter_index_rev.dtype)

        expert_num_global_notrunc = expert_num_global
        self.capacity_tensor = torch.tensor(capacity).to(dtype=expert_num_global.dtype)
        expert_num_global = torch.minimum(expert_num_global, self.capacity_tensor)

        if global_dense_expert_mask is not None:
            expert_num_global = expert_num_global[:-1]
            expert_num_local = expert_num_local[:-1]
            expert_num_global_notrunc = expert_num_global_notrunc[:-1]

        scatter_index = scatter_index.transpose(1, 0)  # [k,s] ->[s,k]
        scatter_index = scatter_index.to(combine_weights_unnorm.device)

        last_local_expert = 0
        expert_offset_global = expert_num_global.cumsum(-1)

        expert_num_global_list = expert_num_global
        if self.use_padding:
            offset = last_local_expert * capacity
        else:
            offset = 0
        local_combine_weights_unnorm = combine_weights_unnorm.contiguous()
        local_scatter_index = torch.where(
            combine_weights_unnorm > 0.0,
            scatter_index + offset,
            scatter_index,
        )
        if self.gate.norm_gate_logits:
            local_combine_weights = local_combine_weights_unnorm / torch.clip(
                local_combine_weights_unnorm.sum(-1, keepdim=True), min=1e-12
            )
        else:
            local_combine_weights = local_combine_weights_unnorm
        local_combine_weights = local_combine_weights.to(dispatched_input.dtype)
        if self.use_padding:
            dispatched_input = dispatched_input.reshape(
                [self.num_local_experts, -1, d_model]
            )
            dispatched_input = dispatched_input.unbind(0)
        else:
            s = 0
            e = self.num_local_experts
            expert_num_local = expert_num_local.tolist()[s:e]
            expert_num_local_valid = [i for i in expert_num_local if i > 0]
            valid_pos = [j for j, i in enumerate(expert_num_local) if i > 0]
            if expert_num_local_valid:
                dispatched_input_list = dispatched_input.split(expert_num_local_valid)
                dispatched_input = [None] * len(expert_num_local)
                for p, t in zip(valid_pos, dispatched_input_list):
                    dispatched_input[p] = t
            else:
                dispatched_input = [dispatched_input] + (
                    [None] * (len(expert_num_local) - 1)
                )

        expert_num_global_list = expert_num_global_list.tolist()

        return (
            dispatched_input,
            global_hidden_states,
            local_combine_weights,
            expert_num_global_notrunc,  # for auxloss calculation.
            expert_num_global,
            expert_num_global_list,
            local_scatter_index,
            scatter_index_rev,
            router_loss,
            (gate_logits_lm, gate_prob_lm),
            (gate_logits_mm, gate_prob_mm),
            expert_num_local,
        )

    def forward_experts(self, *dispatched_input):
        """Execute expert model computations in sequence for Mixture-of-Experts (MoE) layer.

        Core Functionality:
          - Distributes dispatched tokens to local expert models
          - Handles empty expert inputs with zero-initialized fallback
          - Maintains gradient flow for expert outputs
          - Aggregates outputs from all active experts

        Args:
            *dispatched_input: Variable-length expert-specific input tensors

        Returns:
            list: Expert output tensors (None for inactive experts)

        Implementation Details:
          1. Processes valid expert inputs through corresponding expert models
          2. Generates dummy inputs for inactive experts to preserve model structure
          3. Aggregates dummy outputs to first active expert to maintain gradient flow
        """
        expert_outputs = []
        assert isinstance(self.experts, nn.ModuleList), type(self.experts)

        no_tokens_expert_outputs = []
        true_experts = self.experts[
            self.rank
            * self.num_local_experts : (self.rank + 1)
            * self.num_local_experts
        ]
        for iexpert, chunk in enumerate(dispatched_input):
            if chunk is None:
                expert_outputs.append(None)
                continue

            expert_out = true_experts[iexpert](chunk.contiguous())
            expert_outputs.append(expert_out)

        if len(no_tokens_expert_outputs) > 0:
            first_has_tokens_idx = 0
            for idx, expert_out in enumerate(expert_outputs):
                if expert_out is not None:
                    first_has_tokens_idx = idx
                    break
            for idx, expert_out in enumerate(no_tokens_expert_outputs):
                expert_outputs[first_has_tokens_idx] += expert_out

        return expert_outputs


class Ernie4_5_DecoderLayer(nn.Module):
    """A single transformer decoder layer in ERNIE-MoE model.

    Contains self-attention and feed-forward components with optional MoE (Mixture of Experts)
    support, residual connections, and layer normalization.
    """

    _keep_in_fp32_modules = ["mlp.gate", "e_score_correction_bias"]

    def __init__(self, config, layer_idx):
        """Initialize the decoder layer.

        Args:
            config (Ernie4_5_MoEConfig): Model configuration.
            layer_idx (int): Index of this layer in the transformer stack
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.config = config
        self.use_moe = config.use_moe
        self.self_attn = Ernie4_5_Attention(config, layer_idx)

        moe_layer_start_index = (
            min(config.moe_layer_start_index)
            if isinstance(config.moe_layer_start_index, (tuple, list))
            else config.moe_layer_start_index
        )
        moe_layer_end_index = (
            max(config.moe_layer_end_index)
            if isinstance(config.moe_layer_end_index, (tuple, list))
            else config.moe_layer_end_index
        )

        if (
            self.use_moe
            and ((layer_idx + 1) % config.moe_layer_interval == 0)
            and layer_idx >= moe_layer_start_index  # 3
            and layer_idx <= moe_layer_end_index  # 53
        ):
            gate, experts, lm_gate, lm_experts, moe_statics = (
                self._init_gate_and_experts(layer_idx)
            )
            shared_experts = (
                self._init_shared_experts()
                if hasattr(config, "moe_num_shared_experts")
                else None
            )

            dense_experts = None
            moe_cls = MOELayer
            if config.moe_multimodal_dispatch_use_allgather:  # v2
                logger.info("Enable MOEAllGatherLayerV2!")
                moe_cls = partial(
                    MOEAllGatherLayerV2,
                    use_expert_out_alltoall="alltoall"
                    in config.moe_multimodal_dispatch_use_allgather,  # false
                    use_padding=False,
                    enable_reverse_token_drop=config.moe_reverse_token_drop,  # false
                    dense_token_type=config.moe_dense_experts_token_type_id,  # 3
                )
            else:
                assert (
                    dense_experts is None
                ), "only `MOEAllGatherLayerV2` can process dense experts"

            self.mlp = moe_cls(
                gate=gate,
                experts=experts,
                layer_idx=layer_idx,
                shared_experts=shared_experts,
                group=config.moe_group,
                recompute=False,
                k=config.moe_k,
                all_to_all_dropout=config.moe_all_to_all_dropout,
                group_experts=False,
                moe_statics=moe_statics,
                moe_num_experts=config.moe_num_experts,
            )

            _mlp_text = MOELayer(
                gate=lm_gate,
                experts=lm_experts,
                layer_idx=layer_idx,
                shared_experts=shared_experts,
                group=config.moe_group,
                recompute=False,
                k=config.moe_k,
                all_to_all_dropout=config.moe_all_to_all_dropout,
                group_experts=False,
                moe_statics=moe_statics,
                moe_num_experts=config.moe_num_experts,
            )
            self.mlp_text = (
                lambda: _mlp_text
            )  # This lambda prevents the text parameter from being scanned into the state-dict
        else:
            self.mlp = Ernie4_5_MoeMLP(config)

        Norm = Ernie4_5_MoERMSNorm

        self.input_layernorm = Norm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Norm(config.hidden_size, config.rms_norm_eps)

    def _init_shared_experts(self):
        """init shared experts

        Returns:
            _type_: _description_
        """
        cfg = deepcopy(self.config)
        if cfg.moe_num_shared_experts > 0:
            if cfg.moe_intermediate_size:
                inter_size = (
                    next(iter(cfg.moe_intermediate_size))
                    if isinstance(cfg.moe_intermediate_size, (tuple, list))
                    else cfg.moe_intermediate_size
                )
                cfg.intermediate_size = inter_size * cfg.moe_num_shared_experts
            else:
                cfg.intermediate_size = (
                    cfg.intermediate_size * cfg.moe_num_shared_experts
                )
            cfg.disable_ffn_model_parallel = False  # split shared epxert
            shared_experts = Ernie4_5_MoeMLP(cfg)
        else:
            shared_experts = None
        return shared_experts

    def _init_gate_and_experts(self, layer_idx):
        """Initialize MoE gate and expert networks.

        Args:
            layer_idx (int): Current layer index

        Returns:
            Tuple: Contains:
                - gate: MoE routing gate
                - experts: List of expert networks
                - moe_statics: Optional statistics tracker
        """
        cfg = deepcopy(self.config)
        fc_cls = Ernie4_5_MoeMLP
        if cfg.moe_intermediate_size:
            if isinstance(cfg.moe_intermediate_size, (tuple, list)):
                assert isinstance(cfg.moe_num_experts, (tuple, list)) and len(
                    cfg.moe_num_experts
                ) == len(cfg.moe_intermediate_size)
                fc = []
                for _i, (num_experts, intermediate_size) in enumerate(
                    zip(cfg.moe_num_experts, cfg.moe_intermediate_size)
                ):
                    ex_cfg = deepcopy(cfg)
                    ex_cfg.intermediate_size = intermediate_size
                    cur_modality_start_layer_idx = (
                        cfg.moe_layer_start_index[_i]
                        if isinstance(cfg.moe_layer_start_index, (tuple, list))
                        else cfg.moe_layer_start_index
                    )
                    cur_modality_end_layer_idx = (
                        cfg.moe_layer_end_index[_i]
                        if isinstance(cfg.moe_layer_end_index, (tuple, list))
                        else cfg.moe_layer_end_index
                    )
                    if (
                        layer_idx >= cur_modality_start_layer_idx
                        and layer_idx <= cur_modality_end_layer_idx
                    ):
                        if _i == 1:
                            with UniqueNameGuard(f"mm_expert_{layer_idx}_") as guard:
                                fc.append((num_experts, fc_cls(ex_cfg)))
                        else:
                            fc.append((num_experts, fc_cls(ex_cfg)))
                    else:
                        logger.info(
                            f"moe multimodal experts use Identity layer_idx: {layer_idx}"
                        )
                        fc.append((num_experts, nn.Identity()))
            else:
                cfg.intermediate_size = cfg.moe_intermediate_size
                fc = [(cfg.moe_num_experts, fc_cls(cfg, layer_idx))]
        else:
            fc = [(cfg.moe_num_experts, fc_cls(cfg, layer_idx))]
        if cfg.multimodel_experts:
            gate, experts, lm_gate, lm_experts = get_gate(self.config, fc, layer_idx)
        else:
            gate, experts = get_gate(self.config, fc, layer_idx)
            lm_gate, lm_experts = None, None

        # for AuxLoss Free Router:
        if cfg.moe_use_aux_free:
            moe_statics = MoEStatics(cfg, layer_idx)
        else:
            moe_statics = None
        return gate, experts, lm_gate, lm_experts, moe_statics

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask_start_row_indices: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_gate_logits=True,  # PP model should not output gate logits,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the decoder layer.

        Args:
            hidden_states (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size]
            attention_mask (Optional[torch.Tensor]): Attention mask tensor
            attn_mask_start_row_indices (Optional[torch.Tensor]): Indices for variable length attention
            position_ids (Optional[torch.Tensor]): Position indices for rotary embeddings
            output_attentions (Optional[bool]): Whether to return attention weights
            past_key_value (Optional[Tuple[torch.Tensor]]): Cached key/value states
            use_cache (Optional[bool]): Whether to cache key/value states
            output_gate_logits (bool): Whether to return MoE gate logits

        Returns:
            Union: Various output combinations depending on arguments:
                - Base case: Hidden states tensor
                - With attention: Tuple of (hidden_states, attention_weights)
                - With cache: Tuple of (hidden_states, cached_key_value)
                - With MoE: May include gate logits in output tuple
        """
        residual = hidden_states

        if token_type_ids is not None:
            is_multimodel_token = token_type_ids.any()
            has_dense_experts_token = (
                token_type_ids == self.config.moe_dense_experts_token_type_id
            ).any()
            is_multimodel_token_cpu = is_multimodel_token.cpu()
            has_dense_experts_token_cpu = has_dense_experts_token.cpu()
        else:
            is_multimodel_token_cpu = None
            has_dense_experts_token_cpu = None

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        (hidden_states, self_attn_weights, present_key_value, *router_loss_attn) = (
            self.self_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                attn_mask_start_row_indices=attn_mask_start_row_indices,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                token_type_ids=token_type_ids,
            )
        )
        hidden_states = hidden_states + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if isinstance(self.mlp, MOELayer):
            if is_multimodel_token_cpu:
                hidden_states, _, router_loss, gate_logits = self.mlp(
                    hidden_states, token_type_ids
                )
            else:
                hidden_states, _, router_loss, gate_logits = self.mlp_text()(
                    hidden_states, None, is_multimodel=False
                )
        else:
            hidden_states = self.mlp(hidden_states)
            gate_logits, router_loss = None, None

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if self.use_moe:
            # Non-empty only if `use_moe`
            if router_loss_attn:
                router_loss_attn = router_loss_attn[0]
                router_loss = router_loss + router_loss_attn

            if output_gate_logits:
                outputs += (gate_logits,)

        # remove empty tuple for pipeline parallel
        if type(outputs) is tuple and len(outputs) == 1:
            outputs = outputs[0]

        return outputs


class Ernie4_5_PretrainedModel(PreTrainedModel):
    """Base class for ERNIE pretrained models."""

    config_class = Ernie4_5_MoEConfig
    base_model_prefix = "ernie"
    _no_split_modules = ["Ernie4_5_DecoderLayer"]


class Ernie4_5_Model(Ernie4_5_PretrainedModel):
    """The core ERNIE transformer model with MoE (Mixture of Experts) support."""

    def __init__(self, config: Ernie4_5_MoEConfig):
        """Initialize the ERNIE model architecture.

        Args:
            config (Ernie4_5_MoEConfig): Model configuration.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.config = config

        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        self.layers = nn.ModuleList(
            [Ernie4_5_DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        Norm = Ernie4_5_MoERMSNorm
        self.norm = Norm(config.hidden_size, config.rms_norm_eps)

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        attn_mask_start_row_indices=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=None,
        return_dict=False,
    ):
        """Forward pass through the ERNIE model.

        Args:
            input_ids (Optional[torch.Tensor]): Input token IDs
            position_ids (Optional[torch.Tensor]): Position indices
            attention_mask (Optional[torch.Tensor]): Attention mask
            attn_mask_start_row_indices (Optional[torch.Tensor]): Variable length attention indices
            inputs_embeds (Optional[torch.Tensor]): Precomputed embeddings
            use_cache (Optional[bool]): Whether to cache key/value states
            past_key_values (Optional[Tuple[Tuple[torch.Tensor]]]): Cached key/value states
            output_attentions (Optional[bool]): Whether to output attention weights
            output_hidden_states (Optional[bool]): Whether to output all hidden states
            return_dict (Optional[bool]): Whether to return dict or tuple

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
                Various outputs depending on configuration, including:
                - last_hidden_state: Final layer hidden states
                - past_key_values: Cached key/value states if use_cache=True
                - hidden_states: All hidden states if output_hidden_states=True
                - attentions: Attention weights if output_attentions=True
                - router_loss: MoE router loss if use_moe=True
                - gate_logits: MoE gate logits if use_moe=True
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            _, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            _, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[1]
            seq_length_with_past += cache_length
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(self.embed_tokens.weight.dtype)

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        if getattr(self.config, "use_moe", False):
            all_router_loss = torch.tensor(0.0).to(device=inputs_embeds.device)
        else:
            all_router_loss = None
        all_gate_logits = ()

        for idx, (decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask,
                attn_mask_start_row_indices,
                position_ids,
                token_type_ids,
                output_attentions,
                past_key_value,
                use_cache,
            )

            if isinstance(layer_outputs, (tuple, list)):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            if self.config.use_moe:
                layer_outputs, gate_logits = layer_outputs[:-1], layer_outputs[-1]
                all_gate_logits = all_gate_logits + (gate_logits,)

            if past_key_value is not None:
                hidden_states = hidden_states[:, -1:, :]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_loss,
                    all_gate_logits,
                ]
                if v is not None
            )

        # assert all_router_loss is None, f'moe not support `return-dict`'
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=None,
            router_loss=all_router_loss,
            gate_logits=all_gate_logits,
        )


class Ernie4_5_MoeForCausalLM(Ernie4_5_PretrainedModel, GenerationMixin):
    """ERNIE Mixture of Experts (MoE) model for causal language modeling."""

    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        """
        Initializes the ERNIE MoE model for causal language modeling.

        Args:
            config (dict): Model configuration.
        """
        super().__init__(config)

        # initialize-trick for big model,
        # see https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/README.md#std-init
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(
            f"change initializer-range from {config.initializer_range} to {new_initializer_range}"
        )
        config.initializer_range = new_initializer_range
        self.config = config
        self.model = Ernie4_5_Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.use_bias)

        self.post_init()  # maybe weight share

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @staticmethod
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder=False):
        """
        Updates model kwargs for generation.

        Args:
            outputs (Any): Model outputs.
            model_kwargs (dict): Current model kwargs.
            is_encoder_decoder (bool): Whether using encoder-decoder architecture.

        Returns:
            dict: Updated model kwargs.
        """
        # update cache
        if isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], torch.Tensor):
            model_kwargs["past_key_values"] = outputs[1]

        if isinstance(outputs, CausalLMOutputWithCrossAttentions) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs and model_kwargs["token_type_ids"] is not None:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1:]], dim=-1)

        if not is_encoder_decoder and model_kwargs.get("attention_mask", None) is not None:
            # update attention mask
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), dtype=torch.int64, device=attention_mask.device),
                ],
                dim=-1,
            )

        # update role_ids
        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = torch.cat([role_ids, role_ids[:, -1:]], dim=-1)

        if self.config.get('rope_3d', False):
            assert "position_ids" in model_kwargs, "position_ids must be provided if rope_3d is on"
            position_ids = model_kwargs["position_ids"]
            bsz = position_ids.shape[0]

            max_position = position_ids.max(dim=1, keepdim=True)[0]  # [batch_size, 1, hidden_dim]
            new_positions = max_position + 1

            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_positions],
                dim=1
            )

        return model_kwargs


class VisionMlp(nn.Module):
    """VisionMLP"""

    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: VisionMLP output tensor
        """
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    """PatchEmbed"""

    def __init__(
        self,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        """
        Args:
            patch_size (int, optional): patch size. Defaults to 14.
            in_channels (int, optional): number of channels. Defaults to 3.
            embed_dim (int, optional): embedding dimension. Defaults to 1152.
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Linear(
            in_channels * patch_size * patch_size, embed_dim, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): hidden states

        Returns:
            torch.Tensor: output tensor
        """
        target_dtype = self.proj.weight.dtype

        hidden_states = self.proj(hidden_states.to(target_dtype))

        return hidden_states


class VisionRotaryEmbedding(nn.Module):
    """VisionRotaryEmbedding"""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """
        Args:
            dim (int): the dimension of each token.
            theta (float, optional): the frequency factor. Defaults to 10000.0.
        """
        super().__init__()
        self.inv_freq = 1.0 / theta ** (
            torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        """
        Args:
            seqlen (int): length of sequence.

        Returns:
            torch.Tensor: rotary position embedding
        """
        seq = torch.arange(seqlen).to(self.inv_freq.dtype)
        freqs = torch.outer(input=seq, vec2=self.inv_freq)
        return freqs


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)  # shape is the same as x


def apply_rotary_pos_emb_vision(
    tensor: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Applies Rotary Position Embedding to the input tensors.

    Args:
        tensor (torch.Tensor): The input tensor.
        freqs (torch.Tensor): The frequencies used for the rotation.
    Returns:
        output (torch.Tensor): the tensor rotated using the Rotary Position Embedding.
    """
    orig_dtype = tensor.dtype

    tensor = tensor.type(dtype=torch.float32)
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).tile(1, 1, 2).unsqueeze(0).type(dtype=torch.float32)
    sin = sin.unsqueeze(1).tile(1, 1, 2).unsqueeze(0).type(dtype=torch.float32)
    output = tensor * cos + rotate_half(tensor) * sin
    output = output.to(orig_dtype)
    return output


class VisionAttention(nn.Module):
    """VisionAttention"""

    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.head_dim = dim // num_heads  # must added

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """forward function for vision attention"""
        seq_length = hidden_states.shape[0]
        qkv = (
            self.qkv(hidden_states)
            .reshape([seq_length, 3, self.num_heads, -1])
            .permute(1, 0, 2, 3)
        )
        q, k, v = qkv.unbind(axis=0)

        q = apply_rotary_pos_emb_vision(q.unsqueeze(dim=0), rotary_pos_emb).squeeze(
            dim=0
        )
        k = apply_rotary_pos_emb_vision(k.unsqueeze(dim=0), rotary_pos_emb).squeeze(
            dim=0
        )

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=1) for tensor in (q, k, v)
        ]

        attn_output = []
        for q, k, v in zip(*splits):
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_output_splited = torch.matmul(attn_weights, v)
            attn_output_splited = attn_output_splited.transpose(0, 1)
            attn_output.append(attn_output_splited)
        attn_output = torch.cat(attn_output, dim=0)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class DFNRopeVisionBlock(nn.Module):
    """DFNRopeVisionBlock"""

    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        """
        Args:
            config (dict): model configuration.
            attn_implementation (str, optional): attention implementation. Defaults to "sdpa".
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = VisionAttention(config.embed_dim, num_heads=config.num_heads)
        self.mlp = VisionMlp(
            dim=config.embed_dim,
            hidden_dim=mlp_hidden_dim,
            hidden_act=config.hidden_act,
        )
        self.config = config

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        """
        Args:
            hidden_states(torch.Tensor): hidden states
            cu_seqlens (torch.Tensor): cumulative sequence lengths
            rotary_pos_emb: rotary position embedding

        Returns:
            torch.Tensor: output tensor
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class DFNRopeVisionTransformerPreTrainedModel(PreTrainedModel):
    """DFNRopeVisionTransformerPreTrainedModel"""

    config_class = DFNRopeVisionTransformerConfig
    _tp_plan = {}

    def __init__(self, config) -> None:
        """
        Args:
            config (dict): model configuration
        """
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [DFNRopeVisionBlock(config) for _ in range(config.depth)]
        )

        assert (
            config.hidden_size == config.embed_dim
        ), "in DFNRope, vit's config.hidden must be equal to config.embed_dim"
        self.ln = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def rot_pos_emb(self, grid_thw, num_pad=0):
        """rot_pos_emb

        Args:
            grid_thw (torch.Tensor): grid thw of input

        Returns:
            torch.Tensor: rotary position embedding
        """
        pos_ids = []
        grid_hw_array = np.array(grid_thw.cpu(), dtype=np.int64)
        for t, h, w in grid_hw_array:
            hpos_ids = np.arange(h).reshape([-1, 1])
            hpos_ids = np.tile(hpos_ids, (1, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = np.transpose(hpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.flatten()

            wpos_ids = np.arange(w).reshape([1, -1])
            wpos_ids = np.tile(wpos_ids, (h, 1))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = np.transpose(wpos_ids, (0, 2, 1, 3))
            wpos_ids = wpos_ids.flatten()

            stacked_ids = np.stack([hpos_ids, wpos_ids], axis=-1)
            tiled_ids = np.tile(stacked_ids, (t, 1))
            pos_ids.append(tiled_ids)

        pos_ids = np.concatenate(pos_ids, axis=0)
        if num_pad > 0:
            pos_ids = np.concatenate(
                [pos_ids, np.zeros((num_pad, 2), dtype=pos_ids.dtype)]
            )
        max_grid_size = np.amax(grid_hw_array[:, 1:])
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(start_dim=1)
        return rotary_pos_emb

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, num_pad=0
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): input tensor
            grid_thw (torch.Tensor): grid thw of input
            num_pad (int): number of padding tokens

        Returns:
            torch.Tensor: output tensor
        """
        hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw, num_pad=num_pad)
        rotary_pos_emb = rotary_pos_emb.to(hidden_states.device)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)

        if num_pad > 0:
            cu_seqlens = F.pad(cu_seqlens, (1, 1), value=0)
            cu_seqlens[-1] = cu_seqlens[-2] + num_pad
        else:
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for idx, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )

        ret = self.ln(hidden_states)  # add norm
        return ret


class VariableResolutionResamplerModel(nn.Module):
    """
    VariableResolutionResamplerModel, support variable resolution
    """

    def __init__(self, in_dim, out_dim, spatial_conv_size, temporal_conv_size, config):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.config = config
        self.spatial_conv_size = spatial_conv_size
        self.temporal_conv_size = temporal_conv_size
        self.use_temporal_conv = config.use_temporal_conv

        # compress 2d conv(picture) to 1d
        self.spatial_dim = self.in_dim * self.spatial_conv_size * self.spatial_conv_size
        # compress 3d conv(video) to 1d
        self.temporal_dim = (
            self.in_dim
            * self.spatial_conv_size
            * self.spatial_conv_size
            * self.temporal_conv_size
        )

        # using unique name space start with "mm_resampler_"
        with UniqueNameGuard("mm_resampler_") as guard:

            self.spatial_linear = nn.Sequential(
                nn.Linear(self.spatial_dim, self.spatial_dim),
                nn.GELU(),
                nn.Linear(self.spatial_dim, self.spatial_dim),
                nn.LayerNorm(self.spatial_dim, eps=1e-6),
            )

            if self.use_temporal_conv:
                self.temporal_linear = nn.Sequential(
                    nn.Linear(self.temporal_dim, self.spatial_dim),
                    nn.GELU(),
                    nn.Linear(self.spatial_dim, self.spatial_dim),
                    nn.LayerNorm(self.spatial_dim, eps=1e-6),
                )

            self.mlp = nn.Linear(self.spatial_dim, self.out_dim)

            out_config = deepcopy(config)
            out_config.hidden_size = out_dim
            self.after_norm = Ernie4_5_MoERMSNorm(out_config.hidden_size, config.rms_norm_eps)

    def spatial_conv_reshape(self, x, spatial_conv_size):
        """
        reshape before linear to imitation conv
        """
        S, C = x.shape
        x = x.reshape([-1, C * (spatial_conv_size**2)])
        return x

    def forward(self, x, image_mask, token_type_ids, image_type_ids, grid_thw):
        """
        x: image_features
        image_mask: [B]
        token_types_ids: [B]
        image_type_ids:  [B_image]
        grid_thw: [B_image, 3]
        """
        assert image_type_ids is not None

        def fwd_spatial(x):
            """
            x in the shape of [S, H]
            S is ordered in the following way: [ [patch_h*patch_w (row-major traversal)] * patch_time]
            H is simply hidden
            """
            x = self.spatial_conv_reshape(x, self.spatial_conv_size)

            x = self.spatial_linear(x)

            return x

        def fwd_placeholder(x, grid_thw, to_tensor=False):
            """
            x: [S, H]
            grid_thw: [S, 3]
                the second dimension: [t, h, w]
            """

            grid_thw_cpu = grid_thw.cpu().numpy()
            grid_t, grid_hw = grid_thw_cpu[:, 0], grid_thw_cpu[:, 1:]
            grid_hw_after_conv = grid_hw.prod(-1) // (self.spatial_conv_size**2)

            tokens_per_img_or_vid = grid_thw_cpu.prod(-1) // (self.spatial_conv_size**2)
            batch_offset = np.empty(
                tokens_per_img_or_vid.size, dtype=tokens_per_img_or_vid.dtype
            )
            batch_offset[0] = 0
            batch_offset[1:] = tokens_per_img_or_vid.cumsum()[:-1]

            assert (
                self.temporal_conv_size == 2
            ), f"Hard Code: temporal_conv_size==2, got:{self.temporal_conv_size}"

            # TODO: support any temporal conv size
            slice_offsets = []
            for temporoal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(0, temporoal_size, 2):
                    slice_offsets.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets = torch.tensor(np.concatenate(slice_offsets, axis=-1)).to(
                x.device
            )

            slice_offsets2 = []
            for temporoal_size, spatial_size, b_offset in zip(
                grid_t, grid_hw_after_conv, batch_offset
            ):
                for temp_offset in range(
                    1 if temporoal_size > 1 else 0, temporoal_size, 2
                ):
                    slice_offsets2.append(
                        np.arange(
                            b_offset + (temp_offset) * spatial_size,
                            b_offset + (temp_offset + 1) * spatial_size,
                        )
                    )
            slice_offsets2 = torch.tensor(np.concatenate(slice_offsets2, axis=-1)).to(
                x.device
            )

            x_timestep_1 = torch.index_select(x, dim=0, index=slice_offsets)
            x_timestep_2 = torch.index_select(x, dim=0, index=slice_offsets2)
            x = torch.concat([x_timestep_1, x_timestep_2], dim=-1)
            return x

        def fwd_temporal(x):
            x = self.temporal_linear(x)
            return x

        def fwd_mlp(x):
            x = self.mlp(x)
            x = self.after_norm(x)
            return x

        x = fwd_spatial(x)
        if self.use_temporal_conv:
            x = fwd_placeholder(x, grid_thw)
            x = fwd_temporal(x)
        x = fwd_mlp(x)
        return x


class Ernie4_5_VLMoeForConditionalGeneration(Ernie4_5_MoeForCausalLM):
    """Ernie4_5_VLMoeForConditionalGeneration"""

    config_class = Ernie4_5_VLMoEConfig
    main_input_name = "pixel_values"
    _keep_in_fp16_modules = ["vision_model"]
    _tp_plan = {}

    def __init__(
        self, config: Ernie4_5_VLMoEConfig, vision_model=None, resampler_model=None
    ):
        """
        initialize Ernie4_5_VLMoeForConditionalGeneration

        Args:
            config(Ernie4_5_VLMoEConfig): Model configuration.
            vision_model(nn.Module): vision model
            resampler_model(nn.Module): resampler model
        """
        super().__init__(config)

        self.vision_model = DFNRopeVisionTransformerPreTrainedModel(
            config.vision_config
        )

        self.model.resampler_model = VariableResolutionResamplerModel(
            config.pixel_hidden_size,
            config.hidden_size,
            config.spatial_conv_size,
            config.temporal_conv_size,
            config=config,
        )

        self.image_preprocess = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def add_image_preprocess(self, processor):
        """add image preprocess"""
        logger.info("image preprocess is set")

        image_preprocess = processor.image_processor
        image_preprocess.image_mean_tensor = torch.tensor(
            image_preprocess.image_mean, dtype=torch.float32
        ).reshape([1, 3, 1, 1])
        image_preprocess.image_std_tensor = torch.tensor(
            image_preprocess.image_std, dtype=torch.float32
        ).reshape([1, 3, 1, 1])
        image_preprocess.rescale_factor = torch.tensor(
            image_preprocess.rescale_factor, dtype=torch.float32
        )
        image_preprocess.image_mean_tensor = image_preprocess.image_mean_tensor.squeeze(
            [-2, -1]
        ).repeat_interleave(self.config.vision_config.patch_size**2 * 1, -1)
        image_preprocess.image_std_tensor = image_preprocess.image_std_tensor.squeeze(
            [-2, -1]
        ).repeat_interleave(self.config.vision_config.patch_size**2 * 1, -1)

        self.image_preprocess = image_preprocess

    def vision_forward(
        self,
        images,
        image_position_ids,
        image_attention_mask,
        grid_thw,
    ):
        """vision_forward"""
        if self.image_preprocess is not None:
            assert images.dtype == torch.uint8, images.dtype
            current_device = images.device
            self.image_preprocess.image_mean_tensor = (
                self.image_preprocess.image_mean_tensor.to(current_device)
            )
            self.image_preprocess.image_std_tensor = (
                self.image_preprocess.image_std_tensor.to(current_device)
            )
            images = self.image_preprocess.rescale_factor * images.to(torch.float32)
            images = (
                images - self.image_preprocess.image_mean_tensor
            ) / self.image_preprocess.image_std_tensor
            images = images.to(torch.bfloat16)
        else:
            assert images.dtype == torch.bfloat16, images.dtype
        # logger.info(f"extract feature input - {images}--{grid_thw}")
        if grid_thw is not None:
            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
            grid_thw = F.pad(
                torch.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0),
                [1, 0, 0, 0],
                value=1,
            )
        image_features = self.vision_model(images, grid_thw)
        return image_features

    def vision_mapping_forward(
        self,
        token_type_ids,
        token_type_ids_w_video,
        input_ids,
        mm_input_ids,
        image_features,
        inputs_embeds,
        image_type_ids,
        grid_thw,
    ):
        """vision_mapping_forward"""
        image_mask = input_ids == self.config.im_patch_id
        image_features = self.model.resampler_model(
            image_features,
            image_mask,
            token_type_ids_w_video,
            image_type_ids,
            grid_thw,
        )

        if image_features.dim == 2:
            B, N, C = image_features.shape
            image_features = image_features.reshape([B * N, C]).to(inputs_embeds.dtype)
        # Will overwrite the part of `ids==im_patch_id` in `mm_ids_features`
        inputs_embeds[image_mask.to(inputs_embeds.device)] = image_features.to(
            inputs_embeds.device
        )
        return inputs_embeds

    def prepare_inputs_for_generation(
        self,
        input_ids,
        images=None,
        use_cache=False,
        past_key_values=None,
        inputs_embeds=None,
        image_position_ids=None,
        image_attention_mask=None,
        token_type_ids=None,
        image_type_ids=None,
        grid_thw=None,
        **kwargs,
    ):
        """
        Prepare inputs for the decoder that can be used for generation.

        Args:
            input_ids (torch.Tensor): Input ids.
            images (torch.Tensor): Images. Default to None.
            use_cache (bool): Whether to use cache. Default to False.
            past_key_values (list): Past key values. Default to None.
            inputs_embeds (torch.Tensor): Input embeddings. Default to None.
            image_position_ids (torch.Tensor): Image position ids. Default to None.
            image_attention_mask (torch.Tensor): Image attention mask. Default to None.
            token_type_ids (torch.Tensor): Token type ids. Default to None.
            image_type_ids (torch.Tensor): Image type ids. Default to None.
            grid_thw (torch.Tensor): Grid thw. Default to None.
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            image_type_ids = (
                image_type_ids[:, -1:] if image_type_ids is not None else None
            )

        if self.config.use_flash_attention:
            attention_mask = None
        else:
            attention_mask = kwargs.get("attention_mask", None)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": True,
                "attention_mask": attention_mask,
                "images": images,
                "image_position_ids": image_position_ids,
                "image_attention_mask": image_attention_mask,
                "image_type_ids": image_type_ids,
                "token_type_ids": torch.cat(
                    [
                        token_type_ids,
                        torch.zeros(
                            [len(token_type_ids), 1], dtype=token_type_ids.dtype
                        ).to(token_type_ids.device),
                    ],
                    dim=-1,
                ),
                "grid_thw": grid_thw,
            }
        )
        if self.config.rope_3d:
            model_inputs.update({"position_ids": kwargs["position_ids"]})

        return model_inputs

    def _post_init(self, original_init, *args, **kwargs):
        """
        Label all multimodal parameters in the model, only head and Embedding
        Experts parameters are already labeled
        """
        super()._post_init(self, original_init, *args, **kwargs)
        if self.lm_head.mm_head is not None:
            self.lm_head.mm_head.weight.expert_type = "expert_type_1"
        if getattr(self.lm_head.mm_head, "bias", None) is not None:
            self.lm_head.mm_head.bias.expert_type = "expert_type_1"

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        ignored_index: Optional[int] = 0,
        return_dict: Optional[bool] = None,
        image_position_ids: Optional[torch.Tensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        image_type_ids: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward for Ernie4_5_VLMoeForConditionalGeneration

        Args:
            input_ids (torch.Tensor): Input ids.
            position_ids (Optional[torch.Tensor], optional): Position ids. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            past_key_values (Optional[List[torch.Tensor]], optional): Past key values. Defaults to None.
            use_cache (Optional[bool], optional): Use cache. Defaults to None.
            output_attentions (Optional[bool], optional): Output attentions. Defaults to None.
            output_hidden_states (Optional[bool], optional): Output hidden states. Defaults to None.
            labels (Optional[torch.Tensor], optional): Labels. Defaults to None.
            images (Optional[torch.Tensor]): Images. Defaults to None.
            ignored_index (Optional[int], optional): Ignored index. Defaults to 0.
            return_dict (Optional[bool], optional): Return dict. Defaults to None.
            image_position_ids (Optional[torch.Tensor], optional): Image position ids. Defaults to None.
            image_attention_mask (Optional[torch.Tensor], optional): Image attention mask. Defaults to None.
            token_type_ids (Optional[torch.Tensor], optional): Token type ids. Defaults to None.
            image_type_ids (Optional[torch.Tensor], optional): Image type ids. Defaults to None.
            grid_thw (Optional[torch.Tensor], optional): Grid thw. Defaults to None.
        """
        if grid_thw is not None:
            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_mask = input_ids == self.config.im_patch_id

        image_rate = image_mask.to(torch.float32).mean()

        if past_key_values is None:
            if images is not None:
                assert (image_mask).any().item(), (
                    image_mask.detach().cpu().numpy().tolist(),
                    input_ids.detach().cpu().numpy().tolist(),
                    self.config.im_patch_id,
                    images.shape,
                )
                image_features = self.vision_forward(
                    images,
                    image_position_ids,
                    image_attention_mask,
                    grid_thw,
                )
            else:
                image_features = None  # no more faking
        else:
            image_features = None
        if token_type_ids is None:
            token_type_ids = image_mask.to(torch.int64)
            token_type_ids_labels = torch.cat(
                [token_type_ids[:, 1:], token_type_ids[:, -1:]], 1
            )
        else:
            assert (
                token_type_ids.shape[1] == input_ids.shape[1] + 1
            ), f"token_type:{token_type_ids.shape}, ids:{input_ids.shape}"
            token_type_ids_labels = token_type_ids[..., 1:]

        lm_input_ids = input_ids.clone()
        mm_input_ids = input_ids.clone()

        inputs_embeds = self.model.embed_tokens(lm_input_ids)
        token_type_ids_w_video = token_type_ids[..., :-1].clone()
        token_type_ids[token_type_ids == TokenType.video] = TokenType.image

        if images is not None and image_features is not None:
            inputs_embeds = self.vision_mapping_forward(
                token_type_ids[..., :-1],
                token_type_ids_w_video,
                input_ids,
                mm_input_ids,
                image_features,
                inputs_embeds,
                image_type_ids,
                grid_thw,
            )
        else:
            pass  # do nothing, should not hang under DygraphShardingOptimizerV2

        outputs = self.model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not use_cache:
            assert outputs.last_hidden_state.shape[:2] == token_type_ids_labels.shape, (
                outputs.last_hidden_state.shape,
                token_type_ids_labels.shape,
            )
            if self.config.use_recompute_loss_fn:
                logits = outputs.last_hidden_state
            else:
                logits = self.lm_head(outputs.last_hidden_state)
        else:
            logits = self.lm_head(outputs.last_hidden_state[:, -1:, :])

        router_loss = outputs.router_loss

        # aka Generate Decoding
        loss = None
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_loss=outputs.router_loss,
        )

    @staticmethod
    def _resolve_prefix_keys(state_keys_base, state_keys_real, ignore_error=False):
        """_resolve_prefix_keys"""
        # state_keys_map base to real
        state_keys_map = {}

        state_keys_base = set(state_keys_base)
        state_keys_real = set(state_keys_real)

        for key in state_keys_base:
            for x in state_keys_real:
                if "mm_embed_tokens" in x:
                    if "mm_embed_tokens" in key:
                        state_keys_map[key] = x
                        break
                elif x.endswith(key):
                    state_keys_map[key] = x
                    break
            if key not in state_keys_map:
                if not ignore_error:
                    logger.error(f"could not find name {key} in loaded state dict!")
            else:
                state_keys_real.remove(state_keys_map[key])

        return state_keys_map


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model outputs with past key values and cross attention layers,
    with additional support for router components in mixture-of-experts models.

    This extends the base model output to include:
    1. Router-related outputs for expert selection
    2. Maintains all existing functionality from the parent class
    """

    last_hidden_state: Optional[Tuple[torch.Tensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    cross_attentions: Optional[Tuple[torch.Tensor]] = None
    router_loss: Optional[torch.Tensor] = None
    gate_logits: Optional[Tuple[torch.Tensor]] = None


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True`
            is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or
            when `config.output_attentions=True`):
            Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        router_loss (Optional[torch.Tensor]):
            The routing loss computed by the gating network in mixture-of-experts models.
            This is typically the load balancing loss that encourages equal expert utilization.
            None when not using mixture-of-experts routing.
    """

    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    router_loss: Optional[Tuple[torch.Tensor]] = None


__all__ = [
    "Ernie4_5_VLMoeForConditionalGeneration",
    "DFNRopeVisionTransformerPreTrainedModel",
    "VariableResolutionResamplerModel",
]
