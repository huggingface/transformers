from typing import Tuple

import torch
import torch.nn as nn

from ..enums import AttentionHeadType, PositionEmbeddingType
from ..position_embedding import rotate_half


class Attention(nn.Module):
    """Attention class used by all Megatron models"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_head_type: AttentionHeadType,
        position_embedding_type: PositionEmbeddingType,
        causal: bool,
        add_bias: bool,
        scale_attention_weights: bool,
        attention_softmax_in_fp32: bool,
        scale_attention_softmax_in_fp32: bool,
        attn_pdrop: float,
        resid_pdrop: float,
        layer_idx: int = None,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.mask_value = None
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias

        assert (
            self.hidden_size % self.num_heads == 0
        ), f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.hidden_size // self.num_heads
        self.attention_head_type = attention_head_type

        self.position_embedding_type = position_embedding_type
        self.scale_attn_weights = scale_attention_weights

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = scale_attention_softmax_in_fp32 and attention_softmax_in_fp32

        if self.attention_head_type == AttentionHeadType.mha:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = self.num_heads

            assert (
                self.num_heads == self.num_key_value_heads
            ), f"{self.__class__.__name__} should have same number of heads for query, keys and values"
        elif self.attention_head_type == AttentionHeadType.gqa:
            assert (
                self.num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            assert self.num_heads % self.num_key_value_heads == 0, (
                f"`num_heads` ({self.num_heads}) should be a multiple of `num_key_value_heads` "
                f"({self.num_key_value_heads})"
            )
        elif attention_head_type == AttentionHeadType.mqa:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = 1

            assert self.num_key_value_heads == 1, f"{self.__class__.__name__} should have 1 head for keys and values"
        else:
            raise ValueError(f"unexpected attention_head_type ({attention_head_type})")

        # note that the actual layout is different for the output and depends on whether we are using MHA, MQA or GQA
        # (self.hidden_size + 2 * self.num_key_value_heads * self.head_dim) is just the actual number output features
        self.c_attn = nn.Linear(
            self.hidden_size, self.hidden_size + 2 * self.num_key_value_heads * self.head_dim, bias=self.add_bias
        )
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.add_bias)

        self.attn_pdrop = attn_pdrop

        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

    def _prepare_qkv_for_forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        # the output of following is a tuple if using MQA with tensor parallel
        hidden_states = self.c_attn(hidden_states)

        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, [num_heads + num_key_value_heads * 2] * head_dim)
        # ==========================================================================================

        # for MHA, we can get away with doing just 1 transpose which is not true for GQA
        if self.attention_head_type == AttentionHeadType.mha:
            query, key, value = self._prepare_qkv_for_forward_mha(hidden_states)
        elif self.attention_head_type == AttentionHeadType.gqa:
            query, key, value = self._prepare_qkv_for_forward_gqa(hidden_states)
        elif self.attention_head_type == AttentionHeadType.mqa:
            query, key, value = self._prepare_qkv_for_forward_mqa(hidden_states)
        else:
            raise ValueError(f"unexpected attention_head_type ({self.attention_head_type})")

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        return query, key, value

    def _prepare_qkv_for_forward_mha(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        hidden_states = hidden_states.view(batch_size, query_length, self.num_heads, -1)
        hidden_states = hidden_states.transpose(1, 2)

        query, key, value = hidden_states.chunk(3, dim=-1)

        return query, key, value

    def _prepare_qkv_for_forward_gqa(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        hidden_states = hidden_states.view(batch_size, query_length, self.num_key_value_heads, -1)

        query, key, value = hidden_states.split(
            ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim), dim=-1
        )

        # this needs to be a reshape instead of view sadly
        query = query.reshape(batch_size, query_length, -1, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        return query, key, value

    def _prepare_qkv_for_forward_mqa(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        query, key, value = hidden_states.split((self.hidden_size, self.head_dim, self.head_dim), dim=-1)

        query = query.view(batch_size, query_length, self.num_heads, -1)

        query = query.transpose(1, 2)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        return query, key, value

    def _get_mask_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(torch.float16).min, dtype=dtype, device=device)
        return self.mask_value

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ==========================================================================================
        # q -> (batch_size, num_heads, query_length, head_dim)
        # k -> (batch_size, num_key_value_heads, key_length, head_dim)
        # cos -> (key_length, head_dim)
        # sin -> (key_length, head_dim)
        # position_ids -> (batch_size, query_length)
        # ==========================================================================================

        cos, sin = cos_sin
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)

        # ==========================================================================================
        # cos -> (batch_size, 1, query_length, head_dim)
        # sin -> (batch_size, 1, query_length, head_dim)
        # ==========================================================================================

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

        # ==========================================================================================
        # q_embed -> (batch_size, num_heads, query_length, head_dim)
        # k_embed -> (batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        return q_embed, k_embed
