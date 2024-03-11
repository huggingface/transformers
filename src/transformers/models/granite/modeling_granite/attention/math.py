from typing import Tuple, Union

import torch
import torch.nn as nn

from ..enums import AttentionHeadType, PositionEmbeddingType
from .base import Attention


class MathAttention(Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        alibi_bias: torch.Tensor = None,
        rope_cos_sin: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query, key = self._apply_rotary_pos_emb(query, key, rope_cos_sin, position_ids)

        if layer_past is not None:
            key = torch.cat((layer_past[0], key), dim=2)
            value = torch.cat((layer_past[1], value), dim=2)

        present = (key, value) if use_cache else None

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, key_length, head_dim)
        # value -> (batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        attn_output, attn_weights = self._attention(
            query,
            key,
            value,
            attention_mask,
            head_mask,
            alibi_bias,
        )

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            if self.attention_head_type == AttentionHeadType.mqa:
                # Transpose to return weights in the usual format (batch_size, num_heads, query_length, key_length)
                attn_weights = attn_weights.transpose(1, 2)
            elif self.attention_head_type != AttentionHeadType.mha:
                raise NotImplementedError()

            outputs += (attn_weights,)

        return outputs

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        alibi_bias: torch.Tensor = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, key_length, head_dim)
        # value -> (batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        key = key.transpose(-1, -2)

        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype
        upcast = dtype != softmax_dtype

        unscale = self.layer_idx + 1 if self.scale_attention_softmax_in_fp32 and upcast else 1
        scale_factor = 1 / unscale
        if self.scale_attn_weights:
            scale_factor /= self.head_dim**0.5

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, head_dim, key_length)
        # value -> (batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        batch_size = query.shape[0]
        query_length = query.shape[2]
        key_length = key.shape[-1]

        key, value = self._repeat_key_value(key, value)

        # Always copies
        query = query.reshape(batch_size * self.num_heads, query_length, self.head_dim)
        # No copy when layer_past is provided.
        key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)

        # ==========================================================================================
        # query -> (batch_size * num_heads, query_length, head_dim)
        # key -> (batch_size * num_heads, head_dim, key_length)
        # value -> (batch_size, num_heads, key_length, head_dim)
        # ==========================================================================================

        if alibi_bias is None:
            attn_weights = torch.empty(
                (batch_size * self.num_heads, query_length, key_length), device=query.device, dtype=query.dtype
            )
            beta = 0
        else:
            attn_weights = alibi_bias.view(batch_size * self.num_heads, query_length, key_length)
            beta = 1 / unscale

        attn_weights = torch.baddbmm(attn_weights, query, key, beta=beta, alpha=scale_factor).view(
            batch_size, self.num_heads, query_length, key_length
        )

        # ==========================================================================================
        # attn_weights -> (batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        if upcast:
            # Use a fused kernel to prevent a large overhead from casting and scaling.
            # Sub-optimal when the key length is not a multiple of 8.
            if attention_mask is None:
                attn_weights = upcast_softmax(attn_weights, unscale, softmax_dtype)
            else:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)
                attn_weights = upcast_masked_softmax(attn_weights, attention_mask, mask_value, unscale, softmax_dtype)
        else:
            if attention_mask is not None:
                mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)

                # The fused kernel is very slow when the key length is not a multiple of 8, so we skip fusion.
                attn_weights = torch.where(attention_mask, attn_weights, mask_value)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = self.attn_dropout(attn_weights)

        # ==========================================================================================
        # value -> (batch_size, num_heads, key_length, head_dim)
        # attn_weights -> (batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        # Mask heads if we want to
        if head_mask is not None:
            raise NotImplementedError()
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        # ==========================================================================================
        # attn_output -> (batch_size, num_heads, query_length, head_dim)
        # ==========================================================================================

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, -1, self.num_heads * self.head_dim)

        # ==========================================================================================
        # attn_output -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        return attn_output, attn_weights

    def _repeat_key_value(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_key_value_heads == 1:
            key = key.expand(-1, self.num_heads, -1, -1)
            value = value.expand(-1, self.num_heads, -1, -1)
        else:
            num_groups = self.num_heads // self.num_key_value_heads

            if num_groups != 1:
                key = key.repeat_interleave(num_groups, dim=1)
                value = value.repeat_interleave(num_groups, dim=1)

        return key, value



# Fused kernels
# Use separate functions for each case because conditionals prevent kernel fusion.
# TODO: Could have better fused kernels depending on scaling, dropout and head mask.
#  Is it doable without writing 32 functions?
@torch.jit.script
def upcast_masked_softmax(
    x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor, scale: float, softmax_dtype: torch.dtype
):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def upcast_softmax(x: torch.Tensor, scale: float, softmax_dtype: torch.dtype):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x
