# coding=utf-8
# Copyright 2022, Google and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch PEGASUS-X model."""

import dataclasses
import math
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...cache_utils import Cache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    auto_docstring,
    is_torch_flex_attn_available,
    is_torchdynamo_compiling,
    logging,
)
from .configuration_pegasus_x import PegasusXConfig


if is_torch_flex_attn_available():
    from ...integrations.flex_attention import BlockMask, make_flex_block_causal_mask


logger = logging.get_logger(__name__)


@dataclasses.dataclass
class DimensionInfo:
    """Wrapper for dimension info."""

    batch_size: int  # batch size
    seq_len: int  # token length
    block_size: int  # block size
    num_heads: int  # num heads
    hidden_dim: int  # hidden dim
    dim_per_head: int  # dim per head
    num_blocks: int  # num blocks
    global_len: int  # global length
    padded_seq_len: int  # padded token seq length

    # Note: Compared to the original Flax implementation, we will pad the token representations to
    #       a multiple of block size at the start of the encoder layers, so T=P always.


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# Copied from transformers.models.bart.modeling_bart.BartScaledWordEmbedding with Bart->PegasusX
class PegasusXScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


class PegasusXSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, embed_dim, max_scale: int = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_scale = max_scale

    @torch.no_grad()
    def forward(
        self, input_embeds: torch.Tensor, past_key_values_length: int = 0, position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        batch_size, seq_len = input_embeds.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=input_embeds.device
            )[:, None]

        pe = torch.zeros((seq_len, self.embed_dim), device=input_embeds.device, dtype=input_embeds.dtype)
        half_d_feature = self.embed_dim // 2
        div_term = torch.exp(
            torch.arange(half_d_feature, device=input_embeds.device, dtype=torch.int64).type_as(input_embeds)
            * -(np.log(float(self.max_scale)) / (half_d_feature - 1))
        )
        pe[:, :half_d_feature] = torch.sin(position_ids * div_term)
        pe[:, half_d_feature:] = torch.cos(position_ids * div_term)
        return pe[None].expand(batch_size, -1, -1)


# Copied from transformers.models.bart.modeling_bart.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask.view(1, -1, 1, 1)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->PegasusX
class PegasusXAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[PegasusXConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        # TODO: we need a refactor so that the different attention modules can get their specific kwargs
        # ATM, we have mixed things encoder, decoder, and encoder-decoder attn
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # determine input shapes
        bsz, tgt_len = hidden_states.shape[:-1]
        src_len = key_value_states.shape[1] if is_cross_attention else tgt_len

        q_input_shape = (bsz, tgt_len, -1, self.head_dim)
        kv_input_shape = (bsz, src_len, -1, self.head_dim)

        # get query proj
        query_states = self.q_proj(hidden_states).view(*q_input_shape).transpose(1, 2)

        if past_key_value is not None:
            if isinstance(past_key_value, EncoderDecoderCache):
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_states from cache
                    curr_past_key_value = past_key_value.cross_attention_cache
                else:
                    curr_past_key_value = past_key_value.self_attention_cache
            else:
                curr_past_key_value = past_key_value

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            key_states = key_states.view(*kv_input_shape).transpose(1, 2)
            value_states = value_states.view(*kv_input_shape).transpose(1, 2)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            output_attentions=output_attentions,
            head_mask=layer_head_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class PegasusXGlobalLocalAttention(nn.Module):
    """Global + Local attention. For use with Encoder only."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        dim = DimensionInfo(
            batch_size=token_hidden_states.shape[0],
            seq_len=token_hidden_states.shape[1],
            block_size=self.block_size,
            num_heads=self.num_heads,
            hidden_dim=token_hidden_states.shape[2],
            dim_per_head=self.head_dim,
            num_blocks=token_hidden_states.shape[1] // self.block_size,
            global_len=global_hidden_states.shape[1],
            padded_seq_len=token_hidden_states.shape[1],
        )

        # [batch_size, num_heads, padded_seq_len, dim_per_head]
        local_q = self._shape(
            self.q_proj(token_hidden_states) * self.scaling,
            seq_len=dim.padded_seq_len,
            bsz=dim.batch_size,
        )
        local_k = self._shape(
            self.k_proj(token_hidden_states),
            seq_len=dim.padded_seq_len,
            bsz=dim.batch_size,
        )
        local_v = self._shape(
            self.v_proj(token_hidden_states),
            seq_len=dim.padded_seq_len,
            bsz=dim.batch_size,
        )

        # [batch_size, num_heads, global_len, dim_per_head]
        global_q = self._shape(
            self.q_proj(global_hidden_states) * self.scaling,
            seq_len=dim.global_len,
            bsz=dim.batch_size,
        )
        global_k = self._shape(
            self.k_proj(global_hidden_states),
            seq_len=dim.global_len,
            bsz=dim.batch_size,
        )
        global_v = self._shape(
            self.v_proj(global_hidden_states),
            seq_len=dim.global_len,
            bsz=dim.batch_size,
        )

        global_attn_output, global_attn_probs = self.compute_global_attention_representations(
            global_q=global_q,
            global_k=global_k,
            global_v=global_v,
            local_k=local_k,
            local_v=local_v,
            mask=attention_mask,
            dim=dim,
        )
        local_attn_output, local_attn_probs = self.compute_local_attention_representations(
            global_k=global_k,
            global_v=global_v,
            local_q=local_q,
            local_k=local_k,
            local_v=local_v,
            mask=attention_mask,
            dim=dim,
        )

        # [batch_size, global_len, hidden_dim]
        global_attn_output = (
            global_attn_output.transpose(1, 2).contiguous().view(dim.batch_size, dim.global_len, dim.hidden_dim)
        )
        # [batch_size, global_len, hidden_dim]
        global_attn_output = self.out_proj(global_attn_output)
        # [batch_size, num_heads, block_size, num_heads, dim_per_head]
        local_attn_output = local_attn_output.permute(0, 2, 3, 1, 4).contiguous()
        # [batch_size, padded_seq_len, hidden_dim]
        local_attn_output = local_attn_output.view(dim.batch_size, dim.padded_seq_len, dim.hidden_dim)
        # [batch_size, padded_seq_len, hidden_dim]
        local_attn_output = self.out_proj(local_attn_output)

        if output_attentions:
            attn_probs = {"global": global_attn_probs, "local": local_attn_probs}
        else:
            attn_probs = None

        return local_attn_output, global_attn_output, attn_probs

    def compute_global_attention_representations(
        self, global_q, global_k, global_v, local_k, local_v, mask, dim: DimensionInfo
    ):
        """Compute attention representations for global tokens.

        Global tokens will attend to both global tokens as well as all input sequence tokens. Because the input
        sequence tokens are arranged in blocks for local attention, we unblock them and compute attention.

        Args:
            global_q (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                query vectors from global tokens
            global_k (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                key vectors from global tokens
            global_v (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                value vectors from global tokens
            local_k (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                key vectors from local tokens
            local_v (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                value vectors from local tokens
            mask (`torch.FloatTensor`) of shape [batch_size, padded_seq_len]: attention mask
            dim (DimensionInfo): DimensionInfo wrapper for dimensions

        Returns:
            output of shape `[batch_sizes, length, features]`. where length will be padded to a multiple of block_size
        """
        # [batch_size, num_heads, global_len+padded_seq_len, dim_per_head]
        global_and_local_k = torch.cat([global_k, local_k], dim=2)
        # [batch_size, num_heads, global_len+padded_seq_len, dim_per_head]
        global_and_local_v = torch.cat([global_v, local_v], dim=2)

        # [batch_size, global_len+padded_seq_len]
        extended_mask = nn.functional.pad(mask, pad=(dim.global_len, 0), value=0)

        # [batch_size, num_heads, global_len, global_len+padded_seq_len]
        attn_weights = torch.einsum("BHGF,BHXF->BHGX", global_q, global_and_local_k)
        attn_weights = attn_weights + extended_mask[:, None, None, :]
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)

        # [batch_size, num_heads, global_len, F]
        attn_output = torch.einsum("BHGX,BHXF->BHGF", attn_probs, global_and_local_v)
        return attn_output, attn_probs

    def compute_local_attention_representations(
        self, global_k, global_v, local_q, local_k, local_v, mask, dim: DimensionInfo
    ):
        """Compute attention representations for local tokens.

        Local tokens will attend to both global tokens as well as all other tokens within the same local block. Hence,
        we need to tile and concatenate the global tokens to every local block

        Args:
            global_k (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                key vectors from global tokens
            global_v (`torch.FloatTensor`) of shape [batch_size, num_heads, global_len, dim_per_head]:
                value vectors from global tokens
            local_q (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                query vectors from local tokens
            local_k (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                key vectors from local tokens
            local_v (`torch.FloatTensor`) of shape [batch_size, num_heads, padded_seq_len, dim_per_head]:
                value vectors from local tokens
            mask (`torch.FloatTensor`) of shape [batch_size, padded_seq_len]: attention mask
            dim (DimensionInfo): DimensionInfo wrapper for dimensions

        Returns:
            output of shape `[batch_sizes, length, features]`. where length will be padded to a multiple of block_size
        """
        # [batch_size, num_heads, num_blocks, block_size, dim_per_head]
        blocked_local_q = local_q.view(dim.batch_size, dim.num_heads, dim.num_blocks, dim.block_size, dim.dim_per_head)
        # [batch_size, num_heads, num_blocks, block_size, dim_per_head]
        blocked_local_k = local_k.view(dim.batch_size, dim.num_heads, dim.num_blocks, dim.block_size, dim.dim_per_head)
        # [batch_size, num_heads, num_blocks, block_size, dim_per_head]
        blocked_local_v = local_v.view(dim.batch_size, dim.num_heads, dim.num_blocks, dim.block_size, dim.dim_per_head)

        # [batch_size, num_blocks, global_len+block_size]
        extended_mask = nn.functional.pad(
            mask.view(dim.batch_size, dim.num_blocks, dim.block_size),
            pad=(dim.global_len, 0),
            value=0,
        )

        # [batch_size, num_heads, num_blocks, block_size, global_len]
        blocked_local2global = torch.einsum("BHNKF,BHGF->BHNKG", blocked_local_q, global_k)
        # [batch_size, num_heads, num_blocks, block_size, block_size]
        blocked_local2local = torch.einsum("BHNKF,BHNXF->BHNKX", blocked_local_q, blocked_local_k)

        # [batch_size, num_heads, num_blocks, block_size, global_len+block_size]
        attn_weights = torch.cat([blocked_local2global, blocked_local2local], dim=-1)
        attn_weights = attn_weights + extended_mask[:, None, :, None, :]
        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)

        # [batch_size, num_heads, num_blocks, block_size, global_len]
        local2global_attn_probs = attn_probs[:, :, :, :, : dim.global_len]
        # [batch_size, num_heads, num_blocks, block_size, block_size]
        local2local_attn_probs = attn_probs[:, :, :, :, dim.global_len :]

        # [batch_size, num_heads, num_blocks, block_size, dim_per_head]
        local2global_attn_output = torch.einsum("BHNKG,BHGF->BHNKF", local2global_attn_probs, global_v)
        # [batch_size, num_heads, num_blocks, block_size, dim_per_head]
        local2local_attn_output = torch.einsum("BHNKX,BHNXF->BHNKF", local2local_attn_probs, blocked_local_v)
        # [batch_size, num_heads, num_blocks, block_size, dim_per_head]
        attn_output = local2global_attn_output + local2local_attn_output
        return attn_output, attn_probs


class PegasusXEncoderLayer(nn.Module):
    def __init__(self, stagger_blocks_this_layer: bool, config: PegasusXConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = PegasusXGlobalLocalAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            block_size=config.block_size,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.global_self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.stagger_blocks_this_layer = stagger_blocks_this_layer
        self.block_size = config.block_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape *(seq_len, batch, embed_dim)*
            global_hidden_states (`torch.FloatTensor`): global token hidden states
                *(seq_len, num_global_tokens, embed_dim)*
            attention_mask (`torch.FloatTensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        global_residual = global_hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)
        global_hidden_states = self.global_self_attn_layer_norm(global_hidden_states)

        if self.stagger_blocks_this_layer:
            # Pad the blocks to simulate staggering
            hidden_states, attention_mask = self.pad_local_tokens(
                hidden_states=hidden_states, attention_mask=attention_mask, block_size=self.block_size
            )

        hidden_states, global_hidden_states, attn_weights = self.self_attn(
            token_hidden_states=hidden_states,
            global_hidden_states=global_hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        if self.stagger_blocks_this_layer:
            # Undo the padding
            hidden_states = self.unpad_local_tokens(padded_hidden_states=hidden_states, block_size=self.block_size)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        global_hidden_states = nn.functional.dropout(global_hidden_states, p=self.dropout, training=self.training)
        global_hidden_states = global_residual + global_hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        global_residual = global_hidden_states
        global_hidden_states = self.final_layer_norm(global_hidden_states)
        global_hidden_states = self.activation_fn(self.fc1(global_hidden_states))
        global_hidden_states = nn.functional.dropout(
            global_hidden_states, p=self.activation_dropout, training=self.training
        )
        global_hidden_states = self.fc2(global_hidden_states)
        global_hidden_states = nn.functional.dropout(global_hidden_states, p=self.dropout, training=self.training)
        global_hidden_states = global_residual + global_hidden_states
        outputs = (hidden_states, global_hidden_states)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def pad_local_tokens(cls, hidden_states, attention_mask, block_size):
        # hidden_states: [batch_size, seq_len, hidden_dim]
        pad_size = block_size // 2
        mask_min_value = torch.finfo(hidden_states.dtype).min
        padded_hidden_states = torch.nn.functional.pad(
            hidden_states,
            pad=(0, 0, pad_size, pad_size),
        )
        padded_mask = torch.nn.functional.pad(
            attention_mask,
            pad=(pad_size, pad_size),
            value=mask_min_value,
        )
        return padded_hidden_states, padded_mask

    @classmethod
    def unpad_local_tokens(cls, padded_hidden_states, block_size):
        # padded_hidden_states: [batch_size, padded seq_len, hidden_dim]
        pad_size = block_size // 2
        return padded_hidden_states[:, pad_size:-pad_size, :]


class PegasusXDecoderLayer(nn.Module):
    def __init__(self, config: PegasusXConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = PegasusXAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=False,
            config=config,
            layer_idx=layer_idx,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = PegasusXAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=False,
            config=config,
            layer_idx=layer_idx,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape *(seq_len, batch, embed_dim)*
            attention_mask (`torch.FloatTensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape *(seq_len, batch, embed_dim)*
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache: Whether to us KV cache for decoding
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. It is used to update the
                cache in the correct position and to infer the complete sequence length.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, past_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            hidden_states, cross_attn_weights, past_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (past_key_value,)

        return outputs


@auto_docstring
class PegasusXPreTrainedModel(PreTrainedModel):
    config_class = PegasusXConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [r"PegasusXEncoderLayer", r"PegasusXDecoderLayer"]
    _supports_flash_attn_2 = True
    # Flaky logits
    _supports_sdpa = False
    # Compile issues
    _supports_flex_attn = False
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_full_mask
    def _update_full_mask(
        self,
        attention_mask: Union[torch.Tensor, None],
        inputs_embeds: torch.Tensor,
    ):
        if attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask if 0 in attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = make_flex_block_causal_mask(attention_mask, is_causal=False)
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        return attention_mask

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: Optional[Union[torch.Tensor, "BlockMask"]],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
    ):
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            # Other attention flavors support in-built causal (when `mask is None`)
            # while we need to create our specific block mask regardless
            elif attention_mask is None:
                attention_mask = make_flex_block_causal_mask(
                    torch.ones(
                        size=(input_tensor.shape[0], input_tensor.shape[1]),
                        device=attention_mask.device,
                    )
                )
            return attention_mask

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
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
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
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
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    # Copied from transformers.models.bart.modeling_bart.BartPreTrainedModel._update_cross_attn_mask
    def _update_cross_attn_mask(
        self,
        encoder_hidden_states: Union[torch.Tensor, None],
        encoder_attention_mask: Union[torch.Tensor, None],
        input_shape: torch.Size,
        inputs_embeds: torch.Tensor,
    ):
        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self.config._attn_implementation == "flash_attention_2":
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            elif self.config._attn_implementation == "sdpa":
                # output_attentions=True & cross_attn_head_mask can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            elif self.config._attn_implementation == "flex_attention":
                if isinstance(encoder_attention_mask, torch.Tensor):
                    encoder_attention_mask = make_flex_block_causal_mask(
                        encoder_attention_mask,
                        query_length=input_shape[-1],
                        is_causal=False,
                    )
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        return encoder_attention_mask


class PegasusXEncoder(PegasusXPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`PegasusXEncoderLayer`].

    Args:
        config: PegasusXConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: PegasusXConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = PegasusXScaledWordEmbedding(
                config.vocab_size, embed_dim, padding_idx, embed_scale=embed_scale
            )

        self.embed_global = nn.Embedding(config.num_global_tokens, embed_dim)
        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(embed_dim)
        self.layers = nn.ModuleList(
            [
                PegasusXEncoderLayer(
                    stagger_blocks_this_layer=i % 2 == 1 and config.stagger_local_blocks, config=config
                )
                for i in range(config.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings. If position embeddings are learned, increasing the size will add
                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If
                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(self.config.d_model)
        self.embed_positions.to(self.device)

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings matrix
        """
        return self.embed_positions

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(inputs_embeds)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        batch_size, seq_len, _ = hidden_states.shape

        # Setup mask
        if attention_mask is None:
            attention_mask = torch.ones(*input_shape, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        mask_min_value = torch.finfo(hidden_states.dtype).min
        inverted_mask = 1.0 - attention_mask
        attention_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool),
            mask_min_value,
        )

        # padding to block_size
        if seq_len % self.config.block_size != 0:
            pad_len = self.config.block_size - seq_len % self.config.block_size
            hidden_states = nn.functional.pad(hidden_states, pad=(0, 0, 0, pad_len), value=0)
            attention_mask = nn.functional.pad(attention_mask, pad=(0, pad_len), value=mask_min_value)

        # Global tokens
        global_hidden_states = self.embed_global(
            torch.arange(self.config.num_global_tokens, device=hidden_states.device)[None].expand(batch_size, -1)
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        global_hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        global_hidden_states,
                        attention_mask,
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                global_hidden_states = layer_outputs[1]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)

        # Undo padding-to-block-size
        hidden_states = hidden_states[:, :seq_len]

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + ((hidden_states, global_hidden_states),)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class PegasusXDecoder(PegasusXPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`PegasusDecoderLayer`]

    Args:
        config: PegasusXConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: PegasusXConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        padding_idx = config.pad_token_id

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = PegasusXScaledWordEmbedding(
                config.vocab_size, config.d_model, padding_idx=padding_idx, embed_scale=embed_scale
            )

        self.embed_positions = PegasusXSinusoidalPositionalEmbedding(config.d_model)
        self.layers = nn.ModuleList([PegasusXDecoderLayer(config, layer_idx=i) for i in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence. It is used to update the
                cache in the correct position and to infer the complete sequence length.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        # initialize `past_key_values`
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            logger.warning_once(
                "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.58.0. "
                "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
            )
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        batch_size, seq_length = inputs_embeds.size()[:-1]
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        self_attn_cache = (
            past_key_values.self_attention_cache
            if isinstance(past_key_values, EncoderDecoderCache)
            else past_key_values
        )

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            self_attn_cache,
        )
        encoder_attention_mask = self._update_cross_attn_mask(
            encoder_hidden_states,
            encoder_attention_mask,
            input_shape,
            inputs_embeds,
        )

        # embed positions
        position_ids = cache_position.unsqueeze(1)
        position_ids = self.embed_positions(inputs_embeds, past_key_values_length, position_ids)
        position_ids = position_ids.to(inputs_embeds.device)
        hidden_states = inputs_embeds + position_ids
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[3 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@auto_docstring
class PegasusXModel(PegasusXPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: PegasusXConfig):
        super().__init__(config)

        vocab_size = config.vocab_size
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        padding_idx = config.pad_token_id
        self.shared = PegasusXScaledWordEmbedding(
            vocab_size, config.d_model, padding_idx=padding_idx, embed_scale=embed_scale
        )

        self.encoder = PegasusXEncoder(config, self.shared)
        self.decoder = PegasusXDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings. If position embeddings are learned, increasing the size will add
                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If
                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        self.config.max_position_embeddings = new_num_position_embeddings
        self.encoder.resize_position_embeddings(new_num_position_embeddings)
        self.decoder.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        Returns the position embeddings matrix
        """
        return (self.encoder.get_position_embeddings(), self.decoder.get_position_embeddings())

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            PEGASUS-X uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, PegasusModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-large")
        >>> model = PegasusModel.from_pretrained("google/pegasus-x-large")

        >>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
        >>> decoder_inputs = tokenizer("Studies show that", return_tensors="pt")
        >>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 4, 1024]
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The PEGASUS-X for conditional generation (e.g. summarization).
    """
)
class PegasusXForConditionalGeneration(PegasusXPreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: PegasusXConfig):
        super().__init__(config)
        self.model = PegasusXModel(config)
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings. If position embeddings are learned, increasing the size will add
                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If
                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        self.config.max_position_embeddings = new_num_position_embeddings
        self.model.encoder.resize_position_embeddings(new_num_position_embeddings)
        self.model.decoder.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        Returns the position embeddings matrix
        """
        return (self.model.encoder.get_position_embeddings(), self.model.decoder.get_position_embeddings())

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            PEGASUS-X uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


# Copied from transformers.models.bart.modeling_bart.BartDecoderWrapper with Bart->PegasusX
class PegasusXDecoderWrapper(PegasusXPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        self.decoder = PegasusXDecoder(config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


__all__ = ["PegasusXForConditionalGeneration", "PegasusXModel", "PegasusXPreTrainedModel"]
