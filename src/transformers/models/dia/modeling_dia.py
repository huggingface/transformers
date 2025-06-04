# coding=utf-8
# Copyright 2025 The Nari Labs and HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Dia model."""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
)
from ...generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateOutput, GenerationConfig, GenerationMixin
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...tokenization_utils_base import PreTrainedTokenizerBase
from ...utils import (
    is_torch_flex_attn_available,
    is_torchdynamo_compiling,
    logging,
)
from .configuration_dia import DiaConfig, DiaDecoderConfig, DiaEncoderConfig


if is_torch_flex_attn_available():
    from ...integrations.flex_attention import BlockMask, make_flex_block_causal_mask


logger = logging.get_logger(__name__)


# TODO: temporarily for debugging
debug = True


class DiaPreTrainedModel(PreTrainedModel):
    config_class = DiaConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"  # TODO: change this?
    supports_gradient_checkpointing = True
    _no_split_modules = ["DiaEncoderLayer", "DiaDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = getattr(self.config, "init_std", 0.2)
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    # TODO: new masking for causal fails atm (create_causal_masks)

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


class DiaMultiChannelEmbedding(nn.Module):
    """In order to efficiently compute the audio embedding from the 9 different channels,
    we vectorize the embedding process by using a single embedding layer and an offset.
    Example:
    - num_embeds = 4
    - vocab_size = 8
    - num_channels = 3
    We would have offsets = [0, 8, 16]
    If audio_codes = [0, 1, 2, 3], [1, 3, 4, 7], [5, 6, 7, 8],
    then tokens = audio_codes + offsets
                = [0, 1, 2, 3, 9, 11, 12, 15, 21, 22, 23, 24]
    This allows us to use a single embedding layer for all channels.
    """

    def __init__(self, config: DiaDecoderConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size * config.num_channels, config.hidden_size)
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        offsets = torch.arange(config.num_channels, dtype=torch.long) * config.vocab_size  # (C,)
        self.register_buffer("offsets", offsets, persistent=False)

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        tokens = (audio_codes + self.offsets.to(audio_codes.device)).squeeze(1)
        embeds = self.embed(tokens).view(tokens.shape[0], audio_codes.shape[1], -1, self.hidden_size)
        return embeds.sum(dim=2)


# Copied from transformers.models.phi3.modular_phi3.Phi3MLP with Phi3->Dia
class DiaMLP(nn.Module):  # Modular GlmMLP
    def __init__(self, config: DiaEncoderConfig | DiaDecoderConfig):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Dia
class DiaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def apply_rotary_pos_emb(
    tensor: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], unsqueeze_dim: int = 1
) -> torch.Tensor:
    cos = position_embeddings[0]
    sin = position_embeddings[1]
    first_half, second_half = torch.chunk(tensor.to(torch.float32), 2, dim=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    return torch.cat((first_part.to(tensor.dtype), second_part.to(tensor.dtype)), dim=-1)


# TODO: refactor RoPE to transformers rope?
class DiaRotaryEmbedding(nn.Module):
    def __init__(self, config: Union[DiaEncoderConfig, DiaDecoderConfig], device: Optional[torch.device] = None):
        super().__init__()
        self.embedding_dims = config.head_dim
        self.min_timescale = config.rope_min_timescale
        self.max_timescale = config.rope_max_timescale

        half_embedding_dim = self.embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / self.embedding_dims
        freqs = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: this should not be needed when using proper position ids
        if position_ids.ndim == 1:
            # Ensure position_ids is at least 2D, e.g., (1, seq_len)
            position_ids = position_ids.unsqueeze(0)

        position_ids_expanded = position_ids[:, :, None, None].float().repeat(x.shape[0], 1, 1, 1)
        full_freqs = position_ids_expanded.float() / self.freqs.to(
            device=position_ids_expanded.device, dtype=position_ids_expanded.dtype
        )
        cos, sin = full_freqs.cos(), full_freqs.sin()
        return cos, sin


# Copied from transformers.models.llama.modeling_llama.repeat_kv
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


def eager_attention_forward(
    module: Union["DiaSelfAttention", "DiaCrossAttention"],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float],
    **kwargs,
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class DiaSelfAttention(nn.Module):  # Modular : LlamaAttentions
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Union[DiaEncoderConfig, DiaDecoderConfig], layer_idx: int, is_causal: bool = False):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.layer_idx = layer_idx
        self.scaling = 1
        self.is_causal = is_causal
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[EncoderDecoderCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = apply_rotary_pos_emb(query_states, position_embeddings, -2).transpose(1, 2)
        key_states = apply_rotary_pos_emb(key_states, position_embeddings, -2).transpose(1, 2)

        if past_key_values is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            key_states, value_states = past_key_values.self_attention_cache.update(
                key_states, value_states, self.layer_idx, {"cache_positions": cache_position}
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiaCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DiaDecoderConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.num_heads = self.config.cross_num_attention_heads
        self.num_key_value_heads = self.config.cross_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.cross_hidden_size = config.cross_hidden_size
        self.head_dim = config.cross_head_dim
        self.layer_idx = layer_idx
        self.scaling = 1
        self.is_causal = False
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.cross_hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.cross_hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cross_shape = (*cross_attention_states.shape[:-1], -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        is_updated = past_key_values.is_updated.get(self.layer_idx) if past_key_values is not None else False
        if past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_values.cross_attention_cache.key_cache[self.layer_idx]
            value_states = past_key_values.cross_attention_cache.value_cache[self.layer_idx]
        else:
            key_states = self.k_proj(cross_attention_states).view(cross_shape).transpose(1, 2)
            value_states = self.v_proj(cross_attention_states).view(cross_shape).transpose(1, 2)

            if past_key_values is not None:
                # save all states to the cache
                key_states, value_states = past_key_values.cross_attention_cache.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                past_key_values.is_updated[self.layer_idx] = True

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape((*input_shape, -1)).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiaEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DiaEncoderConfig, layer_idx: int):
        super().__init__()
        self.pre_sa_norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attention = DiaSelfAttention(config, layer_idx, is_causal=False)
        self.post_sa_norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = DiaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        normed_states = self.pre_sa_norm(hidden_states)

        # TODO: are kwargs possible with correct gradient passing
        hidden_states, self_attn_weights = self.self_attention(
            normed_states,
            attention_mask,
            position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        normed_states = self.post_sa_norm(hidden_states)
        mlp_out = self.mlp(normed_states)
        hidden_states = residual + mlp_out

        return hidden_states, self_attn_weights


class DiaEncoder(DiaPreTrainedModel):
    def __init__(self, config: DiaEncoderConfig):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DiaEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.rotary_embeddings = DiaRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[BaseModelOutput, Tuple]:
        hidden_states = self.embedding(input_ids)

        # RoPE
        # Note: We expect right padding and hence always generate
        # the position ids on the fly to reduce preparation overhead
        position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device)[None, :]
        position_embeddings = self.rotary_embeddings(hidden_states, position_ids)

        attention_mask = self._update_full_mask(
            attention_mask,
            hidden_states,
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # TODO: check kwargs
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            encoder_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class DiaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DiaDecoderConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attention = DiaSelfAttention(config, layer_idx, is_causal=True)
        self.cross_attention = DiaCrossAttention(config, layer_idx)
        self.pre_sa_norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.pre_ca_norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.pre_mlp_norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = DiaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        normed_states = self.pre_sa_norm(hidden_states)

        # TODO: are kwargs possible with correct gradient passing
        hidden_states, self_attn_weights = self.self_attention(
            normed_states,
            attention_mask,
            position_embeddings,
            past_key_values,
            cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_ca_norm(hidden_states)
        # TODO: are kwargs possible with correct gradient passing
        cross_states, cross_attn_weights = self.cross_attention(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            **kwargs,
        )
        hidden_states = residual + cross_states

        residual = hidden_states
        x_norm = self.pre_mlp_norm(hidden_states)
        mlp_out = self.mlp(x_norm)
        hidden_states = residual + mlp_out

        return hidden_states, self_attn_weights, cross_attn_weights


class DiaDecoder(DiaPreTrainedModel):
    """Transformer Decoder Stack using DenseGeneral."""

    def __init__(self, config: DiaDecoderConfig):
        super().__init__(config)
        self.num_channels = config.num_channels
        self.vocab_size = config.vocab_size
        self.embeddings = DiaMultiChannelEmbedding(config)
        self.rotary_embeddings = DiaRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [DiaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DiaRMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]:
        batch_size, seq_length = input_ids.size()[:-1]
        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=self.device
            )
        if position_ids is None:
            position_ids = cache_position[None, :]

        # RoPE
        hidden_states = self.embeddings(input_ids)
        position_embeddings = self.rotary_embeddings(hidden_states, position_ids)

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=input_ids.device)

        self_attn_cache = (
            past_key_values.self_attention_cache
            if isinstance(past_key_values, EncoderDecoderCache)
            else past_key_values
        )

        attention_mask = self._update_causal_mask(
            attention_mask,
            hidden_states,
            cache_position,
            self_attn_cache,
        )
        encoder_attention_mask = self._update_cross_attn_mask(
            encoder_hidden_states,
            encoder_attention_mask,
            hidden_states.shape[:2],
            hidden_states,
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # TODO: check kwargs
            layer_outputs = layer(
                hidden_states,
                attention_mask,
                position_embeddings,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class DiaModel(DiaPreTrainedModel):
    def __init__(self, config: DiaConfig):
        super().__init__(config)
        self.config = config
        self.encoder = DiaEncoder(config.encoder_config)
        self.decoder = DiaDecoder(config.decoder_config)
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Union[BaseModelOutput, Tuple]] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        if input_ids is None and encoder_outputs is None:
            raise ValueError(
                "You should either provide text ids or the cached text encodings. Neither has been found."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.is_gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        global debug
        if debug:
            if input_ids is not None:
                input_ids = input_ids[:, None, :]
                unconditioned_input_ids = torch.zeros_like(input_ids)
                input_ids = torch.stack([unconditioned_input_ids, input_ids], dim=1).view(-1, input_ids.shape[-1])

            if attention_mask is not None:
                attention_mask = attention_mask.repeat_interleave(2, dim=0)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
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

        if debug:
            if decoder_input_ids is None:
                # (2*bsz, 1, channel)
                decoder_input_ids = torch.full((encoder_outputs[0].shape[0], 1, 9), 1026, device=self.device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,  # TODO: if we prefix audio we will need a mask when left padding - `audio_attention_mask`
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
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
            encoder_last_hidden_state=encoder_outputs[0],
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class DiaClassifierFreeGuidanceFilterLogitsProcessor(LogitsProcessor):
    def __init__(self, cfg_scale: float = 3.0, cfg_filter_top_k: int = 50, device: str = "cpu"):
        self.cfg_scale = torch.tensor(cfg_scale, device=device)
        self.cfg_filter_top_k = cfg_filter_top_k

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] != 2 * input_ids.shape[0]:
            raise ValueError(
                f"Logits should have twice the batch size of the input ids, the first half of batches corresponding to "
                f"the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got "
                f"batch size {scores.shape[0]} for the logits and {input_ids.shape[0]} for the input ids."
            )
        # TODO: reshape from (B * C, V) to (B, C, V)

        # cfg
        scores_last = scores.view(scores.shape[0] // 2, 2, *scores.shape[1:])
        uncond_scores = scores_last[:, 0, :]
        cond_scores = scores_last[:, 1, :]
        scores = cond_scores + self.cfg_scale * (cond_scores - uncond_scores)  # Shape [B_orig, C, V]

        # cfg filter top k
        _, top_k_indices = torch.topk(scores, k=self.cfg_filter_top_k, dim=-1)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask = mask.scatter(dim=-1, index=top_k_indices, value=False)
        scores = cond_scores.masked_fill(mask, -torch.inf)

        return scores


class DiaEOSFilterAndScaleLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_value: int, eos_scale: float, device: str = "cpu"):
        self.eos_value = eos_value
        self.eos_scale = torch.tensor(eos_scale, device=device) if eos_scale != 1.0 else None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        EOS filter, this ensures:
        # 1. only channel 0 can generate EOS
        # 2. if channel 0 has EOS with highest logit, it will be the only candidate
        # 3. if channel 0 has EOS not with highest logit, it will be suppressed
        """
        # TODO: reshape from (B * C, V) to (B, C, V)

        scores[:, 1:, self.eos_value :] = torch.full_like(
            scores[:, 1:, self.eos_value :],
            fill_value=-torch.inf,
        )
        if self.eos_scale is not None:
            scores[:, 0, self.eos_value] *= self.eos_scale

        scores_flat = scores.view(-1, scores.shape[-1])

        top_logit_indices = torch.argmax(scores_flat, dim=-1)
        eos_not_highest_mask = top_logit_indices != self.eos_value
        mask_eos_unless_highest = torch.zeros_like(scores_flat, dtype=torch.bool)
        mask_eos_unless_highest[eos_not_highest_mask, self.eos_value] = True
        scores_flat = scores_flat.masked_fill(mask_eos_unless_highest, -torch.inf)
        eos_highest_mask = top_logit_indices == self.eos_value
        mask_eos_highest = torch.zeros_like(scores_flat, dtype=torch.bool)
        mask_eos_highest[eos_highest_mask, : self.eos_value] = True
        scores_flat = scores_flat.masked_fill(mask_eos_highest, -torch.inf)

        scores = scores_flat.view(scores.shape)

        return scores


class DiaEOSDelayPatternLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        delay_pattern: torch.Tensor,
        eos_value: int,
        pad_value: int,
        max_step: int,
        device: str = "cpu",
    ):
        self.delay_pattern = delay_pattern
        self.max_delay_pattern = delay_pattern.max().item()
        self.num_channels = delay_pattern.shape[0]
        self.eos_value = eos_value
        self.pad_value = pad_value
        self.max_step = torch.tensor(max_step, device=device)
        self.eos_countup: Optional[torch.Tensor] = None
        self.device = device

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        This logits processor is used to apply the delay pattern to the generated tokens when EOS is generated.

        If delay pattern is [0, 2, 3, 4] then:

            s   s+1 s+2 s+3 s+4 s+5 ...
            |   |   |   |   |   |
        C0: EOS PAD PAD PAD PAD PAD ...
        C1: x   x   EOS PAD PAD PAD ...
        C2: x   x   x   EOS PAD PAD ...
        C3: x   x   x   x   EOS PAD ...

        The PAD & EOS are forced from step s+1.
        """
        # TODO: reshape from (B * C, V) to (B, C, V)
        # and reshape from (B * C, S) to (B, S, C)

        # EOS Countup
        # Due to delay pattern, we do not stop generation at the first EOS token.
        # Instead, we force EOS, PAD at delay pattern steps.
        batch_size = scores.shape[0]
        if self.eos_countup is None:
            self.eos_countup = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

        step = input_ids.shape[1]

        # EOS countdown and delay pattern application
        active_mask = self.eos_countup > 0

        if active_mask.any():
            # Logits for active items: [num_active, C, V]
            scores_active = scores[active_mask]
            # Countdown values for active items: [num_active]
            eos_countup_active = self.eos_countup[active_mask]

            # Expand for comparison with delay_pattern: [num_active, C]
            eos_countup_active = eos_countup_active.unsqueeze(1).expand(-1, self.num_channels)
            delay_pattern = self.delay_pattern.unsqueeze(0).expand(scores_active.shape[0], -1)  # [num_active, C]

            # Mask for forcing EOS: [num_active, C]
            force_eos_mask_BxC = eos_countup_active == delay_pattern
            # Mask for forcing PAD: [num_active, C]
            force_pad_mask_BxC = eos_countup_active > delay_pattern

            # Efficiently apply forced EOS and PAD logits
            vocab_size = scores_active.shape[-1]

            # Create template rows for forced EOS and PAD
            eos_row = torch.full((vocab_size,), -torch.inf, device=self.device, dtype=scores_active.dtype)
            eos_row[self.eos_value] = 0.0
            pad_row = torch.full((vocab_size,), -torch.inf, device=self.device, dtype=scores_active.dtype)
            pad_row[self.pad_value] = 0.0

            # Clone the active slice to modify it
            final_modified_slice = scores_active.clone()
            final_modified_slice[force_eos_mask_BxC] = eos_row
            final_modified_slice[force_pad_mask_BxC] = pad_row

            # Update the original logits tensor with the modified slice
            scores[active_mask] = final_modified_slice

        # This is possible because we applied `DiaEOSFilterAndScaleLogitsProcessor`
        last_generated_tokens = torch.argmax(scores, dim=-1)[:, 0]  # Shape [B_orig]
        eos_start_mask = last_generated_tokens == self.eos_value
        eos_start_mask |= step + self.max_delay_pattern >= self.max_step
        eos_start_mask &= self.eos_countup < 0

        # Make sure that the EOS token is the only candidate for the first token
        scores[eos_start_mask, 0, :] = -torch.inf
        scores[eos_start_mask, 0, self.eos_value] = 0.0

        self.eos_countup[eos_start_mask] = 0
        self.eos_countup[self.eos_countup >= 0] += 1

        return scores


class DiaEosTokenCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
    """

    def __init__(self, eos_value: int, delay_pattern: torch.Tensor, device: str = "cpu"):
        self.eos_token_id = torch.tensor(eos_value, device=device)
        self.max_delay_pattern = delay_pattern.max().item()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        """
        This stopping criteria is used to stop generation when EOS is generated.

        If delay pattern is [0, 2, 3, 4] then:

            s   s+1 s+2 s+3 s+4 s+5 ...
            |   |   |   |   |   |
        C0: EOS PAD PAD PAD PAD PAD ...
        C1: x   x   EOS PAD PAD PAD ...
        C2: x   x   x   EOS PAD PAD ...
        C3: x   x   x   x   EOS PAD ...

        We need to stop generation in step s+3, where all of the information is generated.
        We check by if the first channel has EOS in the `step - max_delay_pattern + 1` step.
        """
        # TODO: reshape from (B * C, V) to (B, C, V)
        # and reshape from (B * C, S) to (B, S, C)

        if input_ids.shape[1] < self.max_delay_pattern:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        is_done = input_ids[:, -self.max_delay_pattern + 1, 0] == self.eos_token_id
        return is_done


def build_delay_indices(B: int, T: int, C: int, delay_pattern: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute (t_idx_BxTxC, indices_BTCx3) so that out[t, c] = in[t - delay[c], c].
    Negative t_idx => BOS; t_idx >= T => PAD.
    """
    device = delay_pattern.device

    t_idx_BxT = torch.broadcast_to(torch.arange(T, dtype=torch.long, device=device)[None, :], [B, T])
    t_idx_BxTx1 = t_idx_BxT[..., None]
    t_idx_BxTxC = t_idx_BxTx1 - delay_pattern.view(1, 1, C)

    b_idx_BxTxC = torch.broadcast_to(torch.arange(B, dtype=torch.long, device=device).view(B, 1, 1), [B, T, C])
    c_idx_BxTxC = torch.broadcast_to(torch.arange(C, dtype=torch.long, device=device).view(1, 1, C), [B, T, C])

    # We must clamp time indices to [0..T-1] so gather_nd equivalent won't fail
    t_clamped_BxTxC = torch.clamp(t_idx_BxTxC, 0, T - 1)

    indices_BTCx3 = torch.stack([b_idx_BxTxC.reshape(-1), t_clamped_BxTxC.reshape(-1), c_idx_BxTxC.reshape(-1)], dim=1)

    return t_idx_BxTxC, indices_BTCx3


def apply_audio_delay(
    audio: torch.Tensor,
    pad_value: int,
    bos_value: int,
    precomp: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Applies the delay pattern to batched audio tokens using precomputed indices,
    inserting BOS where t_idx < 0 and PAD where t_idx >= T.

    Args:
        audio: [B, T, C] int16 audio tokens (or int32/float)
        pad_value: the padding token
        bos_value: the BOS token
        precomp:  (t_idx_BxTxC, indices_BTCx3) from build_delay_indices

    Returns:
        result: [B, T, C] delayed audio tokens
    """
    device = audio.device  # Get device from input tensor
    t_idx_BxTxC, indices_BTCx3 = precomp
    t_idx_BxTxC = t_idx_BxTxC.to(device)  # Move precomputed indices to device
    indices_BTCx3 = indices_BTCx3.to(device)

    # Equivalent of tf.gather_nd using advanced indexing
    # Ensure indices are long type if not already (build_delay_indices should handle this)
    gathered_flat = audio[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio.shape)

    # Create masks on the correct device
    mask_bos = t_idx_BxTxC < 0  # => place bos_value
    mask_pad = t_idx_BxTxC >= audio.shape[1]  # => place pad_value

    # Create scalar tensors on the correct device
    bos_tensor = torch.tensor(bos_value, dtype=audio.dtype, device=device)
    pad_tensor = torch.tensor(pad_value, dtype=audio.dtype, device=device)

    # If mask_bos, BOS; else if mask_pad, PAD; else original gather
    # All tensors should now be on the same device
    result = torch.where(mask_bos, bos_tensor, torch.where(mask_pad, pad_tensor, gathered_BxTxC))

    return result


def build_revert_indices(B: int, T: int, C: int, delay_pattern: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute indices for the revert operation using PyTorch.

    Returns:
        A tuple (t_idx_BxTxC, indices_BTCx3) where:
            - t_idx_BxTxC is a tensor of shape [B, T, C] computed as time indices plus the delay.
            - indices_BTCx3 is a tensor of shape [B*T*C, 3] used for gathering, computed from:
                batch indices, clamped time indices, and channel indices.
    """
    # Use default device unless specified otherwise; assumes inputs might define device later
    device = delay_pattern.device

    t_idx_BT1 = torch.broadcast_to(torch.arange(T, dtype=torch.long, device=device).unsqueeze(0), [B, T])
    t_idx_BT1 = t_idx_BT1.unsqueeze(-1)

    t_idx_BxTxC = torch.minimum(
        t_idx_BT1 + delay_pattern.view(1, 1, C), torch.tensor(T - 1, dtype=torch.long, device=device)
    )
    b_idx_BxTxC = torch.broadcast_to(torch.arange(B, dtype=torch.long, device=device).view(B, 1, 1), [B, T, C])
    c_idx_BxTxC = torch.broadcast_to(torch.arange(C, dtype=torch.long, device=device).view(1, 1, C), [B, T, C])

    indices_BTCx3 = torch.stack([b_idx_BxTxC.reshape(-1), t_idx_BxTxC.reshape(-1), c_idx_BxTxC.reshape(-1)], dim=1)

    return t_idx_BxTxC, indices_BTCx3


def revert_audio_delay(
    audio_BxTxC: torch.Tensor,
    pad_value: int,
    precomp: Tuple[torch.Tensor, torch.Tensor],
    T: int,
) -> torch.Tensor:
    """
    Reverts a delay pattern from batched audio tokens using precomputed indices (PyTorch version).

    Args:
        audio_BxTxC: Input delayed audio tensor
        pad_value: Padding value for out-of-bounds indices
        precomp: Precomputed revert indices tuple containing:
            - t_idx_BxTxC: Time offset indices tensor
            - indices_BTCx3: Gather indices tensor for original audio
        T: Original sequence length before padding

    Returns:
        Reverted audio tensor with same shape as input
    """
    t_idx_BxTxC, indices_BTCx3 = precomp
    device = audio_BxTxC.device  # Get device from input tensor

    # Move precomputed indices to the same device as audio_BxTxC if they aren't already
    t_idx_BxTxC = t_idx_BxTxC.to(device)
    indices_BTCx3 = indices_BTCx3.to(device)

    # Using PyTorch advanced indexing (equivalent to tf.gather_nd or np equivalent)
    gathered_flat = audio_BxTxC[indices_BTCx3[:, 0], indices_BTCx3[:, 1], indices_BTCx3[:, 2]]
    gathered_BxTxC = gathered_flat.view(audio_BxTxC.size())  # Use .size() for robust reshaping

    # Create pad_tensor on the correct device
    pad_tensor = torch.tensor(pad_value, dtype=audio_BxTxC.dtype, device=device)
    # Create T tensor on the correct device for comparison
    T_tensor = torch.tensor(T, device=device)
    result_BxTxC = torch.where(t_idx_BxTxC >= T_tensor, pad_tensor, gathered_BxTxC)  # Changed np.where to torch.where

    return result_BxTxC


class DiaForConditionalGeneration(DiaPreTrainedModel, GenerationMixin):
    base_model_prefix = "model"

    def __init__(self, config: DiaConfig):
        super().__init__(config)
        self.config = config
        self.model = DiaModel(config)

        self.num_channels = config.decoder_config.num_channels
        self.vocab_size = config.decoder_config.vocab_size
        self.logits_dense = nn.Linear(
            config.decoder_config.hidden_size, (self.num_channels * self.vocab_size), bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Union[BaseModelOutput, Tuple]] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        last_hidden_state = outputs[0]
        batch_size = last_hidden_state.shape[0]
        audio_logits = self.logits_dense(last_hidden_state).view(batch_size, -1, self.num_channels, self.vocab_size)

        # TODO: loss calculations here
        loss = None

        if not return_dict:
            output = (audio_logits,) + outputs[1:]
            # loss
            return output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=audio_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    # TODO: seems to expect static cache --> we shouldn't have to do anything in this method as the user needs to provide correct values with processor
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        _encoder_length = self.config.encoder_config.max_length
        _encoder_pad = 0

        # We need to pop input_ids from model_kwargs to avoid passing it to the encoder
        input_ids: Optional[torch.Tensor] = model_kwargs.pop("input_ids", None)

        if inputs is not None and input_ids is not None:
            raise ValueError("inputs and input_ids cannot be provided at the same time")
        if input_ids is None and inputs is None:
            raise ValueError("inputs must be provided")
        if input_ids is None:
            input_ids = inputs

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be a 1D or 2D tensor")
        if input_ids.shape[-1] == _encoder_length:
            raise ValueError(f"input_ids length must be {_encoder_length} for generation")

        batch_size = input_ids.shape[0]

        # attention mask
        # Batch size is doubled for unconditioned input
        model_kwargs["attention_mask"] = (input_ids != _encoder_pad).to(self.device).repeat_interleave(2, dim=0)
        model_kwargs["decoder_attention_mask"] = torch.ones((2 * batch_size, 1), dtype=torch.bool, device=self.device)

        return input_ids, "input_ids", model_kwargs

    # TODO: accept 3D and reshape to 2D after processing everything
    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        if device is None:
            device = self.device

        decoder_input_ids = model_kwargs.pop("decoder_input_ids", None)
        decoder_attention_mask = model_kwargs["decoder_attention_mask"]
        decoder_input_ids_length = decoder_input_ids.shape[1] if decoder_input_ids is not None else 0

        delay_pattern = torch.tensor(self.config.delay_pattern, device=device, dtype=torch.long)
        max_delay_pattern = delay_pattern.max().item()

        input_ids = torch.full(
            (batch_size, 1 + decoder_input_ids_length + max_delay_pattern, self.config.decoder_config.num_channels),
            self.config.pad_token_id,
            dtype=torch.long,
            device=device,
        )

        input_ids[:, 0, :] = self.config.bos_token_id
        if decoder_input_ids is not None:
            input_ids[:, 1 : decoder_input_ids_length + 1, :] = decoder_input_ids

        delay_precomp = build_delay_indices(
            B=batch_size,
            T=1 + decoder_input_ids_length + max_delay_pattern,
            C=self.config.decoder_config.num_channels,
            delay_pattern=delay_pattern,
        )

        input_ids = apply_audio_delay(
            audio=input_ids,
            pad_value=self.config.pad_token_id,
            bos_value=self.config.bos_token_id,
            precomp=delay_precomp,
        )

        decoder_input_ids = input_ids[:, : 1 + decoder_input_ids_length, :]
        decoder_hangover = input_ids[:, 1 + decoder_input_ids_length :, :]
        decoder_attention_mask = decoder_attention_mask.expand(-1, decoder_input_ids.shape[1])

        model_kwargs["decoder_hangover"] = decoder_hangover
        model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    # TODO: shouldn't be necessary (esp after using prepare_inputs_for_gen...)
    def _prepare_attention_mask_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        generation_config: GenerationConfig,
        model_kwargs: Dict[str, Any],
    ) -> torch.LongTensor:
        return model_kwargs["attention_mask"]

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        device: torch.device,
    ) -> bool:
        # Batch size is doubled for unconditioned input
        return super()._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size * 2, max_cache_length, device
        )

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: Optional[int] = None,
        encoder_input_ids: torch.LongTensor = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        device: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorList:
        # TODO: hangover processor is removed --> use mask to overwrite the first xxx tokens with mask
        # TODO: make input_ids of audio (bsz * num_channels, seq_len)
        # --> reshape in forward/generate prep
        # --> logits can be processed by any of the other methods

        custom_processors = LogitsProcessorList()

        if generation_config.guidance_scale is not None and generation_config.guidance_scale != 1:
            custom_processors.append(
                DiaClassifierFreeGuidanceFilterLogitsProcessor(
                    cfg_scale=generation_config.guidance_scale
                    if generation_config.guidance_scale is not None else 3.0,
                    cfg_filter_top_k=model_kwargs.get("guidance_top_k", 50),
                    device=device,
                )
            )
            # Avoid adding cfg again
            generation_config.guidance_scale = None

        custom_processors.append(
            DiaEOSFilterAndScaleLogitsProcessor(
                eos_value=self.config.eos_token_id,
                eos_scale=model_kwargs.get("eos_scale", 0.8),
                device=device,
            )
        )

        custom_processors.append(
            DiaEOSDelayPatternLogitsProcessor(
                delay_pattern=torch.tensor(self.config.delay_pattern, device=device, dtype=torch.long),
                eos_value=self.config.eos_token_id,
                pad_value=self.config.pad_token_id,
                max_step=generation_config.max_length,
                device=device,
            )
        )

        return super()._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=encoder_input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=custom_processors,
            device=device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

    def _get_stopping_criteria(
        self,
        generation_config: GenerationConfig,
        stopping_criteria: Optional[StoppingCriteriaList],
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        **kwargs,
    ) -> StoppingCriteriaList:
        custom_stopping_criteria = StoppingCriteriaList()

        # We end generation after `max delays` if every sample generated eos on their first channel
        custom_stopping_criteria.append(
            DiaEosTokenCriteria(
                eos_value=self.config.eos_token_id,
                delay_pattern=torch.tensor(self.config.delay_pattern, device=self.device, dtype=torch.long),
                device=self.device,
            )
        )

        return super()._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=custom_stopping_criteria,
            tokenizer=tokenizer,
            **kwargs,
        )

    def _prepare_generation_config(
        self,
        generation_config: Optional[GenerationConfig],
        use_model_defaults: Optional[bool] = None,
        **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )

        # We allow generation up to max length + max delay pattern
        # (will revert back to max length after generation)
        # TODO: check where max delay wasn't considered but added afterwards
        generation_config.max_length += max(self.config.delay_pattern)

        # TODO: move this to prepare_inputs_for_generation
        # Enabling unconditioned input for generation
        model_kwargs["with_unconditioned_input"] = True

        # We need this for our custom logit processors and stopping criteria
        self.config.eos_token_id = self.config.eos_token_id or generation_config.eos_token_id
        self.config.pad_token_id = self.config.pad_token_id or generation_config.pad_token_id
        self.config.bos_token_id = self.config.bos_token_id or generation_config.bos_token_id

        return generation_config, model_kwargs

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        decoder_input_ids = kwargs.get("decoder_input_ids", None)
        decoder_input_length = decoder_input_ids.shape[1] if decoder_input_ids is not None else 0

        output = super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            use_model_defaults=use_model_defaults,
            custom_generate=custom_generate,
            **kwargs,
        )

        return_dict_in_generate = not isinstance(output, torch.Tensor)

        if return_dict_in_generate:
            output_sequences = output.sequences
        else:
            output_sequences = output

        # 1 for bos token
        output_sequences = output_sequences[:, 1 + decoder_input_length :]
        delay_pattern = torch.tensor(self.config.delay_pattern, dtype=torch.long, device=output_sequences.device)
        max_delay_pattern = delay_pattern.max().item()

        delay_precomp = build_revert_indices(
            B=output_sequences.shape[0],
            T=output_sequences.shape[1],
            C=self.config.decoder_config.num_channels,
            delay_pattern=delay_pattern,
        )
        output_sequences = revert_audio_delay(
            output_sequences,
            pad_value=self.config.pad_token_id,
            precomp=delay_precomp,
            T=output_sequences.shape[1],
        )

        # see `DiaEosTokenCriteria` why we need to +1
        output_sequences = output_sequences[:, -max_delay_pattern + 1 :]

        if return_dict_in_generate:
            output.sequences = output_sequences
        else:
            output = output_sequences

        return output


__all__ = [
    "DiaModel",
    "DiaPreTrainedModel",
    "DiaForConditionalGeneration",
]
