# Copyright 2026 The Dots Studio team and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Dots 3 Note Preview model for Hugging Face Transformers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import math
import random
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...image_transforms import convert_to_rgb
from ...masking_utils import create_bidirectional_mask, create_masks_for_generate
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...tokenization_utils_base import LARGE_INTEGER
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, logging, requires_backends
from ...utils.generic import merge_with_config_defaults, to_numpy
from ...utils.output_capturing import capture_outputs
from ...vision_utils import get_vision_attention_seqlens, get_vision_position_ids
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3RMSNorm
from ..deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32DecoderLayer,
    DeepseekV32Experts,
    DeepseekV32ForCausalLM,
    DeepseekV32MLP,
    DeepseekV32Model,
    DeepseekV32MoE,
    DeepseekV32PreTrainedModel,
    DeepseekV32TopkRouter,
    apply_rotary_pos_emb_interleave,
)
from ..evolla.modeling_evolla import EvollaFeedForward
from ..gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer
from ..glm_ocr.modeling_glm_ocr import GlmOcrVisionAttention
from ..llama.modeling_llama import LlamaMLP
from ..nemotron.modeling_nemotron import NemotronAttention
from ..phi.modeling_phi import PhiRotaryEmbedding
from ..phi3.modeling_phi3 import Phi3DecoderLayer, Phi3MLP
from ..qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from ..qwen2_vl.modeling_qwen2_vl import (
    PatchMerger,
    Qwen2VLVisionBlock,
    Qwen2VLVisionRotaryEmbedding,
)
from ..qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor, smart_resize
from .configuration_dots3_note import (
    Dots3NoteAudioConfig,
    Dots3NoteConfig,
    Dots3NoteVisionConfig,
)
from .feature_extraction_dots3_note import compute_audio_token_length


logger = logging.get_logger(__name__)


def apply_rotary_pos_emb_text(q, k, cos, sin, unsqueeze_dim=1):
    """Reuse DeepSeek's rotation and restore Dots' interleaved output layout."""
    q, k = apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    return tuple(x.unflatten(-1, (2, -1)).transpose(-1, -2).flatten(-2) for x in (q, k))


# -----------------------------------------------------------------------------
# Text decoder
# -----------------------------------------------------------------------------
class Dots3NoteTextRMSNorm(DeepseekV3RMSNorm):
    pass


# ---------------------------------------------------------------------------
# Rotary embedding (decoupled RoPE part only, GPT-J / interleaved style)
# ---------------------------------------------------------------------------
class Dots3NoteTextRotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config: Dots3NoteConfig, device=None):
        config = copy(config)
        config.rope_parameters = {
            layer_type: config.get_layer_config(layer_type).rope_parameters for layer_type in set(config.layer_types)
        }
        super().__init__(config, device)

    @classmethod
    def compute_default_rope_parameters(cls, config: Dots3NoteConfig, device=None, layer_type=None, **kwargs):
        config = copy(config)
        config.head_dim = config.swa_qk_rope_head_dim if layer_type == "sliding_attention" else config.qk_rope_head_dim
        return super().compute_default_rope_parameters(config, device=device, layer_type=layer_type, **kwargs)


# ---------------------------------------------------------------------------
# Dynamic sparse attention indexer
# ---------------------------------------------------------------------------
def _padding_mask_for_key_length(padding_mask, key_length):
    """Trim or right-pad a 2D padding mask to the physical cache width."""
    padding_mask = padding_mask[:, :key_length].to(torch.bool)
    if padding_mask.shape[-1] < key_length:
        padding_mask = F.pad(padding_mask, (0, key_length - padding_mask.shape[-1]), value=False)
    return padding_mask


class Dots3NoteTextIndexer(GlmMoeDsaIndexer):
    # TODO: Support loading original FP8 indexer weights and scales into the standard projections.
    def __init__(self, config: Dots3NoteConfig, layer_idx: int):
        super().__init__(config, layer_idx)


# ---------------------------------------------------------------------------
# Dense MLP
# ---------------------------------------------------------------------------
class Dots3NoteTextMLP(DeepseekV32MLP):
    pass


# ---------------------------------------------------------------------------
# MoE gate (noaux_tc: sigmoid scores + correction bias)
# ---------------------------------------------------------------------------
class Dots3NoteTextTopkRouter(DeepseekV32TopkRouter):
    def __init__(self, config):
        super().__init__(config)
        del self.e_score_correction_bias
        self.e_score_correction_bias = nn.Parameter(torch.empty(self.num_experts, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Sparse MoE FFN
# ---------------------------------------------------------------------------
class Dots3NoteTextExperts(DeepseekV32Experts):
    pass


class Dots3NoteTextMoE(DeepseekV32MoE):
    def __init__(self, config):
        super().__init__(config)
        del self.shared_experts
        shared_inter = config.shared_experts_intermediate_size * config.n_shared_experts
        self.shared_experts = Dots3NoteTextMLP(config, intermediate_size=shared_inter)


# ---------------------------------------------------------------------------
# MLA attention
# ---------------------------------------------------------------------------
def dots3_note_text_eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    fully_masked = None
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        if attention_mask.dtype == torch.bool:
            fully_masked = ~attention_mask.any(dim=-1, keepdim=True)
            attn_weights = attn_weights.masked_fill(~attention_mask, torch.finfo(attn_weights.dtype).min)
        else:
            fully_masked = attention_mask.amax(dim=-1, keepdim=True) < 0
            attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if fully_masked is not None:
        attn_weights = attn_weights.masked_fill(fully_masked, 0)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


def dsa_sparse_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling=None,
    dropout=0.0,
    indices=None,
    query_positions=None,
    padding_mask=None,
    query_chunk_size=512,
    head_chunk_size=32,
    **kwargs,
):
    """Reference DSA fallback that computes attention only over indexer-selected tokens."""
    if indices is None:
        raise ValueError("The DSA sparse attention fallback requires top-k `indices`.")
    if scaling is None:
        scaling = module.scaling

    batch_size, num_heads, query_length, _ = query.shape
    output = value.new_empty(batch_size, num_heads, query_length, value.shape[-1])
    batch_indices = torch.arange(batch_size, device=query.device)[:, None, None, None]
    if query_positions is None:
        query_positions = torch.arange(key.shape[2] - query_length, key.shape[2], device=query.device)

    for query_start in range(0, query_length, query_chunk_size):
        query_stop = min(query_start + query_chunk_size, query_length)
        token_indices = indices[:, query_start:query_stop].long()
        valid = token_indices <= query_positions[None, query_start:query_stop, None]
        if padding_mask is not None:
            selected_padding = _padding_mask_for_key_length(padding_mask, key.shape[2]).gather(
                1, token_indices.flatten(1)
            )
            valid = valid & selected_padding.view_as(token_indices)

        mask_chunk = None
        if attention_mask is not None:
            mask_chunk = attention_mask[:, :, query_start:query_stop, : key.shape[2]]

        for head_start in range(0, num_heads, head_chunk_size):
            head_stop = min(head_start + head_chunk_size, num_heads)
            head_indices = torch.arange(head_start, head_stop, device=query.device)[None, :, None, None]
            selected_key = key[batch_indices, head_indices, token_indices[:, None], :]
            scores = torch.einsum(
                "bhqd,bhqkd->bhqk",
                query[:, head_start:head_stop, query_start:query_stop],
                selected_key,
            ).float()
            scores.mul_(scaling)
            head_valid = valid[:, None]
            selected_mask = None
            if mask_chunk is not None:
                head_mask = mask_chunk if mask_chunk.shape[1] == 1 else mask_chunk[:, head_start:head_stop]
                selected_mask = head_mask.expand(-1, head_stop - head_start, -1, -1).gather(
                    -1, token_indices[:, None].expand(-1, head_stop - head_start, -1, -1)
                )
                if selected_mask.dtype == torch.bool:
                    head_valid = head_valid & selected_mask
                else:
                    scores.add_(selected_mask.float())
            scores.masked_fill_(~head_valid, torch.finfo(scores.dtype).min)
            probabilities = F.softmax(scores, dim=-1).to(query.dtype)
            fully_masked = ~head_valid.any(dim=-1, keepdim=True)
            if selected_mask is not None and selected_mask.dtype != torch.bool:
                fully_masked = fully_masked | (selected_mask.amax(dim=-1, keepdim=True) < 0)
            probabilities.masked_fill_(fully_masked, 0)
            probabilities = F.dropout(probabilities, p=dropout, training=module.training)
            selected_value = value[batch_indices, head_indices, token_indices[:, None], :]
            output[:, head_start:head_stop, query_start:query_stop] = torch.einsum(
                "bhqk,bhqkd->bhqd", probabilities, selected_value
            )

    return output.transpose(1, 2).contiguous(), None


class Dots3NoteTextAttention(DeepseekV32Attention):
    def __init__(self, config: Dots3NoteConfig, layer_idx: int, is_sliding: bool = False):
        original_config = config
        config = config.get_layer_config("sliding_attention" if is_sliding else "full_attention")
        super().__init__(config, layer_idx)
        self.config = original_config
        self.head_dim = config.head_dim
        if is_sliding and self.head_dim != self.qk_head_dim:
            raise ValueError(
                f"SWA head_dim ({self.head_dim}) must equal qk_nope_head_dim + qk_rope_head_dim ({self.qk_head_dim})."
            )

        self.apply_lora_scale = config.apply_mla_qkv_lora_rescale
        self.scaling = self.qk_head_dim**-0.5

        self.sliding_window = config.sliding_window if is_sliding else None

        if self.q_lora_rank is not None:
            self.q_a_layernorm.variance_epsilon = config.rms_norm_eps
        self.kv_a_layernorm.variance_epsilon = config.rms_norm_eps
        # CODEPATH: released checkpoints have no attention bias; custom configs may enable it.
        if config.attention_bias:
            for projection in (self.q_b_proj if self.q_lora_rank is not None else self.q_proj, self.kv_b_proj):
                projection.bias = nn.Parameter(torch.empty(projection.out_features))

        # CODEPATH: released Dots 3 Note Preview checkpoints enable K-RoPE LayerNorm; custom configs may disable it.
        if config.k_rope_only_layernorm:
            self.k_rope_only_layernorm = Dots3NoteTextRMSNorm(self.qk_rope_head_dim, config.rms_norm_eps)
        else:
            self.k_rope_only_layernorm = None

        # output sigmoid gate
        self.attention_gate_type = config.attention_gate_type
        if self.attention_gate_type == "elementwise":
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim, bias=config.attention_bias)
        elif self.attention_gate_type == "headwise":
            self.g_proj = nn.Linear(self.hidden_size, self.num_heads, bias=config.attention_bias)
        else:
            self.g_proj = None

        del self.indexer

    def _prepare_attention(
        self,
        hidden_states,
        q_lora,
        cos,
        sin,
        key_states,
        attention_mask,
        padding_mask,
        query_positions,
        past_key_value,
        output_attentions,
    ):
        return attention_mask, self.config._attn_implementation, dots3_note_text_eager_attention_forward, None

    def forward(
        self,
        hidden_states,
        cos=None,
        sin=None,
        attention_mask=None,
        padding_mask=None,
        past_key_value=None,
        past_key_values=None,
        output_attentions=False,
        position_embeddings=None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if position_embeddings is not None:
            if cos is not None or sin is not None:
                raise ValueError("Pass either `position_embeddings` or `cos`/`sin`, not both")
            cos, sin = position_embeddings
        if cos is None or sin is None:
            raise ValueError("Dots3NoteTextAttention requires rotary position embeddings")
        if past_key_values is not None:
            if past_key_value is not None:
                raise ValueError("Pass either `past_key_values` or `past_key_value`, not both")
            past_key_value = past_key_values
        output_attentions = output_attentions or self.config.output_attentions
        bsz, q_len, _ = hidden_states.size()
        past_len = past_key_value.get_seq_length(self.layer_idx) if past_key_value is not None else 0
        query_positions = torch.arange(q_len, device=hidden_states.device) + past_len

        # ---- query ----
        q_lora = None
        if self.q_lora_rank is not None:
            q_lora = self.q_a_proj(hidden_states)
            q_lora = self.q_a_layernorm(q_lora)
            if self.apply_lora_scale:
                q_lora = q_lora * (self.hidden_size / self.q_lora_rank) ** 0.5
            q = self.q_b_proj(q_lora)
        else:
            q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ---- compressed kv ----
        latent = self.kv_a_proj_with_mqa(hidden_states)
        kv_a, k_pe = torch.split(latent, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        if self.apply_lora_scale:
            kv_a = kv_a * (self.hidden_size / self.kv_lora_rank) ** 0.5

        # decoupled rope key: single (mqa) head, shared across heads
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)  # [B,1,S,rope]
        if self.k_rope_only_layernorm is not None:
            k_pe = self.k_rope_only_layernorm(k_pe)

        q_pe, k_pe = apply_rotary_pos_emb_text(q_pe, k_pe, cos, sin)
        q_pe, k_pe = q_pe.to(q.dtype), k_pe.to(kv_a.dtype)

        query_states = torch.cat([q_nope, q_pe], dim=-1)  # [B,H,S,qk_head_dim]
        key_states, value_states = self.expand_kv(kv_a.unsqueeze(1), k_pe)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        if attention_mask is not None and attention_mask.shape[-1] != key_states.shape[-2]:
            attention_mask = attention_mask[..., -key_states.shape[-2] :]

        attention_mask, attention_backend, attention_default, sparse_indices = self._prepare_attention(
            hidden_states,
            q_lora,
            cos,
            sin,
            key_states,
            attention_mask,
            padding_mask,
            query_positions,
            past_key_value,
            output_attentions,
        )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(attention_backend, attention_default)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            is_causal=attention_mask is None,
            output_attentions=output_attentions,
            indices=sparse_indices,
            query_positions=query_positions,
            padding_mask=padding_mask,
            **kwargs,
        )

        # SDPA treats a finite all-negative additive row as a bias rather than as fully masked.
        # Match the eager and sparse DSA paths by explicitly zeroing those query/head outputs.
        if attention_backend == "sdpa" and isinstance(attention_mask, torch.Tensor):
            fully_masked = (
                ~attention_mask.any(dim=-1) if attention_mask.dtype == torch.bool else attention_mask.amax(dim=-1) < 0
            )
            if fully_masked.ndim == 1:
                fully_masked = fully_masked[:, None, None]
            elif fully_masked.ndim == 2:
                fully_masked = fully_masked[:, :, None]
            else:
                fully_masked = fully_masked.transpose(1, 2)
            attn_output = attn_output.masked_fill(fully_masked.unsqueeze(-1), 0)

        # ---- output sigmoid gate ----
        if self.g_proj is not None:
            g = self.g_proj(hidden_states)  # [B,S,H] (headwise) or [B,S,H*v] (elementwise)
            if self.attention_gate_type == "elementwise":
                g = g.view(bsz, q_len, self.num_heads, self.v_head_dim)
                attn_output = attn_output * torch.sigmoid(g)
            else:  # headwise
                attn_output = attn_output * torch.sigmoid(g).unsqueeze(-1)

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None


class Dots3NoteTextSparseAttention(Dots3NoteTextAttention):
    def __init__(self, config: Dots3NoteConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if self.q_lora_rank is None:
            raise ValueError("DSA requires q_lora_rank to construct the indexer query")
        self.indexer = Dots3NoteTextIndexer(config, layer_idx)

    def _prepare_attention(
        self,
        hidden_states,
        q_lora,
        cos,
        sin,
        key_states,
        attention_mask,
        padding_mask,
        query_positions,
        past_key_value,
        output_attentions,
    ):
        bsz, q_len = hidden_states.shape[:2]
        key_positions = torch.arange(key_states.shape[-2], device=hidden_states.device)
        indexer_mask = key_positions[None, None, :] <= query_positions[None, :, None]
        if padding_mask is not None:
            indexer_mask = indexer_mask & _padding_mask_for_key_length(padding_mask, key_states.shape[-2])[:, None]
        if attention_mask is not None:
            mask = attention_mask
            if mask.ndim == 4:
                if mask.shape[1] != 1:
                    raise ValueError("DSA requires a shared mask; different per-head masks are not supported")
                mask = mask[:, 0]
            if mask.ndim != 3:
                raise ValueError(f"DSA attention_mask must be 3D or 4D, got {attention_mask.ndim}D")
            indexer_mask = (
                indexer_mask & mask
                if mask.dtype == torch.bool
                else mask.float().masked_fill(~indexer_mask, float("-inf"))
            )
        topk_indices = self.indexer(
            hidden_states, q_lora, (cos, sin), indexer_mask, query_positions, past_key_values=past_key_value
        )

        sparse_fallback = (
            not output_attentions
            and self.config._attn_implementation in ("eager", "sdpa")
            and key_states.shape[-2] > topk_indices.shape[-1] * 2
        )
        if sparse_fallback:
            # Generic eager/SDPA cannot consume sparse indices without materializing an O(QK) mask.
            # Dispatch the memory-bounded eager fallback through the common attention interface.
            attention_backend = "eager"
            attention_default = dsa_sparse_attention_forward
            sparse_indices = topk_indices
        else:
            attention_backend = "eager" if output_attentions else self.config._attn_implementation
            attention_default = dots3_note_text_eager_attention_forward
            sparse_indices = None if attention_backend in ("eager", "sdpa") else topk_indices

        if not sparse_fallback and sparse_indices is None:
            index_mask = (
                topk_indices.new_ones((bsz, q_len, key_states.shape[-2]), dtype=torch.bool)
                .scatter(-1, topk_indices.long(), False)
                .unsqueeze(1)
            )
            key_positions = torch.arange(key_states.shape[-2], device=hidden_states.device)
            index_mask |= key_positions[None, None, None, :] > query_positions[None, None, :, None]
            if padding_mask is not None:
                index_mask |= ~_padding_mask_for_key_length(padding_mask, key_states.shape[-2])[:, None, None]
            if attention_mask is None:
                attention_mask = ~index_mask
            elif attention_mask.dtype == torch.bool:
                attention_mask = attention_mask & ~index_mask
            else:
                attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
        return attention_mask, attention_backend, attention_default, sparse_indices


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------
class Dots3NoteTextDecoderLayer(DeepseekV32DecoderLayer):
    def __init__(self, config: Dots3NoteConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = (
            # CODEPATH: released DSA layers are sparse; SWA and non-DSA variants use dense attention.
            Dots3NoteTextSparseAttention(config, layer_idx)
            if config.layer_types[layer_idx] == "deepseek_sparse_attention"
            else Dots3NoteTextAttention(
                config,
                layer_idx,
                is_sliding=config.layer_types[layer_idx] == "sliding_attention",
            )
        )


# ---------------------------------------------------------------------------
# Pretrained base
# ---------------------------------------------------------------------------
@auto_docstring
class Dots3NotePreTrainedModel(DeepseekV32PreTrainedModel):
    config: Dots3NoteConfig
    config_class = Dots3NoteConfig
    _no_split_modules = ["Dots3NoteTextDecoderLayer"]
    _can_compile_fullgraph = False
    _keep_in_fp32_modules = []
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\.46\.", r"^model\.mtp\."]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Dots3NoteTextRotaryEmbedding):
            for layer_type in module.layer_types:
                inv_freq, _ = module.compute_default_rope_parameters(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), inv_freq)


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
class Dots3NoteTextModel(Dots3NotePreTrainedModel, DeepseekV32Model):
    def __init__(self, config: Dots3NoteConfig):
        super().__init__(config)
        self.num_hidden_layers = config.num_hidden_layers
        self.rotary_emb = Dots3NoteTextRotaryEmbedding(config)

    @merge_with_config_defaults
    @capture_outputs
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
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, q_len = inputs_embeds.shape[:2]

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = (torch.arange(q_len, device=inputs_embeds.device) + past_len).unsqueeze(0).expand(bsz, -1)

        position_embeddings = {
            layer_type: self.rotary_emb(
                inputs_embeds.float() if layer_type == "deepseek_sparse_attention" else inputs_embeds,
                position_ids,
                layer_type,
            )
            for layer_type in set(self.config.layer_types)
        }

        padding_mask = (
            attention_mask if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2 else None
        )
        causal_masks = attention_mask
        if not isinstance(causal_masks, dict):
            causal_masks = create_masks_for_generate(
                self.config, inputs_embeds, attention_mask, past_key_values, position_ids
            )

        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers[: self.num_hidden_layers]):
            layer_mask = causal_masks[self.config.layer_types[layer_idx]]
            hidden_states = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings[self.config.layer_types[layer_idx]],
                position_ids=position_ids,
                padding_mask=padding_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Dots3NoteTextForCausalLM(Dots3NotePreTrainedModel, DeepseekV32ForCausalLM):
    pass


# -----------------------------------------------------------------------------
# Audio encoder and adapter
# -----------------------------------------------------------------------------
class Dots3NoteAudioRMSNorm(Dots3NoteTextRMSNorm):
    pass


class Dots3NoteAudioRotaryEmbedding(PhiRotaryEmbedding):
    def __init__(self, config: Dots3NoteAudioConfig, device=None):
        config = copy(config)
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dim = int(head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0)) // 2 * 2
        config.head_dim = rotary_dim or head_dim
        config.rope_parameters = {
            "rope_type": "default",
            **config.rope_parameters,
            "partial_rotary_factor": float(rotary_dim != 0),
        }
        super().__init__(config, device)


class Dots3NoteAudioAttention(NemotronAttention):
    def __init__(self, config: Dots3NoteAudioConfig):
        config = copy(config)
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.num_key_value_heads = config.num_attention_heads
        config.attention_bias = True
        super().__init__(config)
        self.layer_idx = None
        del self.partial_rotary_factor
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.is_causal = False


class Dots3NoteAudioMLP(Phi3MLP):
    def __init__(self, config: Dots3NoteAudioConfig):
        config = copy(config)
        config.hidden_act = "silu"
        super().__init__(config)
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)


class Dots3NoteAudioEncoderLayer(Phi3DecoderLayer):
    def __init__(self, config: Dots3NoteAudioConfig):
        nn.Module.__init__(self)
        hidden_size = config.hidden_size
        # CODEPATH: released checkpoints use RMSNorm; custom audio configs may select LayerNorm.
        norm_class = Dots3NoteAudioRMSNorm if config.use_rms_norm else nn.LayerNorm
        attention_config = copy(config)
        attention_config._attn_implementation = config.attention_backend
        self.self_attn = Dots3NoteAudioAttention(attention_config)
        self.input_layernorm = norm_class(hidden_size)
        self.mlp = Dots3NoteAudioMLP(config)
        self.post_attention_layernorm = norm_class(hidden_size)
        self.resid_attn_dropout = nn.Dropout(config.dropout)
        self.resid_mlp_dropout = nn.Dropout(config.dropout)


class Dots3NoteAudioConvStem(nn.Module):
    def __init__(self, config: Dots3NoteAudioConfig):
        super().__init__()
        hidden_size = config.hidden_size
        downsample_size = config.downsample_hidden_size
        self.conv2d1 = nn.Conv2d(1, downsample_size, 3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(downsample_size, downsample_size, 3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(downsample_size, downsample_size, 3, stride=2, padding=1)
        frequency_bins = config.feature_size
        for _ in range(3):
            frequency_bins = (frequency_bins + 1) // 2
        self.conv_out = nn.Linear(downsample_size * frequency_bins, hidden_size, bias=False)
        self.hop_length = config.hop_length
        self.conv_temporal_stride = config.conv_temporal_stride
        self.conv_bucket_step = config.conv_bucket_step
        self.conv_bucket_max_elements = config.conv_bucket_max_elements

    def _mask_time(self, hidden_states: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(hidden_states.shape[-1], device=hidden_states.device)
        mask = positions[None, :] < valid_lengths[:, None]
        return hidden_states * mask[:, None, None, :]

    def _conv_layers(
        self,
        hidden_states: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self._mask_time(hidden_states, valid_lengths)
        hidden_states = F.gelu(self.conv2d1(hidden_states))
        valid_lengths = (valid_lengths + 1) // 2
        hidden_states = self._mask_time(hidden_states, valid_lengths)
        hidden_states = F.gelu(self.conv2d2(hidden_states))
        valid_lengths = (valid_lengths + 1) // 2
        hidden_states = self._mask_time(hidden_states, valid_lengths)
        hidden_states = F.gelu(self.conv2d3(hidden_states))
        valid_lengths = (valid_lengths + 1) // 2
        hidden_states = self._mask_time(hidden_states, valid_lengths)
        return hidden_states

    def _project_conv_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channels, frequencies, frames = hidden_states.shape
        hidden_states = hidden_states.permute(0, 3, 1, 2).reshape(batch_size, frames, channels * frequencies)
        return self.conv_out(hidden_states)

    def forward(
        self,
        input_features: torch.Tensor,
        audio_sample_lengths: torch.Tensor,
        input_seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = input_features.unsqueeze(1)
        valid_lengths = audio_sample_lengths.to(hidden_states.device) // self.hop_length
        if self.conv_bucket_step is None:
            return self._project_conv_output(self._conv_layers(hidden_states, valid_lengths))

        step_frames = int(self.conv_bucket_step * 100)
        boundaries = list(range(step_frames, hidden_states.shape[-1], step_frames))
        boundaries.append(hidden_states.shape[-1])
        groups: dict[int, list[int]] = {}
        for index, sequence_length in enumerate(input_seq_lens.tolist()):
            actual_mel_frames = sequence_length * self.conv_temporal_stride
            bucket_frames = next(boundary for boundary in boundaries if actual_mel_frames <= boundary)
            groups.setdefault(bucket_frames, []).append(index)

        output = hidden_states.new_zeros(
            hidden_states.shape[0],
            int(input_seq_lens.max().item()),
            self.conv_out.out_features,
        )
        for bucket_frames, indexes in sorted(groups.items()):
            index_tensor = torch.tensor(indexes, device=hidden_states.device)
            bucket = hidden_states[index_tensor, :, :, :bucket_frames]
            bucket_valid_lengths = valid_lengths[index_tensor]
            max_elements = self.conv_bucket_max_elements or bucket_frames * len(indexes)
            sub_batch_size = max(1, max_elements // bucket_frames)
            bucket_outputs = []
            for start in range(0, bucket.shape[0], sub_batch_size):
                bucket_outputs.append(
                    self._conv_layers(
                        bucket[start : start + sub_batch_size],
                        bucket_valid_lengths[start : start + sub_batch_size],
                    )
                )
            projected = self._project_conv_output(torch.cat(bucket_outputs, dim=0))
            for local_index, global_index in enumerate(indexes):
                valid_length = int(input_seq_lens[global_index].item())
                output[global_index, :valid_length] = projected[local_index, :valid_length]
        return output


class Dots3NoteSpeechEncoder(nn.Module):
    def __init__(self, config: Dots3NoteAudioConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.conv_stem = Dots3NoteAudioConvStem(config)
        self.rotary_embedding = Dots3NoteAudioRotaryEmbedding(config)
        self.layers = nn.ModuleList([Dots3NoteAudioEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # CODEPATH: released checkpoints use RMSNorm; custom audio configs may select LayerNorm.
        norm_class = Dots3NoteAudioRMSNorm if config.use_rms_norm else nn.LayerNorm
        self.layer_norm = norm_class(hidden_size)
        self.dropout = config.dropout

    @can_return_tuple
    def forward(
        self,
        input_features: torch.Tensor,
        input_seq_lens: torch.Tensor,
        audio_sample_lens: torch.Tensor,
    ) -> BaseModelOutput | tuple[torch.Tensor]:
        hidden_states = self.conv_stem(input_features, audio_sample_lens, input_seq_lens)
        max_valid_length = int(input_seq_lens.max().item())
        hidden_states = hidden_states[:, :max_valid_length]
        positions = torch.arange(max_valid_length, device=hidden_states.device)[None, :]
        cosine, sine = self.rotary_embedding(hidden_states, positions)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        attention_mask = positions < input_seq_lens[:, None]
        attention_mask = create_bidirectional_mask(self.layers[0].self_attn.config, hidden_states, attention_mask)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_embeddings=(cosine, sine))
        hidden_states = self.layer_norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class Dots3NoteAudioAdapter(EvollaFeedForward):
    def __init__(self, input_size: int, output_size: int):
        nn.Module.__init__(self)
        self.norm = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, output_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(output_size, output_size)


@auto_docstring
@dataclass
class Dots3NoteAudioOutput(ModelOutput):
    """
    Args:
        audio_embeds (`torch.Tensor`, *optional*): Encoded audio tokens in the language-model width.
        audio_token_lengths (`torch.Tensor`, *optional*): Number of encoded tokens for each audio input.
    """

    audio_embeds: torch.Tensor | None = None
    audio_token_lengths: torch.Tensor | None = None


@auto_docstring
class Dots3NoteAudioPreTrainedModel(PreTrainedModel):
    config_class = Dots3NoteAudioConfig
    base_model_prefix = "audio_encoder"
    main_input_name = "input_features"
    _no_split_modules = ["Dots3NoteAudioEncoderLayer"]
    _supports_sdpa = True


@auto_docstring
class Dots3NoteAudioModel(Dots3NoteAudioPreTrainedModel):
    """Audio tower with checkpoint-compatible module names."""

    def __init__(self, config: Dots3NoteAudioConfig):
        super().__init__(config)
        # CODEPATH: all released checkpoints require merge_factor=1; reject incompatible custom configurations.
        if config.merge_factor != 1:
            raise ValueError("the current Dots 3 Note Preview AE release requires merge_factor=1")
        self.audio_adapter = Dots3NoteAudioAdapter(
            config.adapter_input_size,
            config.adapter_output_size,
        )
        self.speech_encoder = Dots3NoteSpeechEncoder(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_features: torch.Tensor,
        chunk_sample_lengths: torch.Tensor,
        chunk_token_lengths: torch.Tensor,
        audio_chunk_counts: torch.Tensor,
        **kwargs,
    ) -> Dots3NoteAudioOutput | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            chunk_sample_lengths (`torch.Tensor`): Number of waveform samples represented by each feature chunk.
            chunk_token_lengths (`torch.Tensor`): Number of encoder tokens produced by each feature chunk.
            audio_chunk_counts (`torch.Tensor`): Number of feature chunks belonging to each audio input.
        """
        encoder_output = self.speech_encoder(
            input_features=input_features,
            input_seq_lens=chunk_token_lengths,
            audio_sample_lens=chunk_sample_lengths,
            return_dict=True,
        ).last_hidden_state

        if int(audio_chunk_counts.sum()) != len(chunk_token_lengths):
            raise ValueError("audio_chunk_counts does not account for every feature chunk")
        valid = (
            torch.arange(encoder_output.shape[1], device=encoder_output.device)[None] < chunk_token_lengths[:, None]
        )
        embeddings = self.audio_adapter(encoder_output[valid])
        lengths = torch.stack(
            [lengths.sum() for lengths in chunk_token_lengths.split(audio_chunk_counts.tolist())]
        ).to(device=embeddings.device, dtype=torch.long)
        return Dots3NoteAudioOutput(audio_embeds=embeddings, audio_token_lengths=lengths)


# -----------------------------------------------------------------------------
# Vision encoder and adapter
# -----------------------------------------------------------------------------
class Dots3NoteVisionRMSNorm(Dots3NoteTextRMSNorm):
    pass


class Dots3NoteVisionRotaryEmbedding(Qwen2VLVisionRotaryEmbedding):
    pass


class Dots3NoteVisionPatchEmbed(nn.Module):
    def __init__(self, config: Dots3NoteVisionConfig):
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        self.proj = nn.Conv2d(self.num_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = Dots3NoteVisionRMSNorm(self.embed_dim, eps=config.rms_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.reshape(
            -1,
            self.num_channels,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(pixel_values.to(self.proj.weight.dtype)).reshape(-1, self.embed_dim)
        return self.norm(hidden_states)


class Dots3NoteVisionMLP(LlamaMLP):
    def __init__(self, config: Dots3NoteVisionConfig, intermediate_size: int | None = None):
        config = copy(config)
        config.hidden_size = config.embed_dim
        config.intermediate_size = intermediate_size or config.intermediate_size
        config.mlp_bias = config.use_bias
        config.hidden_act = "silu"
        super().__init__(config)


class Dots3NoteVisionMoE(nn.Module):
    def __init__(self, config: Dots3NoteVisionConfig, layer_idx: int):
        super().__init__()
        self.num_experts = config.pyramid_num_routed[layer_idx]
        self.top_k = min(int(config.capacity_factor), self.num_experts)
        self.router_scoring_func = config.router_scoring_func
        self.router_scale = config.router_scale
        self.experts = nn.ModuleList(
            [
                Dots3NoteVisionMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )
        self.gate_weight = nn.Parameter(torch.empty(self.num_experts, config.embed_dim, dtype=torch.float32))
        self.router_bias = nn.Buffer(torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, shape[-1])
        router_logits = F.linear(hidden_states.float(), self.gate_weight.float())
        # Keep routing in FP32 so near-tied expert scores do not collapse before top-k selection.
        if self.router_scoring_func == "softmax":
            router_probs = router_logits.softmax(dim=-1, dtype=torch.float32)
        else:
            router_probs = router_logits.sigmoid()

        _, selected_experts = torch.topk(
            router_probs + self.router_bias.float().unsqueeze(0), self.top_k, dim=-1, sorted=False
        )
        routing_weights = router_probs.gather(1, selected_experts)
        if self.router_scoring_func == "sigmoid" and self.top_k > 1:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)
        routing_weights = (routing_weights * self.router_scale).to(hidden_states.dtype)

        output = torch.zeros_like(hidden_states)
        weight_sum = torch.zeros(hidden_states.shape[0], dtype=hidden_states.dtype, device=hidden_states.device)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, top_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            weights = routing_weights[token_idx, top_idx]
            output[token_idx] += expert(hidden_states[token_idx]) * weights.unsqueeze(-1)
            weight_sum[token_idx] += weights
        output = output / (weight_sum.unsqueeze(-1) + 1e-9)
        return output.reshape(shape)


class Dots3NoteVisionAttention(GlmOcrVisionAttention):
    def __init__(self, config: Dots3NoteVisionConfig):
        config = copy(config)
        config.hidden_size = config.embed_dim
        config.attention_bias = config.use_bias
        super().__init__(config)
        self.is_causal = config.is_causal
        # CODEPATH: released checkpoints enable QK normalization; custom vision configs may disable it.
        if not config.use_qk_norm:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()


class Dots3NoteVisionBlock(Qwen2VLVisionBlock):
    def __init__(self, config: Dots3NoteVisionConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.norm1 = Dots3NoteVisionRMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.attn = Dots3NoteVisionAttention(config)
        self.norm2 = Dots3NoteVisionRMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        num_experts = config.pyramid_num_routed[layer_idx]
        self.mlp = Dots3NoteVisionMLP(config) if num_experts < 1 else Dots3NoteVisionMoE(config, layer_idx)


class Dots3NoteVisionAdapter(PatchMerger):
    pass


@auto_docstring
class Dots3NoteVisionPreTrainedModel(PreTrainedModel):
    config_class = Dots3NoteVisionConfig
    base_model_prefix = "vision_encoder"
    main_input_name = "pixel_values"
    input_modalities = ("image", "video")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Dots3NoteVisionBlock"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        if isinstance(module, Dots3NoteVisionMoE):
            init.normal_(module.gate_weight, mean=0.0, std=self.config.initializer_range)
            init.zeros_(module.router_bias)


@auto_docstring
class Dots3NoteVisionModel(Dots3NoteVisionPreTrainedModel):
    def __init__(self, config: Dots3NoteVisionConfig):
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Dots3NoteVisionPatchEmbed(config)
        self.rotary_pos_emb = Dots3NoteVisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList(
            [Dots3NoteVisionBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_trunk_norm = (
            # CODEPATH: released checkpoints use the post-trunk norm; custom vision configs may disable it.
            Dots3NoteVisionRMSNorm(config.embed_dim, eps=config.rms_norm_eps) if config.post_norm else None
        )
        self.adapter = Dots3NoteVisionAdapter(
            dim=config.adapter_out_dim,
            context_dim=config.adapter_in_dim,
            spatial_merge_size=config.adapter_merge_size,
        )
        self.gradient_checkpointing = False
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPooling | tuple[torch.Tensor, ...]:
        """
        Args:
            grid_thw (`torch.Tensor`): Temporal, height, and width patch-grid dimensions for each input.
        """
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        hidden_states = self.patch_embed(pixel_values)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        if self.post_trunk_norm is not None:
            hidden_states = self.post_trunk_norm(hidden_states)
        merged_hidden_states = self.adapter(hidden_states)
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            hidden_states=all_hidden_states,
        )


# -----------------------------------------------------------------------------
# Unified multimodal model
# -----------------------------------------------------------------------------
@auto_docstring
class Dots3NoteModel(Dots3NotePreTrainedModel):
    config_class = Dots3NoteConfig
    input_modalities = ("image", "video", "audio", "text")
    _no_split_modules = [
        "Dots3NoteAudioEncoderLayer",
        "Dots3NoteTextDecoderLayer",
        "Dots3NoteVisionBlock",
    ]

    def __init__(self, config: Dots3NoteConfig):
        super().__init__(config)
        self.language_model = Dots3NoteTextModel(config)
        self.vision_encoder = Dots3NoteVisionModel(config.vision_config)
        self.audio_encoder = Dots3NoteAudioModel(config.audio_config)
        self.post_init()

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Encode image patches and return embeddings in the language-model width."""
        return self.vision_encoder(
            pixel_values.to(dtype=self.vision_encoder.patch_embed.proj.weight.dtype),
            grid_thw=image_grid_thw,
            return_dict=True,
            **kwargs,
        ).pooler_output

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Encode video patches with the shared vision encoder."""
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    @auto_docstring
    def get_audio_features(
        self,
        input_features: torch.Tensor,
        chunk_sample_lengths: torch.Tensor,
        chunk_token_lengths: torch.Tensor,
        audio_chunk_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            chunk_sample_lengths (`torch.Tensor`): Number of waveform samples represented by each feature chunk.
            chunk_token_lengths (`torch.Tensor`): Number of encoder tokens produced by each feature chunk.
            audio_chunk_counts (`torch.Tensor`): Number of feature chunks belonging to each audio input.
        """
        parameter = next(self.audio_encoder.parameters())
        output = self.audio_encoder(
            input_features=input_features.to(device=parameter.device, dtype=parameter.dtype),
            chunk_sample_lengths=chunk_sample_lengths.to(parameter.device),
            chunk_token_lengths=chunk_token_lengths.to(parameter.device),
            audio_chunk_counts=audio_chunk_counts.to(parameter.device),
            return_dict=True,
        )
        return output.audio_embeds, output.audio_token_lengths

    @staticmethod
    def _merge_multimodal_embeddings(
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: torch.Tensor,
        token_id: int,
        modality: str,
    ) -> torch.Tensor:
        special_token_mask = input_ids.eq(token_id)
        placeholder_count = int(special_token_mask.sum())
        embedding_count = multimodal_embeddings.shape[0]
        if placeholder_count != embedding_count:
            raise ValueError(
                f"{modality} embedding/token mismatch: found {placeholder_count} placeholder token(s), "
                f"but the encoder produced {embedding_count} embedding(s)"
            )
        special_token_mask = special_token_mask.unsqueeze(-1).expand_as(inputs_embeds)
        multimodal_embeddings = multimodal_embeddings.to(inputs_embeds.device, inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(special_token_mask, multimodal_embeddings)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        chunk_sample_lengths: torch.Tensor | None = None,
        chunk_token_lengths: torch.Tensor | None = None,
        audio_chunk_counts: torch.Tensor | None = None,
        audio_token_lengths: torch.Tensor | None = None,
        chunk_audio_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast | tuple:
        """
        Args:
            chunk_sample_lengths (`torch.Tensor`, *optional*): Waveform sample count for each audio feature chunk.
            chunk_token_lengths (`torch.Tensor`, *optional*): Encoder token count for each audio feature chunk.
            audio_chunk_counts (`torch.Tensor`, *optional*): Number of feature chunks for each audio input.
            audio_token_lengths (`torch.Tensor`, *optional*): Expected encoded token count for each audio input.
            chunk_audio_indices (`torch.Tensor`, *optional*): Audio ownership metadata emitted by the processor.
        """
        del chunk_audio_indices
        has_multimodal_inputs = any(value is not None for value in (pixel_values, pixel_values_videos, input_features))
        if has_multimodal_inputs:
            if input_ids is None or inputs_embeds is not None:
                raise ValueError("multimodal inputs require input_ids and do not accept inputs_embeds")
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                if image_grid_thw is None:
                    raise ValueError("image_grid_thw is required when pixel_values is provided")
                image_embeddings = self.get_image_features(pixel_values, image_grid_thw)
                inputs_embeds = self._merge_multimodal_embeddings(
                    input_ids,
                    inputs_embeds,
                    image_embeddings,
                    self.config.image_token_id,
                    "image",
                )

            if pixel_values_videos is not None:
                if video_grid_thw is None:
                    raise ValueError("video_grid_thw is required when pixel_values_videos is provided")
                video_embeddings = self.get_video_features(pixel_values_videos, video_grid_thw)
                inputs_embeds = self._merge_multimodal_embeddings(
                    input_ids,
                    inputs_embeds,
                    video_embeddings,
                    self.config.video_token_id,
                    "video",
                )

            if input_features is not None:
                required_audio_inputs = {
                    "chunk_sample_lengths": chunk_sample_lengths,
                    "chunk_token_lengths": chunk_token_lengths,
                    "audio_chunk_counts": audio_chunk_counts,
                }
                missing = [name for name, value in required_audio_inputs.items() if value is None]
                if missing:
                    raise ValueError(f"missing audio model inputs: {', '.join(missing)}")
                audio_embeddings, tower_token_lengths = self.get_audio_features(
                    input_features,
                    chunk_sample_lengths,
                    chunk_token_lengths,
                    audio_chunk_counts,
                )
                if audio_token_lengths is not None and not torch.equal(
                    audio_token_lengths.cpu(), tower_token_lengths.cpu()
                ):
                    raise ValueError(
                        "processor/audio encoder token lengths differ: "
                        f"{audio_token_lengths.tolist()} != {tower_token_lengths.tolist()}"
                    )
                inputs_embeds = self._merge_multimodal_embeddings(
                    input_ids,
                    inputs_embeds,
                    audio_embeddings,
                    self.config.audio_token_id,
                    "audio",
                )
            input_ids = None

        return self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


@auto_docstring
class Dots3NoteForConditionalGeneration(Dots3NoteTextForCausalLM):
    input_modalities = ("image", "video", "audio", "text")
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    _no_split_modules = ["Dots3NoteAudioEncoderLayer", "Dots3NoteTextDecoderLayer", "Dots3NoteVisionBlock"]

    def __init__(self, config: Dots3NoteConfig):
        super().__init__(config)
        self.model = Dots3NoteModel(config)


@auto_docstring
class Dots3NoteForCausalLM(Dots3NoteForConditionalGeneration):
    """Compatibility name recorded in the original multimodal checkpoint's architectures."""


@auto_docstring
class Dots3NoteImageProcessor(Qwen2VLImageProcessorPil):
    """Qwen2-VL PIL preprocessing with Dots image defaults and white-background RGBA compositing."""

    size = {"shortest_edge": 56 * 56, "longest_edge": (36 * 28) ** 2}
    temporal_patch_size = 1

    def convert_to_rgb(self, image):
        if isinstance(image, Image.Image) and image.mode == "RGBA":
            from ..idefics2 import image_processing_pil_idefics2

            return image_processing_pil_idefics2.convert_to_rgb(image)
        return convert_to_rgb(image)


_ALIGN = 28
_MIN_FRAMES = 4
_PF_FLOOR = 128
_PF_CEIL = 1024
_FPS_CAP = 1.0
_FPS_MIN = 0.2
_FRAME_OVERHEAD = 15
_BUDGET_OVERHEAD = 2240
_INTERLEAVE_MIN_SECONDS = 1.0
_AUDIO_SAMPLE_RATE = 16_000
_AUDIO_SAMPLES_PER_TOKEN = 1_280
_AUDIO_CHUNK_SECONDS = 30
_DEFAULT_SEQUENCE_LENGTH = 524_288


class Dots3NoteVideoProcessor(Qwen2VLVideoProcessor):
    """Video processor providing the native Dots 3 Note Preview timestamped image/audio expansion."""

    size = {"shortest_edge": 56 * 56, "longest_edge": (36 * 28) ** 2}
    temporal_patch_size = 1

    def preprocess_native(self, video, **kwargs) -> list[dict]:
        return preprocess_dots3_note_video(video, **kwargs)


def _resolve_video_budget(
    tokenizer, sequence_length: int | None, output_reserve: int | None, max_new_tokens: int
) -> tuple[int, int]:
    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if tokenizer_max_length is None or tokenizer_max_length > LARGE_INTEGER:
        tokenizer_max_length = _DEFAULT_SEQUENCE_LENGTH
    if sequence_length is None:
        sequence_length = tokenizer_max_length
    elif sequence_length > tokenizer_max_length:
        raise ValueError(
            f"sequence_length must not exceed tokenizer.model_max_length ({tokenizer_max_length}), "
            f"got {sequence_length}"
        )
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if output_reserve is not None and output_reserve < 0:
        raise ValueError(f"output_reserve must be non-negative, got {output_reserve}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    effective_reserve = max(sequence_length // 4 if output_reserve is None else output_reserve, max_new_tokens)
    if effective_reserve >= sequence_length:
        raise ValueError("output_reserve/max_new_tokens must leave room for video input")
    return sequence_length, effective_reserve


def _compute_target_size(
    orig_height: int, orig_width: int, min_pixels: int, max_pixels: int, factor: int = 28
) -> tuple[int, int]:
    height, width = smart_resize(orig_height, orig_width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    if height * width > max_pixels:
        height, width = smart_resize(height, width, factor=factor, min_pixels=0, max_pixels=max_pixels)
    return height, width


def _real_patches_at(orig_height: int, orig_width: int, patch_cap: int) -> int:
    height, width = _compute_target_size(
        orig_height,
        orig_width,
        _PF_FLOOR * _ALIGN * _ALIGN,
        max(_PF_FLOOR, patch_cap) * _ALIGN * _ALIGN,
    )
    return (height // _ALIGN) * (width // _ALIGN)


def _solve_degrade(
    visual_budget: int,
    duration: float,
    orig_height: int,
    orig_width: int,
    orig_fps: float,
    sequence_length: int,
) -> tuple[int, int]:
    aligned_height = max(_ALIGN, round(orig_height / _ALIGN) * _ALIGN)
    aligned_width = max(_ALIGN, round(orig_width / _ALIGN) * _ALIGN)
    original_patch_cap = (aligned_height // _ALIGN) * (aligned_width // _ALIGN)
    fps_cap = min(_FPS_CAP, max(orig_fps, 1e-6))
    patch_cap = min(_PF_CEIL, max(original_patch_cap, _PF_FLOOR))
    required = max(1, (sequence_length - _BUDGET_OVERHEAD) // (_PF_FLOOR + _FRAME_OVERHEAD))
    frame_cap = max(1024, 1 << (required - 1).bit_length())

    def usage(scale: float) -> tuple[int, int, int]:
        fps = _FPS_MIN + scale * (fps_cap - _FPS_MIN)
        candidate_patch_cap = _PF_FLOOR + scale * (patch_cap - _PF_FLOOR)
        num_frames = max(_MIN_FRAMES, min(int(round(duration * fps)), frame_cap))
        patches = _real_patches_at(orig_height, orig_width, int(round(candidate_patch_cap)))
        return num_frames * (patches + _FRAME_OVERHEAD), int(round(candidate_patch_cap)), num_frames

    cost, candidate_patch_cap, num_frames = usage(1.0)
    if cost <= visual_budget:
        return num_frames, candidate_patch_cap

    floor_cost = _real_patches_at(orig_height, orig_width, _PF_FLOOR) + _FRAME_OVERHEAD
    if usage(0.0)[0] > visual_budget:
        return max(_MIN_FRAMES, min(visual_budget // floor_cost, frame_cap)), _PF_FLOOR

    low, high = 0.0, 1.0
    for _ in range(50):
        middle = (low + high) / 2
        if usage(middle)[0] <= visual_budget:
            low = middle
        else:
            high = middle
    _, candidate_patch_cap, num_frames = usage(low)
    return num_frames, candidate_patch_cap


def _audio_tokens(duration: float, sample_rate: int) -> int:
    if duration <= 0:
        return 0
    return (
        compute_audio_token_length(
            int(duration * sample_rate),
            chunk_samples=_AUDIO_CHUNK_SECONDS * sample_rate,
            token_stride=_AUDIO_SAMPLES_PER_TOKEN,
        )
        + 2
    )


def _decode_audio(video_bytes: bytes, sample_rate: int) -> tuple[np.ndarray | None, float]:
    requires_backends(_decode_audio, ["torchcodec"])
    from torchcodec.decoders import AudioDecoder

    try:
        samples = AudioDecoder(video_bytes, sample_rate=sample_rate, num_channels=1).get_all_samples()
    except Exception as error:
        logger.warning("The video audio track could not be decoded and will be omitted: %s", error)
        return None, 0.0
    waveform = samples.data
    if waveform is None or waveform.numel() == 0:
        logger.warning("The video has no decodable audio samples and will be processed without audio")
        return None, 0.0
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform[0]
    pcm = (np.clip(waveform.cpu().numpy(), -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm, float(pcm.shape[0]) / sample_rate


def _open_video(video_bytes: bytes):
    requires_backends(_open_video, ["torchcodec"])
    from torchcodec.decoders import VideoDecoder

    try:
        return VideoDecoder(video_bytes, dimension_order="NHWC", num_ffmpeg_threads=1, seek_mode="approximate")
    except TypeError:
        return VideoDecoder(video_bytes, dimension_order="NHWC", num_ffmpeg_threads=1)


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _decode_frames(
    decoder,
    visual_budget: int,
    sequence_length: int,
    jpeg_quality: int,
) -> tuple[list[tuple[float, Image.Image]], float]:
    metadata = decoder.metadata
    duration = float(metadata.duration_seconds or 0)
    orig_height = int(metadata.height)
    orig_width = int(metadata.width)
    total_frames = int(getattr(metadata, "num_frames", 0) or 0)
    orig_fps = float(getattr(metadata, "average_fps", 0) or 0) or 25.0
    if duration <= 0 or orig_height <= 0 or orig_width <= 0:
        raise ValueError(f"Invalid video metadata: duration={duration}, height={orig_height}, width={orig_width}")
    if total_frames <= 0:
        total_frames = max(1, int(duration * orig_fps))

    num_frames, patch_cap = _solve_degrade(visual_budget, duration, orig_height, orig_width, orig_fps, sequence_length)
    aligned_height = max(_ALIGN, round(orig_height / _ALIGN) * _ALIGN)
    aligned_width = max(_ALIGN, round(orig_width / _ALIGN) * _ALIGN)
    original_patch_cap = (aligned_height // _ALIGN) * (aligned_width // _ALIGN)
    target_height, target_width = _compute_target_size(
        orig_height,
        orig_width,
        _PF_FLOOR * _ALIGN * _ALIGN,
        min(patch_cap, original_patch_cap) * _ALIGN * _ALIGN,
    )
    num_frames = max(_MIN_FRAMES, min(num_frames, total_frames))
    step = (total_frames - 1) / (num_frames - 1) if num_frames > 1 else 0
    indices = sorted({max(0, min(int(round(index * step)), total_frames - 1)) for index in range(num_frames)})

    try:
        decoded = decoder.get_frames_at(indices=indices).data
    except (IndexError, RuntimeError):
        safe_indices = [index for index in indices if index < total_frames]
        while safe_indices and safe_indices[-1] > 0:
            try:
                decoded = decoder.get_frames_at(indices=safe_indices).data
                break
            except (IndexError, RuntimeError):
                safe_indices = safe_indices[:-1]
        else:
            raise

    actual_fps = round(len(decoded) / max(duration, 1e-6), 4)
    frames = []
    for frame_number, frame in enumerate(decoded):
        image = Image.fromarray(np.asarray(to_numpy(frame)))
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        # SGLang's train flattener recomputes timestamps from the sampled-frame index
        # and the rounded effective FPS instead of preserving source-frame timestamps.
        timestamp = round(frame_number / actual_fps, 3)
        frames.append((timestamp, _jpeg_roundtrip(image, jpeg_quality)))
    return frames, duration


def _prepare_decoded_frames(
    video,
    visual_budget: int,
    sequence_length: int,
    jpeg_quality: int,
) -> tuple[list[tuple[float, Image.Image]], float]:
    metadata = None
    if isinstance(video, tuple):
        video, metadata = video
    frames = (
        np.stack([np.asarray(to_numpy(frame)) for frame in video])
        if isinstance(video, (list, tuple))
        else np.asarray(to_numpy(video))
    )
    if frames.ndim == 4 and frames.shape[1] in (3, 4) and frames.shape[-1] not in (3, 4):
        frames = frames.transpose(0, 2, 3, 1)
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise TypeError("Decoded Dots 3 Note Preview video must have shape (frames, height, width, channels)")
    if not np.issubdtype(frames.dtype, np.integer):
        frames = np.clip(frames * 255.0 if frames.max(initial=0) <= 1.0 else frames, 0, 255).astype(np.uint8)
    fps = float((metadata or {}).get("fps", 1.0)) if metadata else 1.0
    fps = max(fps, 1e-6)
    duration = len(frames) / fps
    orig_height, orig_width = frames.shape[1:3]
    num_frames, patch_cap = _solve_degrade(visual_budget, duration, orig_height, orig_width, fps, sequence_length)
    num_frames = min(max(1, num_frames), len(frames))
    indices = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    target_height, target_width = _compute_target_size(
        orig_height, orig_width, _PF_FLOOR * _ALIGN * _ALIGN, patch_cap * _ALIGN * _ALIGN
    )
    selected_indices = sorted(set(indices.tolist()))
    actual_fps = round(len(selected_indices) / max(duration, 1e-6), 4)
    output = []
    for frame_number, index in enumerate(selected_indices):
        image = Image.fromarray(frames[index]).convert("RGB")
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        output.append((round(frame_number / actual_fps, 3), _jpeg_roundtrip(image, jpeg_quality)))
    return output, duration


def _format_timestamp(seconds: float) -> str:
    centiseconds = int(round(max(seconds, 0.0) * 100))
    hours = centiseconds // 360_000
    minutes = (centiseconds // 6_000) % 60
    secs = (centiseconds // 100) % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{centiseconds % 100:02d}"


def _group_bounds(num_frames: int, duration: float, mode: str, rng: random.Random) -> list[int]:
    if num_frames <= 1 or duration <= 0:
        return [0, num_frames]
    max_groups = min(num_frames, max(1, int(duration // _INTERLEAVE_MIN_SECONDS)))
    if mode == "whole" or max_groups <= 1:
        groups = 1
    elif mode == "eval30":
        groups = round(math.sqrt(max_groups))
    elif mode == "eval_ek":
        groups = round((max_groups - 1) / math.log(max_groups))
    elif mode == "logk":
        groups = round(math.exp(rng.uniform(0.0, math.log(max_groups))))
    else:
        raise ValueError(f"Unsupported video k_mode: {mode}")
    groups = max(1, min(max_groups, groups))
    if groups == 1:
        return [0, num_frames]
    if mode == "logk":
        cuts = sorted(rng.sample(range(1, num_frames), groups - 1))
    else:
        cuts = sorted(
            {
                round(index * num_frames / groups)
                for index in range(1, groups)
                if 0 < round(index * num_frames / groups) < num_frames
            }
        )
    return [0, *cuts, num_frames]


def _read_video_bytes(video) -> bytes | None:
    if isinstance(video, (bytes, bytearray)):
        return bytes(video)
    if isinstance(video, Path):
        return video.read_bytes()
    if not isinstance(video, str):
        return None
    if video.startswith(("http://", "https://")):
        with urlopen(video, timeout=30) as response:
            return response.read()
    if video.startswith("data:"):
        return base64.b64decode(video.split(",", 1)[1])
    path = Path(video)
    if path.is_file():
        return path.read_bytes()
    try:
        return base64.b64decode(video, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("video string must be a path, URL, data URI, or base64 payload") from error


def preprocess_dots3_note_video(
    video,
    *,
    tokenizer,
    question: str = "",
    sequence_length: int | None = None,
    output_reserve: int | None = None,
    audio_cap: float = 1.0,
    audio_sample_rate: int = _AUDIO_SAMPLE_RATE,
    k_mode: str = "eval_ek",
    max_new_tokens: int = 0,
    jpeg_quality: int = 85,
) -> list[dict]:
    """Expand one video into SGLang-compatible timestamped image/audio parts."""
    sequence_length, effective_reserve = _resolve_video_budget(
        tokenizer, sequence_length, output_reserve, max_new_tokens
    )
    if audio_cap < 0:
        raise ValueError(f"audio_cap must be non-negative, got {audio_cap}")
    if audio_sample_rate <= 0:
        raise ValueError(f"audio_sample_rate must be positive, got {audio_sample_rate}")
    if k_mode not in {"logk", "eval30", "eval_ek", "whole"}:
        raise ValueError(f"Unsupported video k_mode: {k_mode}")

    input_length = sequence_length - effective_reserve
    minimum_input_length = _BUDGET_OVERHEAD + _MIN_FRAMES * (_PF_FLOOR + _FRAME_OVERHEAD)
    if input_length < minimum_input_length:
        raise ValueError(
            f"output_reserve/max_new_tokens must leave at least {minimum_input_length} tokens for video input"
        )
    video_bytes = _read_video_bytes(video)
    decoder = _open_video(video_bytes) if video_bytes is not None else None
    video_duration_hint = float(decoder.metadata.duration_seconds or 0) if decoder is not None else 0.0
    pcm = None
    audio_duration = 0.0
    if video_bytes is not None and audio_cap > 0:
        pcm, audio_duration = _decode_audio(video_bytes, audio_sample_rate)

    audio_token_count = _audio_tokens(audio_duration, audio_sample_rate) if pcm is not None else 0
    precheck_frame_bound = max(1, int(audio_duration * _FPS_CAP))
    precheck_groups = min(precheck_frame_bound, max(1, int(audio_duration // _INTERLEAVE_MIN_SECONDS)))
    precheck_audio_tokens = audio_token_count + 3 * precheck_groups if pcm is not None else 0
    minimum_visual_tokens = _MIN_FRAMES * (_PF_FLOOR + _FRAME_OVERHEAD)
    if audio_token_count > audio_cap * input_length:
        logger.warning("The video audio track exceeds audio_cap and will be omitted")
        pcm = None
        audio_duration = 0.0
    elif precheck_audio_tokens + minimum_visual_tokens + _BUDGET_OVERHEAD > input_length:
        logger.warning("The video audio track does not fit alongside the minimum visual input and will be omitted")
        pcm = None
        audio_duration = 0.0

    frame_upper_bound = max(1, int(video_duration_hint * _FPS_CAP))
    max_groups = min(frame_upper_bound, max(1, int(audio_duration // _INTERLEAVE_MIN_SECONDS)))
    reserved_audio_tokens = audio_token_count + 3 * max_groups if pcm is not None else 0

    overhead = (
        len(tokenizer.encode("<|system|>You are a helpful assistant.<|endofsystem|>\n", add_special_tokens=False))
        + 2
        + len(tokenizer.encode("<video_0>", add_special_tokens=False))
        + 64
    )
    visual_budget = max(_PF_FLOOR + _FRAME_OVERHEAD, input_length - overhead - reserved_audio_tokens)
    if video_bytes is not None:
        frames, _ = _decode_frames(decoder, visual_budget, input_length, jpeg_quality)
    else:
        frames, _ = _prepare_decoded_frames(video, visual_budget, input_length, jpeg_quality)

    if pcm is None:
        output = []
        for timestamp, image in frames:
            output.append({"type": "text", "text": f"<{_format_timestamp(timestamp)}>"})
            output.append({"type": "image", "image": image})
        return output

    video_id = hashlib.sha1(video_bytes, usedforsecurity=False).hexdigest()
    record_key = hashlib.sha1(f"{video_id}|{question}".encode(), usedforsecurity=False).hexdigest()
    seed = hashlib.sha1(f"42|flatten|{record_key}".encode(), usedforsecurity=False).hexdigest()
    rng = random.Random(int(seed[:8], 16))
    bounds = _group_bounds(len(frames), audio_duration, k_mode, rng)
    output = []
    for group in range(len(bounds) - 1):
        start, end = bounds[group], bounds[group + 1]
        if end <= start:
            continue
        start_time = 0.0 if group == 0 else frames[start][0]
        end_time = audio_duration if group == len(bounds) - 2 else frames[end][0]
        if end_time <= start_time:
            end_time = start_time + audio_duration / max(1, len(bounds) - 1)
        for timestamp, image in frames[start:end]:
            output.append({"type": "text", "text": f"<{_format_timestamp(timestamp)}>"})
            output.append({"type": "image", "image": image})
        sample_start = max(0, int(round(start_time * audio_sample_rate)))
        sample_end = min(len(pcm), int(round(end_time * audio_sample_rate)))
        if sample_end > sample_start:
            waveform = np.ascontiguousarray(pcm[sample_start:sample_end].astype(np.float32) / 32768.0)
            output.append({"type": "audio", "audio": waveform})
    return output


__all__ = [
    "Dots3NoteImageProcessor",
    "Dots3NoteVideoProcessor",
    "Dots3NoteAudioModel",
    "Dots3NoteAudioPreTrainedModel",
    "Dots3NoteForCausalLM",
    "Dots3NoteForConditionalGeneration",
    "Dots3NoteModel",
    "Dots3NotePreTrainedModel",
    "Dots3NoteTextForCausalLM",
    "Dots3NoteTextModel",
    "Dots3NoteVisionModel",
    "Dots3NoteVisionPreTrainedModel",
]
