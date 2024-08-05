# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Qwen2-VL model."""

import inspect
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, LayerNorm

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from .configuration_qwen2_vl import Qwen2VLConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

    from ...modeling_flash_attention_utils import _flash_attention_forward

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None


logger = logging.get_logger(__name__)


@dataclass
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2VL causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        state (`torch.FloatTensor`, *optional*):
            Attentions mask, used to update attention mask and position_ids.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(seqlens, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    state: Optional[torch.FloatTensor] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, position_ids, mrope_section=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal Sections for t,h,w in Multimodal inputs
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    if mrope_section:
        cos = cos[position_ids]
        sin = sin[position_ids]
        mrope_section = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )
    else:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    if apply_rotary_emb is None:
        t_ = t.float()
        cos = freqs.cos()
        sin = freqs.sin()
        cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        output = (t_ * cos) + (rotate_half(t_) * sin)
        output = output.type_as(t)
    else:
        t_ = t.float()
        cos = freqs.cos()
        sin = freqs.sin()
        output = apply_rotary_emb(t_, cos, sin).type_as(t)
    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (
                self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device) / self.dim)
            )
            seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seqlen = x.shape[0]
        x = x.view(seqlen, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(seqlen, self.embed_dim)
        return x


class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN["quick_gelu"]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, use_flash_attention: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.use_flash_attention = use_flash_attention
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor = None) -> torch.Tensor:
        L, _ = x.shape
        q, k, v = self.qkv(x).reshape(L, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
            k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        if (
            flash_attn_varlen_func is not None
            and q.dtype in [torch.float16, torch.bfloat16]
            and self.use_flash_attention
        ):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            x = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
            x = x.reshape(L, -1)
        else:
            attention_mask = torch.zeros([1, L, L], device=q.device, dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            x = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0).transpose(0, 1).reshape(L, -1)
        x = self.proj(x)
        return x


class Qwen2VLVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        norm_layer: nn.Module = partial(LayerNorm, eps=1e-6),
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.attn = VisionAttention(dim, num_heads=num_heads, use_flash_attention=use_flash_attention)
        self.mlp = VisionMlp(dim=dim, hidden_dim=mlp_hidden_dim)

    def forward(self, x, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 378,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        in_chans: int = 3,
        hidden_size: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = partial(LayerNorm, eps=1e-6),
        use_flash_attention: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.spatial_merge_size = spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen2VLVisionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_layer=norm_layer,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(depth)
            ]
        )
        self.merger = PatchMerger(dim=hidden_size, context_dim=embed_dim)

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(x)


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (indices, cu_seqlens, max_seqlen_in_batch)


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
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


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2MLP
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


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


class Qwen2VLAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if self.rope_scaling and "mrope_section" in self.rope_scaling:
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids, self.rope_scaling["mrope_section"]
            )
        else:
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2VLFlashAttention2(Qwen2VLAttention):
    """
    Qwen2 flash attention module, following Qwen2 attention module. This module inherits from `Qwen2VLAttention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        if self.rope_scaling and "mrope_section" in self.rope_scaling:
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids, self.rope_scaling["mrope_section"]
            )
        else:
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and self.config.use_sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2VLSdpaAttention(Qwen2VLAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2VLModel is using Qwen2VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        if self.rope_scaling and "mrope_section" in self.rope_scaling:
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids, self.rope_scaling["mrope_section"]
            )
        else:
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


QWEN2_VL_ATTENTION_CLASSES = {
    "eager": Qwen2VLAttention,
    "flash_attention_2": Qwen2VLFlashAttention2,
    "sdpa": Qwen2VLSdpaAttention,
}


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


QWEN2VL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2VLConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2VL Model outputting raw hidden-states without any specific head on top.",
    QWEN2VL_START_DOCSTRING,
)
class Qwen2VLPreTrainedModel(PreTrainedModel):
    config_class = Qwen2VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@add_start_docstrings(
    "The bare Qwen2VL Model outputting raw hidden-states without any specific head on top.",
    QWEN2VL_START_DOCSTRING,
)
class Qwen2VLModel(Qwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
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
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class Qwen2VLForConditionalGeneration(Qwen2VLPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2VisionTransformer(**config.vision_config)
        self.model = Qwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"  # set it to left by default, user can use setter to change padding_sides

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        vision_grid_thw: List[List[int]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        spatial_merge_size = self.config.vision_config["spatial_merge_size"]
        vision_token_id = self.config.vision_token_id
        mrope_position_deltas = []
        if vision_grid_thw is not None:
            total_input_ids = input_ids
            total_llm_positions = []
            v_index = 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                img_num = (input_ids == vision_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                for _ in range(img_num):
                    t, h, w = vision_grid_thw[v_index][0], vision_grid_thw[v_index][1], vision_grid_thw[v_index][2]
                    ed = input_tokens.index(vision_token_id, st)
                    llm_grid_t = t
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + 1
                    v_index += 1

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                total_llm_positions.append(llm_positions)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            total_llm_positions = torch.cat(total_llm_positions, dim=1).to(input_ids.device)
            return total_llm_positions, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        feature_lens: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_token_index: int = None,
        ignore_index: int = -100,
    ):
        """
        Merge input_ids with with image features into final embeddings

        Args:
            image_features (`torch.Tensor` of shape `(all_feature_lens, embed_dim)`):
                All vision vectors of all images in the batch
            feature_lens (`torch.LongTensor` of shape `(num_images)`):
                The length of visual embeddings of each image as stacked in `image_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with visual embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with image token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                :abels need to be recalculated to support training (if provided)
            image_token_index (`int`, *optional*)
                Token id used to indicate the special "image" token. Defaults to `config.image_token_index`
            ignore_index (`int`, *optional*)
                Value that is used to pad `labels` and will be ignored when calculated loss. Default: -100.
        Returns:
            final_embedding, final_attention_mask, position_ids, final_labels

        Explanation:
            each image has variable length embeddings, with length specified by feature_lens
            image_features is concatenation of all visual embed vectors
            task: fill each <image> with the correct number of visual embeddings
            Example:
                X (5 patches), Y (3 patches), Z (8)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but image token sizes are different, then cannot infer left or right padding
                ```python
                cat_img = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
                chart_img = Image.open(requests.get("https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true", stream=True).raw)
                prompts = [
                    "[INST] <image>\nWhat is shown in this image? [/INST]",
                    "[INST] <image>\nWhat is shown in this image? [/INST]",
                ]
                inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")
                    chart_img has 2634 tokens, while cat_img has 2340 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        image_features = image_features.to(inputs_embeds.device)
        feature_lens = feature_lens.to(inputs_embeds.device)
        position_ids = position_ids.to(inputs_embeds.device)
        image_token_index = image_token_index if image_token_index is not None else self.config.vision_token_id
        ignore_index = ignore_index if ignore_index is not None else self.config.ignore_index

        with torch.no_grad():
            num_images = feature_lens.size(0)
            num_image_features, embed_dim = image_features.shape
            if feature_lens.sum() != num_image_features:
                raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
            batch_size = input_ids.shape[0]
            _left_padding = torch.any(attention_mask[:, 0] == 0)
            _right_padding = torch.any(attention_mask[:, -1] == 0)

            left_padding = True
            if batch_size > 1:
                if _left_padding and not _right_padding:
                    left_padding = True
                elif not _left_padding and _right_padding:
                    left_padding = False
                elif not _left_padding and not _right_padding:
                    # both side is 1, so cannot tell
                    left_padding = self.padding_side == "left"
                else:
                    # invalid attention_mask
                    raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

            # Whether to turn off right padding
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == image_token_index
            # special_image_token_mask: [bsz, seqlen]
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # num_special_image_tokens: [bsz]
            # Reserve for padding of num_images
            total_num_special_image_tokens = torch.sum(special_image_token_mask)
            if total_num_special_image_tokens != num_images:
                raise ValueError(
                    f"Number of image tokens in input_ids ({total_num_special_image_tokens}) different from num_images ({num_images})."
                )
            # Compute the maximum embed dimension
            # max_image_feature_lens is max_feature_lens per batch
            feature_lens = feature_lens.to(input_ids.device)
            feature_lens_batch = feature_lens.split(num_special_image_tokens.tolist(), dim=0)
            feature_lens_batch_sum = torch.tensor([x.sum() for x in feature_lens_batch], device=input_ids.device)
            embed_sequence_lengths = (
                (attention_mask == 1).long().sum(-1) - num_special_image_tokens + feature_lens_batch_sum
            )
            max_embed_dim = embed_sequence_lengths.max()

            batch_indices, non_image_indices = torch.where((input_ids != image_token_index) & (attention_mask == 1))
            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            # ! instead of special_image_token_mask * (num_image_patches - 1)
            #   special_image_token_mask * (num_feature_len - 1)
            special_image_token_mask = special_image_token_mask.long()
            special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
            new_token_positions = torch.cumsum((special_image_token_mask + 1), -1) - 1
            if left_padding:
                # shift right token positions so that they are ending at the same number
                # the below here was incorrect? new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
                new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_embed_dim), self.config.eos_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )
        final_position_ids = torch.full(
            (3, batch_size, max_embed_dim), 1, dtype=input_ids.dtype, device=inputs_embeds.device
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_image_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full_like(final_attention_mask, ignore_index).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        with torch.no_grad():
            image_to_overwrite = torch.full(
                (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
            embed_indices = embed_indices.expand(batch_size, max_embed_dim)
            embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

            if left_padding:
                # exclude padding on the left
                max_embed_dim = max_embed_dim.to(target_device)
                val = (max_embed_dim - embed_indices) <= embed_seq_lens
            else:
                # exclude padding on the right
                val = embed_indices < embed_seq_lens
            image_to_overwrite &= val

            if image_to_overwrite.sum() != num_image_features:
                raise ValueError(
                    f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                    f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. "
                    f"This prevents correct indexing and breaks batch generation."
                )
        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim)
        final_attention_mask |= image_to_overwrite
        final_position_ids[:, final_attention_mask == 1] = position_ids

        return final_embedding, final_attention_mask, final_position_ids, final_labels, final_input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        vision_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> processor = AutoProcessor.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> messages = [{"role": "user", "content": "Hey, are you conscious? Can you talk to me?"}]
        >>> text, vision_infos= processor.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True,add_vision_id=False)
        >>> inputs = processor(text=[text], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_embeds = None
        if inputs_embeds is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(input_ids, vision_grid_thw, attention_mask)
            else:
                B, L = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = (
                    torch.arange(L, device=input_ids.device)
                    .view(1, -1)
                    .expand(B, -1)
                    .add(delta)
                    .unsqueeze(0)
                    .expand(3, -1, -1)
                )
            if pixel_values is not None and input_ids.shape[1] != 1 and pixel_values.size(0) > 0:
                vision_seqlens = vision_grid_thw.prod(dim=-1) // (self.config.vision_config["spatial_merge_size"] ** 2)
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=vision_grid_thw)
                inputs_embeds = self.model.embed_tokens(input_ids)
                inputs_embeds, attention_mask, position_ids, labels, _ = self._merge_input_ids_with_image_features(
                    image_embeds,
                    vision_seqlens,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                    position_ids,
                    labels=labels,
                )
            else:
                inputs_embeds = self.model.embed_tokens(input_ids)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)
        else:
            B, L = inputs_embeds.shape[:2]
            position_ids = torch.arange(L, device=inputs_embeds.device).view(1, 1, -1).expand(3, B, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            state=attention_mask,
            vision_hidden_states=image_embeds,
            rope_deltas=rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        pixel_values=None,
        vision_grid_thw=None,
        **kwargs,
    ):
        past_length = 0
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            # Past key values are always initialized with a `Cache` object -> no need for if-else anymore
            past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        input_length = input_ids.shape[-1]
        if past_length != 0 and kwargs.get("state", None) is not None:
            attention_mask = kwargs["state"]
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], input_length))], dim=-1
            )
            cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_length == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        rope_deltas = kwargs["rope_deltas"] if kwargs.get("rope_deltas", None) is not None else None

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "pixel_values": pixel_values,
                "vision_grid_thw": vision_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
