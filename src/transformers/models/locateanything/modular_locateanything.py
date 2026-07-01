# coding=utf-8
# Copyright 2026 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch LocateAnything model."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, is_flash_attn_2_available, logging
from ...utils.generic import maybe_autocast
from ..qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm, apply_rotary_pos_emb, repeat_kv
from .configuration_locateanything import LocateAnythingConfig, LocateAnythingVisionConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None


logger = logging.get_logger(__name__)


# ====================================================================================================
# Vision encoder (MoonViT)
# ====================================================================================================


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def apply_rope_vision(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the 2D rotary position embedding to the query and key tensors of the vision encoder."""
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def vision_flash_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    if flash_attn_varlen_func is None:
        return vision_sdpa_attention(q, k, v, q_cu_seqlens=q_cu_seqlens, k_cu_seqlens=k_cu_seqlens)
    max_seqlen_q = (q_cu_seqlens[1:] - q_cu_seqlens[:-1]).max().item()
    max_seqlen_k = (k_cu_seqlens[1:] - k_cu_seqlens[:-1]).max().item()
    attn_out = flash_attn_varlen_func(
        q, k, v, q_cu_seqlens, k_cu_seqlens, max_seqlen_q, max_seqlen_k, causal=False
    )
    return attn_out.flatten(start_dim=-2)


def vision_sdpa_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    seq_length = q.shape[0]
    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[..., q_cu_seqlens[i - 1] : q_cu_seqlens[i], q_cu_seqlens[i - 1] : q_cu_seqlens[i]] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
    attn_output = attn_output.transpose(0, 1)
    return attn_output.reshape(seq_length, -1)


def vision_eager_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
    seq_length = q.shape[0]
    attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[..., q_cu_seqlens[i - 1] : q_cu_seqlens[i], q_cu_seqlens[i - 1] : q_cu_seqlens[i]] = True
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight = attn_weight.masked_fill(~attention_mask, torch.finfo(attn_weight.dtype).min)
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    return attn_output.reshape(seq_length, -1)


VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": vision_flash_attention,
    "sdpa": vision_sdpa_attention,
    "eager": vision_eager_attention,
}


class LocateAnythingVisionRotaryEmbedding(nn.Module):
    """2D rotary position embedding with multi-resolution support for the MoonViT encoder."""

    def __init__(self, dim: int, max_height: int = 512, max_width: int = 512, theta_base: float = 10000):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("`dim` must be divisible by 4 for the 2D rotary position embedding")
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self.freqs_cis = None

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        n = self.max_height * self.max_width
        flat_pos = torch.arange(0, n).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()
        y_freqs = torch.outer(y_pos, freqs).float()
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def forward(self, grid_hws: torch.Tensor) -> torch.Tensor:
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_freqs_cis(grid_hws.device)
        shapes = grid_hws.tolist()
        freqs_cis = torch.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2) for h, w in shapes], dim=0
        )
        return freqs_cis


class LocateAnythingLearnable2DInterpPosEmb(nn.Module):
    def __init__(self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic") -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for shape in grid_hws.tolist():
            if shape == list(self.weight.shape[:-1]):
                pos_embs.append(self.weight.flatten(end_dim=1))
            else:
                pos_embs.append(
                    F.interpolate(
                        self.weight.permute((2, 0, 1)).unsqueeze(0),
                        size=shape,
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute((1, 2, 0))
                    .flatten(end_dim=1)
                )
        return x + torch.cat(pos_embs)


class LocateAnythingVisionPatchEmbed(nn.Module):
    def __init__(self, out_dim: int, in_dim: int = 3, patch_size: int = 14, pos_emb_height: int = 64, pos_emb_width: int = 64):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_emb = LocateAnythingLearnable2DInterpPosEmb(height=pos_emb_height, width=pos_emb_width, dim=out_dim)

    def forward(self, x: torch.Tensor, grid_hws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        x = self.pos_emb(x, grid_hws)
        return x


class LocateAnythingVisionMLP(nn.Module):
    def __init__(self, dims: list[int], activation):
        super().__init__()
        self.fc0 = nn.Linear(dims[0], dims[1], bias=True)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=True)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.activation(self.fc0(x)))


class LocateAnythingVisionLayer(GradientCheckpointingLayer):
    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_dim = config.hidden_size
        self.head_dim = self.hidden_dim // self.num_heads
        self.config = config

        self.norm0 = nn.LayerNorm(self.hidden_dim)
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.mlp = LocateAnythingVisionMLP(
            [self.hidden_dim, config.intermediate_size, self.hidden_dim], ACT2FN["gelu_pytorch_tanh"]
        )
        self.wqkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=True)
        self.wo = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

    def attention(self, x: torch.Tensor, cu_seqlens: torch.Tensor, rope_freqs_cis: torch.Tensor) -> torch.Tensor:
        xqkv = self.wqkv(x)
        qkv_shape = xqkv.size()[:-1] + (3, self.num_heads, self.head_dim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)
        xq, xk = apply_rope_vision(xq, xk, rope_freqs_cis)
        attn_func = VISION_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_out = attn_func(xq, xk, xv, q_cu_seqlens=cu_seqlens, k_cu_seqlens=cu_seqlens)
        return self.wo(attn_out)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rope_freqs_cis: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(self.norm0(hidden_states), cu_seqlens, rope_freqs_cis)
        hidden_states = hidden_states + self.mlp(self.norm1(hidden_states))
        return hidden_states


def patch_merger(x: torch.Tensor, grid_hws: torch.Tensor, merge_kernel_size: tuple[int, int] = (2, 2)) -> torch.Tensor:
    d_model = x.size(-1)
    outputs = []
    pre_sum = 0
    kernel_height, kernel_width = merge_kernel_size
    for x_shape in grid_hws.tolist():
        height, width = x_shape[0], x_shape[1]
        seq = x[pre_sum : pre_sum + height * width]
        new_height, new_width = height // kernel_height, width // kernel_width
        reshaped_seq = seq.view(new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 2, 1, 3, 4).contiguous()
        reshaped_seq = reshaped_seq.view(new_height * new_width, -1)
        outputs.append(reshaped_seq)
        pre_sum += height * width
    return torch.cat(outputs, dim=0)


@auto_docstring
class LocateAnythingVisionPreTrainedModel(PreTrainedModel):
    config: LocateAnythingVisionConfig
    base_model_prefix = "vision_model"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    supports_gradient_checkpointing = True
    _no_split_modules = ["LocateAnythingVisionLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True


@auto_docstring(
    custom_intro="The MoonViT vision encoder used by LocateAnything, producing merged patch features for the projector."
)
class LocateAnythingVisionModel(LocateAnythingVisionPreTrainedModel):
    def __init__(self, config: LocateAnythingVisionConfig):
        super().__init__(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size

        self.patch_embed = LocateAnythingVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
        )
        self.rotary_pos_emb = LocateAnythingVisionRotaryEmbedding(config.hidden_size // config.num_attention_heads)
        self.blocks = nn.ModuleList([LocateAnythingVisionLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_init()

    def forward(self, pixel_values: torch.Tensor, image_grid_hws: torch.Tensor, **kwargs) -> BaseModelOutput:
        r"""
        image_grid_hws (`torch.Tensor` of shape `(num_images, 2)`):
            The grid height and width (in patches) of each image, used to unpack the flattened patch sequence.
        """
        hidden_states = self.patch_embed(pixel_values, image_grid_hws)
        rope_freqs_cis = self.rotary_pos_emb(image_grid_hws)

        lengths = torch.cat(
            (
                torch.zeros(1, device=hidden_states.device, dtype=image_grid_hws.dtype),
                image_grid_hws[:, 0] * image_grid_hws[:, 1],
            )
        )
        cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, rope_freqs_cis)

        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = patch_merger(hidden_states, image_grid_hws, merge_kernel_size=self.merge_kernel_size)
        return BaseModelOutput(last_hidden_state=hidden_states)


# ====================================================================================================
# Multimodal projector
# ====================================================================================================


class LocateAnythingMultiModalProjector(nn.Module):
    def __init__(self, config: LocateAnythingConfig):
        super().__init__()
        vit_hidden_size = config.vision_config.hidden_size
        merge = config.vision_config.merge_kernel_size[0] * config.vision_config.merge_kernel_size[1]
        llm_hidden_size = config.text_config.hidden_size
        self.layer_norm = nn.LayerNorm(vit_hidden_size * merge)
        self.linear_1 = nn.Linear(vit_hidden_size * merge, llm_hidden_size)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(llm_hidden_size, llm_hidden_size)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(image_features)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# ====================================================================================================
# Block-diffusion attention masking utilities (Parallel Box Decoding)
# ====================================================================================================


def find_prefix_seq_length_by_pe(position_ids: torch.Tensor) -> torch.Tensor:
    batch_size, _ = position_ids.shape
    prev = position_ids[:, :-1]
    curr = position_ids[:, 1:]
    drop_mask = curr < prev
    out = torch.full((batch_size,), -1, dtype=torch.long)
    for b in range(batch_size):
        drop_pos = torch.nonzero(drop_mask[b], as_tuple=False)
        if drop_pos.numel() > 0:
            out[b] = drop_pos[0].item() + 1
    return out


def update_causal_mask_for_one_gen_window_2d(
    attn_mask_2d: torch.Tensor, block_size: int, use_cache: bool = True, causal_attn: bool = False
) -> torch.Tensor:
    """Make the last Parallel Box Decoding block (the diffusion window) bidirectional during inference."""
    if not causal_attn:
        attn_mask_2d[-block_size:, -block_size:] = 0.0
    if use_cache:
        attn_mask_2d[-block_size:, -block_size - 1] = torch.finfo(attn_mask_2d.dtype).min
    return attn_mask_2d


def create_block_diff_mask_by_pe_4d(
    block_size: int, x0_len_list: torch.Tensor, position_ids: torch.Tensor, causal_attn: bool = False, dtype=torch.float32
) -> torch.Tensor:
    """Generate a 4D block-difference attention mask used during training / full-sequence forward passes."""
    batch_size, seq_len = position_ids.shape
    device = position_ids.device

    q_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1)
    kv_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len)

    x0_len = x0_len_list.to(device).view(batch_size, 1, 1)
    x0_flag_q = q_idx < x0_len
    x0_flag_kv = kv_idx < x0_len

    q_block_idx = (q_idx - x0_len) // block_size
    kv_block_idx = (kv_idx - x0_len) // block_size

    block_causal = x0_flag_q & x0_flag_kv & (q_idx >= kv_idx)
    mutual_condition = (q_idx >= kv_idx) if causal_attn else torch.ones_like(q_idx, dtype=torch.bool)
    block_mutual = ~x0_flag_q & ~x0_flag_kv & (q_block_idx == kv_block_idx) & mutual_condition

    q_blk = torch.div(q_idx - x0_len, block_size, rounding_mode="floor")
    q_blk_start = (x0_len_list.to(device).view(batch_size, 1) + q_blk[:, :, 0] * block_size).clamp(min=0, max=seq_len - 1)
    prefix_len = position_ids.gather(1, q_blk_start).unsqueeze(2)
    block_prefix = (~x0_flag_q & x0_flag_kv) & (kv_idx < prefix_len)

    final_mask = block_causal | block_mutual | block_prefix
    customized_mask = torch.full_like(final_mask, torch.finfo(dtype).min, dtype=dtype)
    customized_mask.masked_fill_(final_mask, 0.0)
    return customized_mask.unsqueeze(1).to(device=device)


# ====================================================================================================
# Text decoder (Qwen2 backbone with block-diffusion attention)
# ====================================================================================================


class LocateAnythingTextRMSNorm(Qwen2RMSNorm):
    pass


class LocateAnythingTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.rope_type = config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class LocateAnythingTextMLP(Qwen2MLP):
    pass


class LocateAnythingTextAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
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
            is_causal=False,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LocateAnythingTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = LocateAnythingTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = LocateAnythingTextMLP(config)
        self.input_layernorm = LocateAnythingTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LocateAnythingTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LocateAnythingTextModel(nn.Module):
    """Qwen2-based decoder with block-diffusion attention for Parallel Box Decoding."""

    def __init__(self, config: LocateAnythingConfig):
        super().__init__()
        self.config = config
        text_config = config.text_config
        self.padding_idx = text_config.pad_token_id
        self.vocab_size = text_config.vocab_size
        self.block_size = config.block_size
        self.causal_attn = config.causal_attn
        self.text_mask_token_id = config.text_mask_token_id

        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LocateAnythingTextDecoderLayer(text_config, layer_idx) for layer_idx in range(text_config.num_hidden_layers)]
        )
        self.norm = LocateAnythingTextRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.rotary_emb = LocateAnythingTextRotaryEmbedding(config=text_config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def inject_visual_features(self, input_ids, visual_features, image_token_index):
        input_embeds = self.embed_tokens(input_ids)
        if visual_features is None:
            return input_embeds
        batch_size, num_tokens, channels = input_embeds.shape
        flat_embeds = input_embeds.reshape(batch_size * num_tokens, channels)
        flat_ids = input_ids.reshape(batch_size * num_tokens)
        selected = flat_ids == image_token_index
        if selected.sum() > 0:
            flat_embeds[selected] = visual_features.reshape(-1, channels).to(flat_embeds.device, flat_embeds.dtype)
        return flat_embeds.reshape(batch_size, num_tokens, channels)

    def _build_attention_mask(self, attention_mask, input_ids, position_ids, inputs_embeds, past_length, use_cache):
        batch_size, q_len = position_ids.shape
        kv_len = past_length + q_len
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        min_value = torch.finfo(dtype).min

        if not self.training:
            q_pos = torch.arange(q_len, device=device).view(q_len, 1) + past_length
            k_pos = torch.arange(kv_len, device=device).view(1, kv_len)
            allowed = k_pos <= q_pos
            mask = torch.full((q_len, kv_len), min_value, dtype=dtype, device=device)
            mask = mask.masked_fill(allowed, 0.0)
            mask = mask[None, None].expand(batch_size, 1, q_len, kv_len).clone()
            if attention_mask is not None and attention_mask.dim() == 2:
                padding = (attention_mask[:, None, None, :kv_len] == 0)
                mask = mask.masked_fill(padding, min_value)

            is_ar = q_len == 1 or (input_ids is not None and input_ids[0, -1].item() != self.text_mask_token_id)
            if is_ar:
                return mask

            for b in range(batch_size):
                mask[b, 0] = update_causal_mask_for_one_gen_window_2d(
                    mask[b, 0], block_size=self.block_size, use_cache=use_cache, causal_attn=self.causal_attn
                )
            return mask

        x0_len = find_prefix_seq_length_by_pe(position_ids)
        return create_block_diff_mask_by_pe_4d(
            block_size=self.block_size,
            x0_len_list=x0_len,
            position_ids=position_ids,
            causal_attn=self.causal_attn,
            dtype=dtype,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        image_token_index: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        # `input_ids` may be passed alongside `inputs_embeds`: in that case it is only used to build the
        # block-diffusion attention mask (visual features are already injected into `inputs_embeds`).
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config.text_config)

        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.inject_visual_features(input_ids, visual_features, image_token_index)

        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)

        causal_mask = self._build_attention_mask(
            attention_mask, input_ids, position_ids, inputs_embeds, past_length, bool(use_cache)
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ====================================================================================================
# Composite model
# ====================================================================================================


@auto_docstring
class LocateAnythingPreTrainedModel(PreTrainedModel):
    config: LocateAnythingConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _no_split_modules = ["LocateAnythingVisionLayer", "LocateAnythingTextDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True

    def _init_weights(self, module):
        # Delegate standard modules (Linear/Conv/Embedding/LayerNorm/RMSNorm/RotaryEmbedding) to the base
        # implementation, which uses the flag-aware `transformers.initialization` functions. Those respect the
        # `_is_hf_initialized` marker so that weights already loaded from a checkpoint are never re-initialized.
        super()._init_weights(module)
        if isinstance(module, LocateAnythingLearnable2DInterpPosEmb):
            std = getattr(self.config, "initializer_range", None) or self.config.text_config.initializer_range
            init.normal_(module.weight, mean=0.0, std=std)


@auto_docstring(
    custom_intro="""
    Base class for LocateAnything outputs, with hidden states and attentions.
    """
)
@dataclass
class LocateAnythingModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        The image features produced by the vision encoder after projection.
    """

    image_hidden_states: Optional[torch.FloatTensor] = None


@auto_docstring(
    custom_intro="""
    The LocateAnything model which consists of a MoonViT vision backbone and a Qwen2 language model, without a
    language modeling head.
    """
)
class LocateAnythingModel(LocateAnythingPreTrainedModel):
    def __init__(self, config: LocateAnythingConfig):
        super().__init__(config)
        # The text decoder is a plain `nn.Module` (not an `AutoModel`), so it does not self-configure its attention
        # implementation the way the vision tower does. Mirror the resolved top-level value onto `text_config`.
        config.text_config._attn_implementation = config._attn_implementation
        self.vision_tower = LocateAnythingVisionModel(config.vision_config)
        self.multi_modal_projector = LocateAnythingMultiModalProjector(config)
        self.language_model = LocateAnythingTextModel(config)
        self.image_token_index = config.image_token_id
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_hws: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(dtype=self.dtype)
        vision_features = self.vision_tower(pixel_values=pixel_values, image_grid_hws=image_grid_hws).last_hidden_state
        image_features = self.multi_modal_projector(vision_features)
        return image_features

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_hws: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LocateAnythingModelOutputWithPast]:
        r"""
        image_grid_hws (`torch.Tensor` of shape `(num_images, 2)`, *optional*):
            The height and width (in vision patches) of each image, used to unpack and project the variable-length
            `pixel_values` produced by the MoonViT vision encoder.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        image_features = None
        if inputs_embeds is None:
            if pixel_values is not None:
                image_features = self.get_image_features(pixel_values=pixel_values, image_grid_hws=image_grid_hws)
            inputs_embeds = self.language_model.inject_visual_features(
                input_ids, image_features, self.image_token_index
            )
            input_ids_for_mask = input_ids
        else:
            input_ids_for_mask = None

        outputs = self.language_model(
            input_ids=input_ids_for_mask,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        return LocateAnythingModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            image_hidden_states=image_features,
        )


@auto_docstring(
    custom_intro="""
    Base class for LocateAnything causal language model (or autoregressive) outputs.
    """
)
@dataclass
class LocateAnythingCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed):
        Pre-computed hidden-states that can be used to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        The image features produced by the vision encoder after projection.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


@auto_docstring(
    custom_intro="""
    The LocateAnything model for visual grounding, with a language modeling head and Parallel Box Decoding generation.
    """
)
class LocateAnythingForConditionalGeneration(LocateAnythingPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: LocateAnythingConfig):
        super().__init__(config)
        self.model = LocateAnythingModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.token_ids = get_token_ids_from_config(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_image_features(self, pixel_values, image_grid_hws):
        return self.model.get_image_features(pixel_values=pixel_values, image_grid_hws=image_grid_hws)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_hws: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LocateAnythingCausalLMOutputWithPast]:
        r"""
        image_grid_hws (`torch.Tensor` of shape `(num_images, 2)`, *optional*):
            The grid height and width (in patches) of each image.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return LocateAnythingCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            image_hidden_states=outputs.image_hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        image_grid_hws: Optional[torch.Tensor] = None,
        tokenizer=None,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        **kwargs,
    ) -> Union[str, torch.LongTensor]:
        r"""
        Generate structured grounding output using Parallel Box Decoding (PBD).

        Args:
            generation_mode (`str`, *optional*, defaults to `"hybrid"`):
                One of `"fast"` (Multi-Token Prediction only), `"slow"` (auto-regressive only) or `"hybrid"` (MTP
                with auto-regressive fallback on uncertain boxes).
            tokenizer ([`PreTrainedTokenizer`], *optional*):
                Tokenizer used to decode the generated ids into text. If provided, the decoded string is returned;
                otherwise the generated token ids are returned.
        """
        if generation_mode not in ("fast", "slow", "hybrid"):
            raise ValueError(f"Unsupported generation_mode='{generation_mode}'. Use 'fast', 'slow', or 'hybrid'.")

        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("LocateAnything Parallel Box Decoding only supports batch_size == 1.")

        block_size = self.config.block_size
        token_ids = self.token_ids
        sampling_kwargs = {
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "generation_mode": generation_mode,
        }

        visual_features = None
        if pixel_values is not None:
            if isinstance(image_grid_hws, torch.Tensor):
                image_grid_hws = image_grid_hws.to(device=device, dtype=torch.int32)
            else:
                image_grid_hws = torch.as_tensor(image_grid_hws, device=device, dtype=torch.int32)
            visual_features = self.get_image_features(
                pixel_values.to(self.dtype), image_grid_hws
            )

        language_model = self.model.language_model
        generated = input_ids.clone()
        max_len = seq_len + max_new_tokens
        if tokenizer is not None:
            max_len = min(getattr(tokenizer, "model_max_length", max_len), max_len)

        past_key_values = DynamicCache(config=self.config.text_config)
        full_position_ids = torch.arange(0, max_len + block_size, device=device).unsqueeze(0)
        default_mask_token_id = token_ids["default_mask_token_id"]
        pre_mask_tokens = torch.full(
            (batch_size, block_size - 1), default_mask_token_id, dtype=generated.dtype, device=device
        )

        use_mtp = generation_mode in ("fast", "hybrid")
        iter_round = 0
        while generated.size(1) < max_len:
            iter_round += 1
            start_idx = past_key_values.get_seq_length()

            if use_mtp:
                sequence = torch.cat((generated, generated[:, -1:], pre_mask_tokens), dim=1)
                position_ids = full_position_ids[:, start_idx : sequence.size(1)].clone()
                position_ids[0, -block_size:] -= 1
            else:
                sequence = generated
                position_ids = full_position_ids[:, start_idx : sequence.size(1)].clone()

            step_input_ids = sequence[:, start_idx:]
            step_visual_features = visual_features if iter_round == 1 else None

            outputs = language_model(
                input_ids=step_input_ids,
                visual_features=step_visual_features,
                image_token_index=self.config.image_token_id,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = self.lm_head(outputs.last_hidden_state)
            past_key_values.crop(generated.shape[1])

            if use_mtp:
                out_type, out_token = self._sample_block_mtp(logits, generated, token_ids, block_size, sampling_kwargs)
            else:
                out_type, out_token = self._sample_token_ar(logits, generated, token_ids, generation_mode, sampling_kwargs)

            generated = torch.cat([generated, out_token.unsqueeze(0)], dim=1)

            if out_type == "im_end":
                break
            if generation_mode == "hybrid":
                if out_type == "error_box":
                    use_mtp = False
                elif out_type == "box_end_ar":
                    use_mtp = True

        generated_ids = generated[:, seq_len:]
        if tokenizer is not None:
            return tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return generated_ids

    def _sample_block_mtp(self, logits, generated, token_ids, block_size, sampling_kwargs):
        next_token_logits = logits[:, -block_size:, :]
        _, _, x0, box_avg = sample_tokens(next_token_logits, generated, token_ids, keep_k=5, **sampling_kwargs)
        is_box_empty = (box_avg[0] == 0).all()
        new_tokens = x0[0] if is_box_empty else box_avg[0]
        out_pattern = handle_pattern(new_tokens, token_ids, sampling_kwargs["generation_mode"])
        out_token = torch.tensor(out_pattern["tokens"], dtype=x0.dtype, device=x0.device)
        return out_pattern["type"], out_token

    def _sample_token_ar(self, logits, generated, token_ids, generation_mode, sampling_kwargs):
        next_token_logits = logits[:, -1:, :]
        _, _, x0, _ = sample_tokens(next_token_logits, generated, token_ids, **sampling_kwargs)
        out_token = x0[0]
        token_val = out_token[0].item()
        out_type = "continue_ar"
        if generation_mode == "hybrid":
            if token_val == token_ids["box_end_token_id"]:
                out_type = "box_end_ar"
            elif (
                token_ids["coord_start_token_id"] <= token_val <= token_ids["coord_end_token_id"]
                or token_val == token_ids["none_token_id"]
            ):
                out_type = "coord_ar"
            else:
                out_type = "im_end"
        else:
            if token_val == token_ids["im_end_token_id"]:
                out_type = "im_end"
        return out_type, out_token


# ====================================================================================================
# Sampling / decoding utilities for Parallel Box Decoding
# ====================================================================================================


def get_token_ids_from_config(config) -> dict[str, int]:
    text_config = getattr(config, "text_config", None)
    return {
        "box_start_token_id": config.box_start_token_id,
        "box_end_token_id": config.box_end_token_id,
        "coord_start_token_id": config.coord_start_token_id,
        "coord_end_token_id": config.coord_end_token_id,
        "ref_start_token_id": config.ref_start_token_id,
        "ref_end_token_id": config.ref_end_token_id,
        "none_token_id": config.none_token_id,
        "null_token_id": config.null_token_id,
        "switch_token_id": config.switch_token_id,
        "default_mask_token_id": config.text_mask_token_id,
        "im_end_token_id": getattr(text_config, "eos_token_id", 151645),
    }


def top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, repetition_penalty: float) -> torch.Tensor:
    if repetition_penalty == 1.0:
        return logits
    squeeze_back = False
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)
        squeeze_back = True
    batch_size, seq_len, vocab_size = logits.shape
    token_mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=logits.device)
    for b in range(batch_size):
        unique_tokens = input_ids[b].unique()
        valid_tokens = unique_tokens[(unique_tokens >= 0) & (unique_tokens < vocab_size)]
        if valid_tokens.numel() > 0:
            token_mask[b, valid_tokens] = True
    token_mask = token_mask.unsqueeze(1).expand(-1, seq_len, -1)
    positive = logits > 0
    logits = torch.where(token_mask & positive, logits / repetition_penalty, logits)
    logits = torch.where(token_mask & ~positive, logits * repetition_penalty, logits)
    if squeeze_back:
        logits = logits.squeeze(1)
    return logits


def sample_tokens(logits: torch.Tensor, generated: torch.Tensor, token_ids: dict[str, int], keep_k: int = 5, **kwargs):
    batch_size, seq_len, vocab_size = logits.shape
    repetition_penalty = kwargs.get("repetition_penalty", 1.0)
    temperature = kwargs.get("temperature", 0.0)
    top_p = kwargs.get("top_p", None)
    top_k = kwargs.get("top_k", None)

    if repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated, repetition_penalty)
    if temperature and temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = torch.softmax(logits, dim=-1)
    if temperature and temperature > 0:
        try:
            x0 = torch.distributions.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if seq_len == 1:
        return probs, confidence, x0, None

    box_avg = []
    fallback_box = torch.zeros(1, dtype=x0.dtype, device=x0.device)
    for b in range(batch_size):
        decoded_box = decode_bbox_avg(
            logits[b], probs[b], token_ids, keep_k=kwargs.get("keep_k_avg", 4),
            generation_mode=kwargs.get("generation_mode", "hybrid"),
        )
        if decoded_box is not None:
            box_avg.append(decoded_box)
        else:
            out_ref = decode_ref(logits[b], probs[b], token_ids)
            box_avg.append(out_ref if out_ref is not None else fallback_box)
    box_avg = torch.stack(box_avg)
    return probs, confidence, x0, box_avg


def is_valid_box_frame(probs, token_ids: dict[str, int], start_thresh=0.6, end_thresh=0.2) -> str:
    box_start_token_id = token_ids["box_start_token_id"]
    box_end_token_id = token_ids["box_end_token_id"]
    null_token_id = token_ids["null_token_id"]
    im_end_token_id = token_ids["im_end_token_id"]
    none_token_id = token_ids["none_token_id"]

    if probs[0, box_start_token_id] >= start_thresh:
        if (
            probs[1, none_token_id] > 0.2
            and probs[2, box_end_token_id] > 0.2
            and probs[3, null_token_id] > 0.1
            and probs[4, null_token_id] > 0.1
        ):
            return "empty_box"

    end_target_ids = torch.tensor([box_end_token_id, null_token_id, im_end_token_id], device=probs.device)
    if probs[5, end_target_ids].sum() >= end_thresh:
        return "legal_box"
    return "illegal_box"


def decode_bbox_avg(logits, probs, token_ids, keep_k=5, start_thresh=0.7, end_thresh=0.2, generation_mode="hybrid"):
    coord_start_token_id = token_ids["coord_start_token_id"]
    coord_end_token_id = token_ids["coord_end_token_id"]
    box_start_token_id = token_ids["box_start_token_id"]
    box_end_token_id = token_ids["box_end_token_id"]
    none_token_id = token_ids["none_token_id"]
    device = logits.device

    box_type = is_valid_box_frame(probs, token_ids, start_thresh=start_thresh, end_thresh=end_thresh)
    if box_type == "empty_box":
        return torch.tensor(
            [box_start_token_id, none_token_id, box_end_token_id, token_ids["null_token_id"],
             token_ids["null_token_id"], token_ids["null_token_id"]],
            dtype=torch.long, device=device,
        )
    elif box_type == "illegal_box":
        return None

    pos_probs, pos_ids = torch.topk(probs[1:5], k=keep_k, dim=-1)
    mask = (pos_ids >= coord_start_token_id) & (pos_ids <= coord_end_token_id)
    if not mask.any(dim=-1).all():
        return None

    first_valid_idx = mask.long().argmax(dim=-1, keepdim=True)
    first_valid_probs = pos_probs.gather(-1, first_valid_idx).squeeze(-1)
    first_valid_ids = pos_ids.gather(-1, first_valid_idx).squeeze(-1)

    if generation_mode == "hybrid":
        valid_counts = mask.sum(dim=-1)
        large_num, small_num = 999999, -999999
        valid_ids_for_max = torch.where(mask, pos_ids, torch.tensor(small_num, device=device))
        valid_ids_for_min = torch.where(mask, pos_ids, torch.tensor(large_num, device=device))
        valid_max = valid_ids_for_max.max(dim=-1)[0]
        valid_min = valid_ids_for_min.min(dim=-1)[0]
        is_abnormal = (first_valid_probs < 0.9) & (valid_counts > 1) & ((valid_max - valid_min) > 60)
        final_coords = torch.where(is_abnormal, torch.tensor(0, device=device), first_valid_ids)
    else:
        final_coords = first_valid_ids

    start_t = torch.tensor([box_start_token_id], dtype=final_coords.dtype, device=device)
    end_t = torch.tensor([box_end_token_id], dtype=final_coords.dtype, device=device)
    return torch.cat([start_t, final_coords, end_t])


def decode_ref(logits, probs, token_ids, keep_k=5, start_thresh=0.6):
    ref_start_token_id = token_ids.get("ref_start_token_id")
    coord_start_token_id = token_ids["coord_start_token_id"]
    coord_end_token_id = token_ids["coord_end_token_id"]
    device = probs.device

    if probs[0, ref_start_token_id] < start_thresh:
        return None

    pos_probs, pos_ids = torch.topk(probs[1:], k=keep_k, dim=-1)
    is_coord = (pos_ids >= coord_start_token_id) & (pos_ids <= coord_end_token_id)
    is_valid = ~is_coord
    if not is_valid.any(dim=-1).all():
        return None

    first_valid_idx = is_valid.long().argmax(dim=-1, keepdim=True)
    final_text_ids = pos_ids.gather(-1, first_valid_idx).squeeze(-1)
    start_t = torch.tensor([ref_start_token_id], dtype=final_text_ids.dtype, device=device)
    return torch.cat([start_t, final_text_ids])


def handle_pattern(x0, token_ids: dict[str, int], generation_mode: str = "hybrid") -> dict:
    null_token_id = token_ids["null_token_id"]
    im_end_token_id = token_ids["im_end_token_id"]
    box_start_token_id = token_ids["box_start_token_id"]
    box_end_token_id = token_ids["box_end_token_id"]
    none_token_id = token_ids["none_token_id"]
    coord_start_token_id = token_ids["coord_start_token_id"]
    coord_end_token_id = token_ids["coord_end_token_id"]
    ref_end_token_id = token_ids["ref_end_token_id"]

    x0 = x0.tolist()

    if x0[0] == null_token_id or x0[0] == im_end_token_id:
        return {"type": "im_end", "tokens": [im_end_token_id]}
    elif x0[:2] == [box_start_token_id, none_token_id]:
        return {"type": "empty_box", "tokens": [box_start_token_id, none_token_id, box_end_token_id]}
    elif x0[0] == box_start_token_id:
        coord_ix = 1
        for coord in x0[1:5]:
            if coord_start_token_id <= coord <= coord_end_token_id:
                coord_ix += 1
            else:
                break
        if coord_ix == 5 and x0[5] == box_end_token_id:
            return {"type": "coord_box", "tokens": x0}
        elif coord_ix == 3 and x0[3] == box_end_token_id:
            return {"type": "point_box", "tokens": x0[:4]}
        else:
            if generation_mode == "fast":
                return {"type": "coord_box", "tokens": x0}
            return {"type": "error_box", "tokens": x0[:coord_ix]}
    else:
        for i, token in enumerate(x0):
            if token == null_token_id:
                x0 = x0[:i]
                break
        if len(x0) >= 2 and x0[-1] == x0[-2] == ref_end_token_id:
            x0 = x0[:-1]
        return {"type": "ref_object", "tokens": x0}


__all__ = [
    "LocateAnythingVisionPreTrainedModel",
    "LocateAnythingVisionModel",
    "LocateAnythingPreTrainedModel",
    "LocateAnythingModel",
    "LocateAnythingForConditionalGeneration",
]
