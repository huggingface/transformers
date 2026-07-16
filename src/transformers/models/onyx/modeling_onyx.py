# coding=utf-8
# Copyright 2026 the HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from .configuration_onyx import OnyxConfig


logger = logging.get_logger(__name__)


class OnyxRMSNorm(nn.Module):
    """RMSNorm where weight stores offset from 1.0: forward does rms_norm(x) * (1 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), w.float(), self.eps).to(x.dtype)


class OnyxFinalRMSNorm(nn.Module):
    """RMSNorm where weight IS the scale (gain_center=0). Used for output norm only."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


class OnyxScalelessRMSNorm(nn.Module):
    """RMSNorm without learnable parameters. Used for QK norm and embedding norm."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (self.dim,), None, self.eps).to(x.dtype)


class OnyxRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, theta: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("freqs", freqs, persistent=True)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        freqs = torch.outer(position_ids.float().squeeze(0), self.freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_out = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_out.to(x.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for grouped-query attention."""
    bs, num_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(bs, num_kv_heads, n_rep, slen, head_dim)
    return x.reshape(bs, num_kv_heads * n_rep, slen, head_dim)


class OnyxMLP(nn.Module):
    def __init__(self, config: OnyxConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class OnyxVisionAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        slen: int,
        freqs_cis: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs_slen, d = x.shape
        bs = bs_slen // slen

        xq = self.q_proj(x).view(bs, slen, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(bs, slen, self.n_heads, self.head_dim)
        xv = self.v_proj(x).view(bs, slen, self.n_heads, self.head_dim)

        if freqs_cis is not None:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_bias)
        out = out.transpose(1, 2).contiguous().view(bs_slen, -1)
        return self.o_proj(out)


class OnyxVisionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.c_fc = nn.Linear(dim, hidden_dim, bias=True)
        self.c_proj = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class OnyxVisionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, mlp_hidden: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = OnyxVisionAttention(dim, n_heads, head_dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = OnyxVisionMLP(dim, mlp_hidden)

    def forward(
        self,
        x: torch.Tensor,
        slen: int,
        freqs_cis: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, s, d = x.shape
        flat = x.view(bs * s, d)
        h = self.ln_1(flat)
        h = self.attn(h, slen=s, freqs_cis=freqs_cis, attn_bias=attn_bias)
        flat = flat + h
        h = self.ln_2(flat)
        h = self.mlp(h)
        flat = flat + h
        return flat.view(bs, s, d)


class OnyxVisionEncoder(nn.Module):
    def __init__(self, config: OnyxConfig):
        super().__init__()
        latent_dim = config.vision_latent_dim
        n_heads = config.vision_heads
        head_dim = latent_dim // n_heads
        mlp_hidden = int(config.vision_mlp_ratio * latent_dim)
        patch_dim = config.vision_patch_temporal * 3 * config.vision_patch_size**2

        self.latent_dim = latent_dim
        self.patch_size = config.vision_patch_size
        self.patch_temporal = config.vision_patch_temporal
        self.downsample_factor = config.vision_downsample_factor
        self.sparse_attention_factor = config.vision_sparse_attention_factor
        self.pos_emb_grid_h = config.vision_pos_emb_grid_h
        self.pos_emb_grid_w = config.vision_pos_emb_grid_w
        self.head_dim = head_dim

        self.conv1_linear = nn.Linear(patch_dim, latent_dim, bias=False)
        self.positional_embedding_vlm = nn.Parameter(
            torch.zeros(config.vision_pos_emb_grid_h * config.vision_pos_emb_grid_w, latent_dim)
        )
        self.ln_pre = nn.LayerNorm(latent_dim)
        self.transformer = nn.ModuleList(
            [OnyxVisionBlock(latent_dim, n_heads, head_dim, mlp_hidden) for _ in range(config.vision_layers)]
        )
        self.ln_post = nn.LayerNorm(latent_dim)

    def _make_2d_rope(self, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
        half_dim = self.head_dim // 2
        quarter = half_dim // 2
        theta = 10000.0
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, half_dim, 2, dtype=torch.float32, device=device)[:quarter] / half_dim)
        )
        idx_h = torch.arange(1, grid_h + 1, dtype=torch.float32, device=device)
        idx_w = torch.arange(1, grid_w + 1, dtype=torch.float32, device=device)
        idx_ij_h = idx_h.unsqueeze(1).expand(-1, grid_w).reshape(-1)
        idx_ij_w = idx_w.unsqueeze(0).expand(grid_h, -1).reshape(-1)
        freq_h = torch.outer(idx_ij_h, inv_freq)
        freq_w = torch.outer(idx_ij_w, inv_freq)
        freq = torch.cat([freq_w, freq_h], dim=-1)
        return torch.view_as_complex(torch.stack([torch.cos(freq), torch.sin(freq)], dim=-1))

    def _get_pos_emb(self, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
        # Interpolate in the encoder's native dtype (bf16); float32 causes small drift.
        dtype = self.positional_embedding_vlm.dtype
        pos_emb = (
            self.positional_embedding_vlm.view(self.pos_emb_grid_h, self.pos_emb_grid_w, self.latent_dim)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        inv_h = 1.0 / grid_h
        inv_w = 1.0 / grid_w
        ys = torch.linspace(-1 + inv_h, 1 - inv_h, grid_h, device=device, dtype=dtype)
        xs = torch.linspace(-1 + inv_w, 1 - inv_w, grid_w, device=device, dtype=dtype)
        pos_xy = torch.stack(torch.meshgrid(ys, xs, indexing="xy"), dim=-1).reshape(-1, 2)[None, None]
        sampled = F.grid_sample(pos_emb, pos_xy, mode="bilinear", align_corners=False)
        return sampled[0, :, 0, :].T

    def _pixel_shuffle_downsample(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        f = self.downsample_factor
        n_out = (grid_h // f) * (grid_w // f)
        device = x.device
        ds_perm = torch.arange(grid_h * grid_w, device=device)
        ds_perm = ds_perm.view(grid_h // f, f, grid_w // f, f)
        ds_perm = ds_perm.permute(0, 2, 1, 3).reshape(-1)
        x_flat = x.squeeze(0)
        ds_x = x_flat[ds_perm]
        d = x_flat.shape[-1]
        ds_x = ds_x.view(n_out, f * f, d).permute(0, 2, 1).contiguous().view(n_out, d * f * f)
        return ds_x.unsqueeze(0)

    def _get_sparse_perm_and_mask(self, grid_h: int, grid_w: int, device: torch.device):
        gh, gw = self.pos_emb_grid_h, self.pos_emb_grid_w
        pad_h = math.ceil(grid_h / gh) * gh
        pad_w = math.ceil(grid_w / gw) * gw
        idx = torch.arange(grid_h * grid_w, device=device).view(grid_h, grid_w)
        idx = F.pad(idx, (0, pad_w - grid_w, 0, pad_h - grid_h), value=-1).flatten()
        idx = idx.view(pad_h // gh, gh, pad_w // gw, gw)
        idx = idx.permute(0, 2, 1, 3).reshape(-1)
        sp_perm = idx[idx != -1]
        valid = (idx != -1).view(-1, gh * gw)
        sp_slens = valid.sum(dim=1).tolist()
        return sp_perm, sp_slens

    def _make_block_diag_mask(self, slens: list[int], device: torch.device) -> torch.Tensor:
        total = sum(slens)
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        offset = 0
        for s in slens:
            mask[offset : offset + s, offset : offset + s] = True
            offset += s
        return mask

    def _compute_grid_size(self, img_w: int, img_h: int, patch_hw: int, max_tokens: int) -> tuple[int, int, int]:
        i_nph = img_h / patch_hw
        i_npw = img_w / patch_hw
        ratio = i_npw / i_nph if i_nph > 0 else 1.0
        if i_nph * i_npw > max_tokens:
            i_nph = (max_tokens / ratio) ** 0.5
            i_npw = i_nph * ratio
        candidates = list(
            set(
                itertools.product(
                    [math.floor(i_nph), math.ceil(i_nph)],
                    [math.floor(i_npw), math.ceil(i_npw)],
                )
            )
        )
        candidates = [(nph, npw) for nph, npw in candidates if nph >= 1 and npw >= 1 and nph * npw <= max_tokens]
        if not candidates:
            candidates = [(max(1, round(i_nph)), max(1, round(i_npw)))]
        nph, npw = min(candidates, key=lambda c: abs(c[0] / c[1] - img_h / img_w))
        return nph * patch_hw, npw * patch_hw, nph * npw

    def compute_image_size(self, img_w: int, img_h: int) -> tuple[int, int, int]:
        ph = self.patch_size * self.downsample_factor
        return self._compute_grid_size(img_w, img_h, ph, 4096)

    def compute_video_frame_size(
        self,
        img_w: int,
        img_h: int,
        compression_ratio: int = 1,
        max_num_tokens: int = 144,
    ) -> tuple[int, int, int]:
        pool_factor = int(math.sqrt(compression_ratio))
        ph = self.patch_size * self.downsample_factor * pool_factor
        i_nph = img_h / ph
        i_npw = img_w / ph
        ratio = i_npw / i_nph if i_nph > 0 else 1.0
        if i_nph * i_npw > max_num_tokens:
            i_nph = (max_num_tokens / ratio) ** 0.5
            i_npw = i_nph * ratio
        candidates = list(
            set(
                itertools.product(
                    [math.floor(i_nph), math.ceil(i_nph)],
                    [math.floor(i_npw), math.ceil(i_npw)],
                )
            )
        )
        candidates = [(nph, npw) for nph, npw in candidates if nph >= 1 and npw >= 1 and nph * npw <= max_num_tokens]
        if not candidates:
            candidates = [(max(1, round(i_nph)), max(1, round(i_npw)))]
        nph, npw = min(candidates, key=lambda c: abs(c[0] / c[1] - img_h / img_w))
        return nph * ph, npw * ph, nph * npw

    def forward(self, images: list[torch.Tensor]) -> torch.Tensor:
        device = self.conv1_linear.weight.device
        dtype = self.conv1_linear.weight.dtype
        ps = self.patch_size
        sf = self.sparse_attention_factor

        all_x: list[torch.Tensor] = []
        all_freqs: list[torch.Tensor] = []
        all_sp_slens: list[int] = []
        all_global_slens: list[int] = []
        per_image_meta: list[tuple[int, int, int, torch.Tensor | None]] = []

        for img in images:
            img = img.to(device=device, dtype=dtype)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            _, c, h, w = img.shape
            grid_h, grid_w = h // ps, w // ps
            n_tokens = grid_h * grid_w

            if c == 3:
                patches = img.unfold(2, ps, ps).unfold(3, ps, ps)
                patches = patches.contiguous().view(1, c, grid_h, grid_w, ps, ps)
                patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
                patches = patches.unsqueeze(3).expand(-1, -1, -1, self.patch_temporal, -1, -1, -1)
            else:
                frame_patches = []
                for t in range(self.patch_temporal):
                    frame = img[:, t * 3 : (t + 1) * 3]
                    fp = frame.unfold(2, ps, ps).unfold(3, ps, ps)
                    fp = fp.contiguous().view(1, 3, grid_h, grid_w, ps, ps)
                    fp = fp.permute(0, 2, 3, 1, 4, 5).contiguous()
                    frame_patches.append(fp)
                patches = torch.stack(frame_patches, dim=3)
            patches = patches.reshape(1, n_tokens, -1)

            x = self.conv1_linear(patches)
            pos_emb = self._get_pos_emb(grid_h, grid_w, device)
            x = x + pos_emb.unsqueeze(0).to(dtype)
            x = self.ln_pre(x.view(-1, self.latent_dim)).view(1, -1, self.latent_dim)

            freqs_cis = self._make_2d_rope(grid_h, grid_w, device)

            sp_perm = None
            if sf > 1:
                sp_perm, sp_slens = self._get_sparse_perm_and_mask(grid_h, grid_w, device)
                x = x[:, sp_perm]
                freqs_cis = freqs_cis[sp_perm]
                all_sp_slens.extend(sp_slens)

            all_x.append(x.squeeze(0))
            all_freqs.append(freqs_cis)
            all_global_slens.append(n_tokens)
            per_image_meta.append((grid_h, grid_w, n_tokens, sp_perm))

        x = torch.cat(all_x, dim=0).unsqueeze(0)
        freqs_cis = torch.cat(all_freqs, dim=0)
        total_tokens = x.shape[1]

        global_mask = self._make_block_diag_mask(all_global_slens, device) if len(images) > 1 else None
        sp_mask = self._make_block_diag_mask(all_sp_slens, device) if all_sp_slens else None

        for i, block in enumerate(self.transformer):
            is_global = (i == len(self.transformer) - 1) or ((i + 1) % sf == 0)
            if is_global or not all_sp_slens:
                x = block(x, slen=total_tokens, freqs_cis=freqs_cis, attn_bias=global_mask)
            else:
                x = block(x, slen=total_tokens, freqs_cis=freqs_cis, attn_bias=sp_mask)

        all_features = []
        offset = 0
        for grid_h, grid_w, n_tokens, sp_perm in per_image_meta:
            img_x = x[:, offset : offset + n_tokens]
            offset += n_tokens

            if sp_perm is not None:
                inv_perm = torch.empty_like(sp_perm)
                inv_perm[sp_perm] = torch.arange(len(sp_perm), device=device)
                img_x = img_x[:, inv_perm]

            img_x = self.ln_post(img_x.view(-1, self.latent_dim)).view(1, -1, self.latent_dim)
            img_x = self._pixel_shuffle_downsample(img_x, grid_h, grid_w)
            all_features.append(img_x.squeeze(0))

        return torch.cat(all_features, dim=0)


class OnyxVisionAdapter(nn.Module):
    def __init__(self, config: OnyxConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.vision_output_dim, config.vision_adapter_dim, bias=False)
        self.c_proj = nn.Linear(config.vision_adapter_dim, config.vision_adapter_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.c_proj(F.gelu(self.c_fc(x))))


class OnyxAttention(nn.Module):
    def __init__(self, config: OnyxConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)

        self.use_output_gate = config.use_attn_output_gate
        if self.use_output_gate:
            self.output_gate_proj = nn.Linear(config.hidden_size, q_dim, bias=False)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.qk_norm = OnyxScalelessRMSNorm(self.head_dim, config.rms_norm_eps)
            self.scale_query_by = config.qk_scale_factor / math.sqrt(self.head_dim)

        self.use_rope = config.no_rope_layers[layer_idx] == 1
        self.layer_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, slen, _ = hidden_states.shape

        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(bs, slen, self.num_heads, self.head_dim)
        xk = xk.view(bs, slen, self.num_kv_heads, self.head_dim)
        xv = xv.view(bs, slen, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            xq = self.qk_norm(xq) * self.scale_query_by
            xk = self.qk_norm(xk)

        if self.use_rope:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if past_key_values is not None:
            xk, xv = past_key_values.update(xk, xv, self.layer_idx, {"cache_position": cache_position})

        if self.num_kv_groups > 1:
            xk = repeat_kv(xk, self.num_kv_groups)
            xv = repeat_kv(xv, self.num_kv_groups)

        if attention_mask is None:
            attn_out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=slen > 1)
        else:
            attn_out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attention_mask)

        attn_out = attn_out.transpose(1, 2).contiguous()

        if self.use_output_gate:
            og = self.output_gate_proj(hidden_states).view(bs, slen, self.num_heads, self.head_dim)
            attn_out = torch.sigmoid(og) * attn_out

        attn_out = attn_out.reshape(bs, slen, -1)
        return self.o_proj(attn_out)


class OnyxDecoderLayer(nn.Module):
    def __init__(self, config: OnyxConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = OnyxRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = OnyxAttention(config, layer_idx)
        self.post_attn_norm = OnyxRMSNorm(config.hidden_size, config.post_norm_eps)
        self.post_attention_layernorm = OnyxRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = OnyxMLP(config)
        self.post_ffn_norm = OnyxRMSNorm(config.hidden_size, config.post_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states, freqs_cis, attention_mask, past_key_values, cache_position)
        hidden_states = residual + self.post_attn_norm(attn_out)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_out = self.mlp(hidden_states)
        hidden_states = residual + self.post_ffn_norm(ffn_out)

        return hidden_states


@auto_docstring
class OnyxPreTrainedModel(PreTrainedModel):
    config: OnyxConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OnyxDecoderLayer"]
    _supports_sdpa = True
    _supports_cache_class = True


@auto_docstring
class OnyxModel(OnyxPreTrainedModel):
    def __init__(self, config: OnyxConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        if config.normalize_tok_embeddings:
            self.embed_norm = OnyxScalelessRMSNorm(config.hidden_size, config.rms_norm_eps)
        else:
            self.embed_norm = None

        self.has_vision = config.has_vision
        if self.has_vision:
            self.vision_encoder = OnyxVisionEncoder(config)
            self.vision_adapter = OnyxVisionAdapter(config)
            self.vision_projection = nn.Linear(config.vision_adapter_dim, config.hidden_size, bias=False)
            if config.normalize_tok_embeddings:
                self.perception_emb_norm = OnyxScalelessRMSNorm(config.hidden_size, config.rms_norm_eps)
            else:
                self.perception_emb_norm = None

        self.rotary_emb = OnyxRotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

        self.layers = nn.ModuleList([OnyxDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = OnyxFinalRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        pixel_values: list[torch.Tensor] | None = None,
        vision_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Cache | None]:
        bs, slen = input_ids.shape
        use_cache = use_cache if use_cache is not None else False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_len, past_len + slen, device=input_ids.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        h = self.embed_tokens(input_ids)

        if self.embed_norm is not None:
            h = self.embed_norm(h)

        if self.has_vision and pixel_values is not None:
            if vision_mask is None:
                vision_mask = (input_ids == self.config.patch_token_id) | (input_ids == self.config.video_token_id)
            vision_features = self.vision_encoder(pixel_values)
            vision_features = self.vision_adapter(vision_features)
            vision_features = self.vision_projection(vision_features)
            if self.perception_emb_norm is not None:
                vision_features = self.perception_emb_norm(vision_features)
            h[vision_mask] = vision_features.to(h.dtype)

        h = h.to(self.layers[0].input_layernorm.weight.dtype)

        freqs_cis = self.rotary_emb(h, position_ids)

        # position_ids passed as None: we never use packed sequences, and passing it
        # would trigger packed-sequence detection that disables the SDPA is_causal
        # fast path.
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": h,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": None,
        }
        causal_mask = create_causal_mask(**mask_kwargs)
        sliding_mask = create_sliding_window_causal_mask(**mask_kwargs)

        for i, layer in enumerate(self.layers):
            layer_mask = sliding_mask if self.config.layer_types[i] == "sliding_attention" else causal_mask
            h = layer(h, freqs_cis, layer_mask, past_key_values, cache_position)

        h = self.norm(h)
        return h, (past_key_values if use_cache else None)


@auto_docstring
class OnyxForCausalLM(OnyxPreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self):
        return dict.fromkeys(self._tied_weights_keys, 0)

    def __init__(self, config: OnyxConfig):
        super().__init__(config)
        self.model = OnyxModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        vision_mask=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        # Vision features are only consumed on the prefill step: drop pixel_values
        # on decode steps to avoid re-encoding and a shape mismatch.
        if cache_position is not None and cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["vision_mask"] = vision_mask
        return model_inputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        pixel_values: list[torch.Tensor] | None = None,
        vision_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else True

        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            pixel_values=pixel_values,
            vision_mask=vision_mask,
        )

        logits = self.lm_head(hidden_states).float()

        if self.config.output_soft_cap_temp is not None:
            temp = self.config.output_soft_cap_temp
            logits = temp * torch.tanh(logits * self.config.output_multiplier / temp)
        else:
            logits = logits * self.config.output_multiplier

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if return_dict:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=past_key_values,
            )
        return (loss, logits, past_key_values)


__all__ = ["OnyxPreTrainedModel", "OnyxModel", "OnyxForCausalLM"]
