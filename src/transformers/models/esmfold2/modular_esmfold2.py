# Copyright 2026 Biohub. All rights reserved.
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
"""PyTorch ESMFold2 model — the standard released architecture.

Quickstart::

    from transformers import ESMFold2Model

    model = ESMFold2Model.from_pretrained("biohub/ESMFold2", dtype=torch.bfloat16).cuda().eval()
    open("ubq.pdb", "w").write(model.infer_protein_as_pdb("MQIFVKTLTGKT..."))

``infer_protein`` / ``infer_protein_as_pdb`` cover single-chain protein folding from a
sequence; multi-chain, ligand, or MSA inputs are supported by building the structural
feature tensors yourself and passing them directly to ``forward``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ... import initialization as init
from ...integrations import use_kernel_forward_from_hub
from ...modeling_outputs import ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils import is_flash_attn_2_available
from ..auto import AutoModel

# ESMFold2 reuses Llama's plain attention helpers verbatim. `eager_attention_forward`
# upcasts the softmax to fp32 (Llama's variant, NOT esm's no-upcast one) which the
# bit-exact bf16 backbone relies on, and pulls in `repeat_kv` as a dependency. The
# atom-track 3D RoPE (`apply_rotary_emb_3d`) reuses Llama's `rotate_half`.
from ..llama.modeling_llama import eager_attention_forward, rotate_half
from .configuration_esmfold2 import ESMFold2Config


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHAR_VOCAB_SIZE = 64
MAX_CHARS = 4
XYZ_DIMS = 3
MAX_ATOMIC_NUMBER = 128

# Input feature dim = 3 + 1 + 1 + 128 + 64*4 = 389
ATOM_FEATURE_DIM = XYZ_DIMS + 1 + 1 + MAX_ATOMIC_NUMBER + CHAR_VOCAB_SIZE * MAX_CHARS


NUM_RES_TYPES = 33

_EPS = 1e-5

# Default for the triangle / OPM / pair-transition L² ops. Caps peak memory
# so L≈2k folds on an 80 GB GPU (~76 GB peak at chunk=128 for L=1438;
# chunk=64 leaves headroom for the largest foldbench targets). Override via
# ``model.set_chunk_size(...)``; pass None to disable chunking (faster for
# short L but OOM-prone past ~600).
_DEFAULT_CHUNK_SIZE = 64


# ===========================================================================
# Atom-token utilities
# ===========================================================================


def gather_token_to_atom(token_features: Tensor, atom_to_token_idx: Tensor) -> Tensor:
    """Broadcast per-token features to per-atom features using gather.

    Args:
        token_features: [B, L, d]
        atom_to_token_idx: [B, A] int64

    Returns:
        [B, A, d]
    """
    idx = atom_to_token_idx.unsqueeze(-1).expand(-1, -1, token_features.size(-1))
    return torch.gather(token_features, 1, idx)


def scatter_atom_to_token(
    atom_features: Tensor,
    atom_to_token_idx: Tensor,
    n_tokens: int,
    atom_mask: Tensor | None = None,
) -> Tensor:
    """Aggregate per-atom features to per-token features (mean).

    Args:
        atom_features: [B, A, d]
        atom_to_token_idx: [B, A] int64
        n_tokens: L
        atom_mask: [B, A] bool

    Returns:
        [B, L, d]
    """
    B, A, d = atom_features.shape
    n_out = n_tokens
    idx = atom_to_token_idx
    if atom_mask is not None:
        idx = torch.where(atom_mask, atom_to_token_idx, n_tokens)
        n_out = n_tokens + 1
    idx_expanded = idx.unsqueeze(-1).expand(B, A, d)
    out = torch.zeros(B, n_out, d, device=atom_features.device, dtype=atom_features.dtype)
    out.scatter_reduce_(1, idx_expanded, atom_features, reduce="mean", include_self=False)
    return out[:, :n_tokens, :]


def gather_rep_atom_coords(coords: Tensor, rep_atom_idx: Tensor) -> Tensor:
    """Gather representative atom coordinates for each token.

    Args:
        coords: [B, A, 3]
        rep_atom_idx: [B, L] int64

    Returns:
        [B, L, 3]
    """
    idx = rep_atom_idx.unsqueeze(-1).expand(-1, -1, coords.size(-1))
    return torch.gather(coords, 1, idx)


def _compute_intra_token_idx(atom_to_token: Tensor) -> Tensor:
    """Compute local atom index within each token (vectorised).

    Atoms belonging to the same token are contiguous, so this computes a
    running count that resets at each token boundary.

    Args:
        atom_to_token: [B, A] flat index mapping each atom to its token.

    Returns:
        [B, A] tensor with values in [0, max_atoms_per_token - 1].
    """
    same_as_prev = F.pad(atom_to_token[:, 1:] == atom_to_token[:, :-1], (1, 0), value=False)
    ones = torch.ones_like(atom_to_token)
    cumsum = torch.cumsum(ones, dim=-1)
    group_start = cumsum.masked_fill(same_as_prev, 0)
    group_start = torch.cummax(group_start, dim=-1).values
    return cumsum - group_start


def _categorical_mean(logits: Tensor, start: float, end: float) -> Tensor:
    """Expected value of a categorical distribution over evenly-spaced bins.

    Equivalent to ``CategoricalMixture(logits, bins=logits.shape[-1], start, end).mean()``.

    Args:
        logits: [..., n_bins]
        start: left boundary
        end: right boundary

    Returns:
        [...] expected value
    """
    n_bins = logits.shape[-1]
    edges = torch.linspace(start, end, n_bins + 1, device=logits.device, dtype=torch.float32)
    v_bins = (edges[:-1] + edges[1:]) / 2  # [n_bins]
    return (logits.float().softmax(-1) @ v_bins.unsqueeze(1)).squeeze(-1)


# ===========================================================================
# fp32-compute LayerNorm
# ===========================================================================


class ESMFold2LayerNorm(nn.LayerNorm):
    """LayerNorm that always computes in fp32, with its weight stored at the model dtype.

    ESMFold2 runs in the loaded dtype (bf16 for inference) but its LayerNorms must match
    the reference's fp32 norm *compute* (the fork upcasts via autocast). Keeping the weight
    at the model dtype means ``from_pretrained(dtype=bf16)`` rounds it to bf16 exactly like
    the reference's bf16-trained norm weights, while this ``forward`` upcasts to fp32 for
    the computation and casts the result back — so no post-load weight fix-up is needed
    (an fp32 load keeps full-precision weights and an fp32 compute, both no-ops here).
    """

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(x.dtype)


# ===========================================================================
# ESMFold2TransitionLayer (used in ESMFold2DiffusionConditioning)
# ===========================================================================


class ESMFold2TransitionLayer(nn.Module):
    """ESMFold2SwiGLU transition: norm -> a_proj, b_proj -> silu(a)*b -> out_proj."""

    def __init__(self, d_model: int, n: int, eps: float = 1e-5) -> None:
        super().__init__()
        hidden = n * d_model
        self.norm = ESMFold2LayerNorm(d_model, eps=eps)
        self.a_proj = nn.Linear(d_model, hidden, bias=False)
        self.b_proj = nn.Linear(d_model, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        a = self.a_proj(x)
        b = self.b_proj(x)
        return self.out_proj(F.silu(a) * b)


# ===========================================================================
# ESMFold2AdaptiveLayerNorm (used in ESMFold2DiffusionTransformer)
# ===========================================================================


class ESMFold2AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalization (adaLN-Zero)."""

    def __init__(self, d_model: int, d_cond: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_cond = d_cond
        self.eps = eps
        self.s_scale = nn.Parameter(torch.empty(d_cond))  # ones-init in _init_weights
        self.s_gate = nn.Linear(d_cond, d_model, bias=True)
        self.s_shift = nn.Linear(d_cond, d_model, bias=False)

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        a_norm = F.layer_norm(a.float(), (self.d_model,), None, None, self.eps)
        s_norm = F.layer_norm(s.float(), (self.d_cond,), self.s_scale.float(), None, self.eps).to(s.dtype)
        # gate/shift in bf16 (matches the reference's autocast); a_norm is fp32 so the
        # affine promotes to fp32, then downcast for the next op.
        gate = torch.sigmoid(self.s_gate(s_norm))
        shift = self.s_shift(s_norm)
        return (gate * a_norm + shift).to(a.dtype)


# ===========================================================================
# ESMFold2FourierEmbedding
# ===========================================================================


class ESMFold2FourierEmbedding(nn.Module):
    """Fourier embedding: cos(2*pi*(t*w + b))."""

    w: Tensor
    b: Tensor

    def __init__(self, c: int) -> None:
        super().__init__()
        self.c = c
        self.register_buffer("w", torch.randn(c))
        self.register_buffer("b", torch.randn(c))

    def forward(self, t_hat: Tensor) -> Tensor:
        # w/b are kept fp32 (ESMFold2Model._keep_in_fp32_modules_strict), so the random
        # frequencies/phases — and the cos embedding — are computed at full precision.
        t = torch.as_tensor(t_hat, device=self.w.device, dtype=self.w.dtype).reshape(-1)
        return torch.cos(2.0 * torch.pi * (t[:, None] * self.w[None, :] + self.b[None, :]))


# ===========================================================================
# ESMFold2SwiGLU / ESMFold2SwiGLUMLP
# ===========================================================================


class ESMFold2SwiGLU(nn.Module):
    """ESMFold2SwiGLU with packed w12 and output w3."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.hidden_features = hidden_features

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.split(self.hidden_features, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class ESMFold2SwiGLUMLP(ESMFold2SwiGLU):
    """ESMFold2SwiGLU MLP with packed weights, no bias."""

    def __init__(self, d_model: int, expansion_ratio: int = 4, bias: bool = False) -> None:
        hidden = expansion_ratio * d_model
        super().__init__(in_features=d_model, hidden_features=hidden, out_features=d_model, bias=bias)


# ===========================================================================
# SWA Atom Attention components
# ===========================================================================


def apply_rotary_emb_3d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE with batch-dependent cos/sin.

    Args:
        x: [B, L, H, D]
        cos: [B, L, D/2]
        sin: [B, L, D/2]
    """
    ro_dim = cos.shape[-1] * 2
    cos = cos.unsqueeze(2).repeat(1, 1, 1, 2)
    sin = sin.unsqueeze(2).repeat(1, 1, 1, 2)
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim]) * sin, x[..., ro_dim:]],
        dim=-1,
    )


@torch.compiler.disable
def build_3d_rope(
    ref_pos: Tensor,
    ref_space_uid: Tensor,
    head_dim: int,
    n_spatial_per_axis: int = 4,
    n_uid_pairs: int = 2,
    spatial_base_freq: float = 10000.0,
    uid_base_freq: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """Build cos/sin for 3D RoPE + UID RoPE."""
    device = ref_pos.device
    B, N = ref_pos.shape[:2]
    half_dim = head_dim // 2
    n_spatial_total = 3 * n_spatial_per_axis

    spatial_inv_freq = 1.0 / (
        spatial_base_freq
        ** (torch.arange(0, n_spatial_per_axis, dtype=torch.float32, device=device) / n_spatial_per_axis)
    )
    uid_inv_freq = 1.0 / (
        uid_base_freq ** (torch.arange(0, n_uid_pairs, dtype=torch.float32, device=device) / n_uid_pairs)
    )

    pos_f32 = ref_pos.float()
    spatial_freqs = torch.einsum("bna,k->bnak", pos_f32, spatial_inv_freq)
    spatial_freqs = spatial_freqs.reshape(B, N, n_spatial_total)

    uid_f32 = ref_space_uid.float()
    uid_freqs = torch.einsum("bn,k->bnk", uid_f32, uid_inv_freq)

    n_active = n_spatial_total + n_uid_pairs
    freqs = torch.cat([spatial_freqs, uid_freqs], dim=-1)

    if n_active < half_dim:
        padding = torch.zeros(B, N, half_dim - n_active, device=device, dtype=torch.float32)
        freqs = torch.cat([freqs, padding], dim=-1)

    cos = freqs.cos().to(torch.bfloat16)
    sin = freqs.sin().to(torch.bfloat16)
    return cos, sin


def qk_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


# ===========================================================================
# ESMFold2SwiGLUFFN (atom transformer blocks)
# ===========================================================================


class ESMFold2SwiGLUFFN(nn.Module):
    """ESMFold2SwiGLU FFN with rounded hidden size for hardware alignment."""

    def __init__(self, d_model: int, expansion_ratio: int = 2) -> None:
        super().__init__()
        hidden_size = ((expansion_ratio * (d_model // 3) * 2) + 255) // 256 * 256
        self.w_up = nn.Linear(d_model, 2 * hidden_size, bias=False)
        self.w_down = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.w_up.weight.dtype)
        x1, x2 = self.w_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(x1) * x2)


# ===========================================================================
# ESMFold2SWA3DRoPEAttention
# ===========================================================================


class ESMFold2SWA3DRoPEAttention(nn.Module):
    """Sliding window self-attention with 3D RoPE. Has Wqkv, gate_proj, out_proj.

    The plain ``softmax(QKᵀ)V`` core is dispatched through the v5 attention
    interface (``config._attn_implementation``: ``eager`` / ``sdpa`` / ...),
    with the sliding window expressed as an additive attention mask. The custom
    flash-attention path (native bidirectional ``window_size``, plus varlen for
    packed inputs) is kept as an opt-in backend, selected when
    ``_attn_implementation == "flash_attention_2"``. ``config`` is attached by
    the parent ``ESMFold2Model`` after construction; it is ``None`` (→ ``sdpa``)
    when the module is used standalone.
    """

    def __init__(self, d_model: int, n_heads: int, half_window: int = 64) -> None:
        super().__init__()
        self.config = None
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.half_window = half_window
        # No grouped-query attention; identity repeat keeps the interface happy.
        self.num_key_value_groups = 1
        # Bidirectional encoder: never let the sdpa/flash interface default to
        # causal masking when attention_mask happens to be None.
        self.is_causal = False

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, attention_params: tuple) -> Tensor:
        B, N = x.shape[:2]
        cos, sin = attention_params[0], attention_params[1]

        x_input = x
        qkv = self.Wqkv(x)
        qkv = qkv.view(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv.unbind(0)
        q, k = qk_norm(q), qk_norm(k)

        q = apply_rotary_emb_3d(q, cos, sin)
        k = apply_rotary_emb_3d(k, cos, sin)

        input_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        attn_impl = self.config._attn_implementation if self.config is not None else "sdpa"
        use_flash = attn_impl == "flash_attention_2" and is_flash_attn_2_available()

        if use_flash and len(attention_params) > 2:
            indices, cu_seqlens, max_seqlen = (
                attention_params[2],
                attention_params[3],
                attention_params[4],
            )
            q_unpad = index_first_axis(q.reshape(-1, self.n_heads, self.head_dim), indices)
            k_unpad = index_first_axis(k.reshape(-1, self.n_heads, self.head_dim), indices)
            v_unpad = index_first_axis(v.reshape(-1, self.n_heads, self.head_dim), indices)
            out_unpad = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                softmax_scale=self.scale,
                window_size=(self.half_window, self.half_window),
            )
            out = pad_input(out_unpad, indices, B, N)
        elif use_flash:
            out = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=self.scale,
                window_size=(self.half_window, self.half_window),
            )
        else:
            if len(attention_params) > 2:
                valid = torch.zeros(B * N, dtype=torch.bool, device=q.device)
                valid[attention_params[2]] = True
                valid = valid.view(B, N)
            else:
                valid = torch.ones(B, N, dtype=torch.bool, device=q.device)
            rank = torch.cumsum(valid, dim=1) - 1
            within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= self.half_window
            allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
            allowed |= torch.eye(N, dtype=torch.bool, device=q.device)
            # Sliding window as an additive bias: 0 where allowed, -inf elsewhere.
            attn_mask = torch.zeros(B, 1, N, N, dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(q.dtype).min)

            attention_interface: Callable = eager_attention_forward
            if attn_impl != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(attn_impl, eager_attention_forward)
            out, _ = attention_interface(
                self,
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask,
                dropout=0.0,
                scaling=self.scale,
            )
            out = out * valid.unsqueeze(-1).unsqueeze(-1)

        out = out.to(input_dtype).reshape(B, N, -1)
        out = out * torch.sigmoid(self.gate_proj(x_input))
        return self.out_proj(out)


# ===========================================================================
# ESMFold2SWAAtomBlock, ESMFold2SWAAtomTransformer
# ===========================================================================


def _rms_adaln(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],)) * (1 + scale) + shift


def _gated_residual(x: Tensor, gate: Tensor, y: Tensor) -> Tensor:
    return x + gate * y


class ESMFold2SWAAtomBlock(nn.Module):
    """adaLN-Zero + SWA attention + ESMFold2SwiGLU FFN.

    Creates adaln_modulation = Sequential(SiLU(), Linear) -> keys like adaln_modulation.1.weight
    """

    def __init__(
        self,
        d_atom: int,
        n_heads: int,
        half_window: int = 64,
        expansion_ratio: int = 2,
    ) -> None:
        super().__init__()
        # adaln-Zero gate; zero-init lives in ESMFold2PreTrainedModel._init_weights.
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_atom, 6 * d_atom, bias=False))

        self.attn = ESMFold2SWA3DRoPEAttention(d_atom, n_heads, half_window=half_window)
        self.ffn = ESMFold2SwiGLUFFN(d_atom, expansion_ratio)

    def forward(self, x: Tensor, c_l: Tensor, attention_params: tuple) -> Tensor:
        mod = self.adaln_modulation(c_l)
        if mod.dim() == 2:
            mod = mod.unsqueeze(1)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = mod.chunk(6, dim=-1)

        attn_input = _rms_adaln(x, scale_a, shift_a)
        attn_out = self.attn(attn_input, attention_params)
        x = _gated_residual(x, gate_a, attn_out)

        ffn_input = _rms_adaln(x, scale_f, shift_f)
        ffn_out = self.ffn(ffn_input)
        x = _gated_residual(x, gate_f, ffn_out)
        return x


class ESMFold2SWAAtomTransformer(nn.Module):
    """Stack of SWAAtomBlocks."""

    def __init__(
        self,
        d_atom: int = 128,
        n_blocks: int = 3,
        n_heads: int = 4,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.swa_window_size = swa_window_size
        self.head_dim = d_atom // n_heads
        self.spatial_rope_base_frequency = spatial_rope_base_frequency
        self.n_spatial_rope_pairs_per_axis = n_spatial_rope_pairs_per_axis
        self.n_uid_rope_pairs = n_uid_rope_pairs
        self.uid_rope_base_frequency = uid_rope_base_frequency

        self.blocks = nn.ModuleList(
            [
                ESMFold2SWAAtomBlock(
                    d_atom=d_atom,
                    n_heads=n_heads,
                    half_window=swa_window_size // 2,
                    expansion_ratio=expansion_ratio,
                )
                for _ in range(n_blocks)
            ]
        )

    def _build_3d_rope(self, ref_pos: Tensor, ref_space_uid: Tensor) -> tuple[Tensor, Tensor]:
        return build_3d_rope(
            ref_pos=ref_pos,
            ref_space_uid=ref_space_uid,
            head_dim=self.head_dim,
            n_spatial_per_axis=self.n_spatial_rope_pairs_per_axis,
            n_uid_pairs=self.n_uid_rope_pairs,
            spatial_base_freq=self.spatial_rope_base_frequency,
            uid_base_freq=self.uid_rope_base_frequency,
        )

    def forward(
        self,
        q_l: Tensor,
        c_l: Tensor,
        attention_params: tuple,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        intermediates: list[Tensor] = []
        for block in self.blocks:
            q_l = block(q_l, c_l, attention_params)
            if return_intermediates:
                intermediates.append(q_l)
        if return_intermediates:
            return q_l, intermediates
        return q_l


# ===========================================================================
# ESMFold2AtomEncoder (for both inputs_embedder and diffusion_module)
# ===========================================================================


class ESMFold2AtomEncoder(nn.Module):
    """SWA atom encoder with atom_linear, atom_norm, atom_to_token_linear, [coords_linear], atom_transformer.

    Args:
        d_atom: atom hidden dim
        d_token: token dim for atom_to_token aggregation
        n_blocks, n_heads, swa_window_size, expansion_ratio: transformer params
        structure_prediction: if True, creates coords_linear and uses full d_token
        spatial_rope_base_frequency, n_spatial_rope_pairs_per_axis,
        n_uid_rope_pairs, uid_rope_base_frequency: 3D RoPE config
    """

    def __init__(
        self,
        d_atom: int = 128,
        d_token: int = 768,
        n_blocks: int = 3,
        n_heads: int = 4,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        structure_prediction: bool = True,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_atom = d_atom
        self.d_token = d_token
        self.structure_prediction = structure_prediction

        self.atom_linear = nn.Linear(ATOM_FEATURE_DIM, d_atom, bias=False)
        self.atom_norm = ESMFold2LayerNorm(d_atom)

        if structure_prediction:
            self.coords_linear = nn.Linear(6, d_atom, bias=False)

        self.atom_transformer = ESMFold2SWAAtomTransformer(
            d_atom=d_atom,
            n_blocks=n_blocks,
            n_heads=n_heads,
            swa_window_size=swa_window_size,
            expansion_ratio=expansion_ratio,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=n_uid_rope_pairs,
            uid_rope_base_frequency=uid_rope_base_frequency,
        )

        # Output aggregation: d_token for structure prediction, d_token//2 for inputs
        out_dim = d_token if structure_prediction else d_token // 2
        self.atom_to_token_linear = nn.Linear(d_atom, out_dim, bias=False)

    def forward(
        self,
        ref_pos: Tensor,
        atom_attention_mask: Tensor,
        ref_space_uid: Tensor,
        ref_charge: Tensor,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        atom_to_token: Tensor,
        r_l: Tensor | None = None,
        pred_r1: Tensor | None = None,
        num_diffusion_samples: int = 1,
        return_intermediates: bool = False,
        inference_cache: dict | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, tuple, list[Tensor]]:
        """Returns (a, q, c, attention_params, intermediates).

        ``inference_cache`` caches step-invariant tensors (c_base, 3D RoPE,
        attention indices, n_tokens) across diffusion steps.
        """
        B, N = ref_pos.shape[:2]

        layer_cache = None
        if inference_cache is not None:
            layer_cache = inference_cache.setdefault("atomencoder", {})

        if layer_cache is None or len(layer_cache) == 0:
            atom_feats = torch.cat(
                [
                    ref_pos,
                    ref_charge.unsqueeze(-1),
                    atom_attention_mask.unsqueeze(-1),
                    ref_element,
                    ref_atom_name_chars.reshape(B, N, MAX_CHARS * CHAR_VOCAB_SIZE),
                ],
                dim=-1,
            )
            c_base = self.atom_norm(self.atom_linear(atom_feats.to(self.atom_linear.weight.dtype)).float()).to(
                self.atom_linear.weight.dtype
            )
            cos, sin = self.atom_transformer._build_3d_rope(ref_pos, ref_space_uid)
            cos = cos.repeat_interleave(num_diffusion_samples, 0)
            sin = sin.repeat_interleave(num_diffusion_samples, 0)
            mask_exp = atom_attention_mask.repeat_interleave(num_diffusion_samples, 0)
            seqlens = mask_exp.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(mask_exp.flatten(), as_tuple=False).flatten()
            max_seqlen = int(seqlens.max().item())
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
            attention_params = (cos, sin, indices, cu_seqlens, max_seqlen)
            n_tokens = int(atom_to_token.max().item()) + 1
            if layer_cache is not None:
                layer_cache["c_base"] = c_base
                layer_cache["attention_params"] = attention_params
                layer_cache["mask_exp"] = mask_exp
                layer_cache["n_tokens"] = n_tokens
                layer_cache["atom_to_token_exp"] = atom_to_token.repeat_interleave(num_diffusion_samples, 0)
        else:
            c_base = layer_cache["c_base"]
            attention_params = layer_cache["attention_params"]
            mask_exp = layer_cache["mask_exp"]
            n_tokens = layer_cache["n_tokens"]

        c = c_base

        q = c

        if self.structure_prediction and r_l is not None:
            q = q.repeat_interleave(num_diffusion_samples, 0)
            if pred_r1 is None:
                pred_r1 = torch.zeros_like(r_l)
            r_input = torch.cat([r_l, pred_r1], dim=-1)
            r_to_q = self.coords_linear(r_input.to(self.coords_linear.weight.dtype))
            q = q + r_to_q

        c = c.repeat_interleave(num_diffusion_samples, 0)

        result = self.atom_transformer(
            q_l=q,
            c_l=c,
            attention_params=attention_params,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            q, intermediates = result
        else:
            q = result
            intermediates = []

        q_to_a = F.relu(self.atom_to_token_linear(q))
        if layer_cache is not None and "atom_to_token_exp" in layer_cache:
            atom_to_token_exp = layer_cache["atom_to_token_exp"]
        else:
            atom_to_token_exp = atom_to_token.repeat_interleave(num_diffusion_samples, 0)
        a = scatter_atom_to_token(q_to_a, atom_to_token_exp, n_tokens, atom_mask=mask_exp.bool())

        return a, q, c, attention_params, intermediates


# ===========================================================================
# ESMFold2AtomDecoder
# ===========================================================================


class ESMFold2AtomDecoder(nn.Module):
    """SWA atom decoder with token_to_atom_linear, atom_transformer, norm, output_linear."""

    def __init__(
        self,
        d_atom: int = 128,
        d_token: int = 768,
        n_blocks: int = 3,
        n_heads: int = 4,
        swa_window_size: int = 128,
        expansion_ratio: int = 2,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.token_to_atom_linear = nn.Linear(d_token, d_atom, bias=False)

        self.atom_transformer = ESMFold2SWAAtomTransformer(
            d_atom=d_atom,
            n_blocks=n_blocks,
            n_heads=n_heads,
            swa_window_size=swa_window_size,
            expansion_ratio=expansion_ratio,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=n_uid_rope_pairs,
            uid_rope_base_frequency=uid_rope_base_frequency,
        )

        self.norm = ESMFold2LayerNorm(d_atom)
        self.output_linear = nn.Linear(d_atom, XYZ_DIMS, bias=False)

    def forward(
        self,
        a_i: Tensor,
        q_l: Tensor,
        c_l: Tensor,
        p_lm: tuple,
        atom_to_token: Tensor,
        atom_attention_mask: Tensor,
        num_diffusion_samples: int = 1,
        return_intermediates: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        """Returns (r_update, intermediates)."""
        atom_to_token_exp = atom_to_token.repeat_interleave(num_diffusion_samples, 0)
        a_to_q = self.token_to_atom_linear(a_i)
        a_to_q = gather_token_to_atom(a_to_q, atom_to_token_exp)
        q_l = q_l + a_to_q

        result = self.atom_transformer(
            q_l=q_l,
            c_l=c_l,
            attention_params=p_lm,
            return_intermediates=return_intermediates,
        )
        if return_intermediates:
            q_l, intermediates = result
        else:
            q_l = result
            intermediates = []

        r_l = self.output_linear(self.norm(q_l))
        return r_l, intermediates


# ===========================================================================
# ESMFold2AttentionPairBias (ESMFold2DiffusionTransformer attention block)
# ===========================================================================


class ESMFold2AttentionPairBias(nn.Module):
    """Gated multi-head attention with pair bias conditioning."""

    def __init__(
        self,
        d_model: int,
        d_pair: int,
        num_heads: int,
        d_cond: int | None = None,
        use_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        d_cond = d_cond or d_model

        if use_conditioning:
            self.adaln = ESMFold2AdaptiveLayerNorm(d_model, d_cond, eps=1e-5)
            # adaln-Zero gate (weight 0, bias -2); init in ESMFold2PreTrainedModel._init_weights.
            self.out_gate = nn.Linear(d_cond, d_model, bias=True)
        else:
            self.pre_norm = ESMFold2LayerNorm(d_model, eps=1e-5)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if d_pair > 0:
            self.pair_norm = ESMFold2LayerNorm(d_pair, eps=1e-5)
            self.pair_bias_proj = nn.Linear(d_pair, num_heads, bias=False)

    def compute_pair_bias(self, z: Tensor, bsz: int, num_diffusion_samples: int = 1) -> Tensor:
        """Project the (normed) pair representation to per-head attention biases.

        Depends only on ``z`` and this block's fixed weights, so it is invariant
        across diffusion sampling steps — the sampler computes it once and reuses
        it (see ``ESMFold2DiffusionTransformer.forward``). Bit-identical to computing it
        inline every step.
        """
        if z.dim() == 4 and z.shape[0] != bsz and num_diffusion_samples > 1:
            z = z.repeat_interleave(num_diffusion_samples, dim=0)
        if z.dim() == 4:
            return self.pair_bias_proj(self.pair_norm(z))
        return z.unsqueeze(-1)

    def forward(
        self,
        a: Tensor,
        s: Tensor | None,
        z: Tensor,
        attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
        pair_bias: Tensor | None = None,
    ) -> Tensor:
        bsz, n_queries, d_model = a.shape

        if s is not None:
            x = self.adaln(a, s)
        else:
            x = self.pre_norm(a)

        n_keys = x.shape[1]
        q = self.q_proj(x).view(bsz, n_queries, self.num_heads, self.head_dim)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(bsz, n_keys, self.num_heads, self.head_dim)
        v = v.view(bsz, n_keys, self.num_heads, self.head_dim)

        if attention_mask is not None and attention_mask.shape[0] != bsz and num_diffusion_samples > 1:
            attention_mask = attention_mask.repeat_interleave(num_diffusion_samples, dim=0)

        # Standard attention with pair bias
        g = torch.sigmoid(self.g_proj(x)).view(bsz, n_queries, self.num_heads, self.head_dim)

        logits = torch.einsum("... i h d, ... j h d -> ... i j h", q, k) * self.scale

        # ``pair_bias`` is step-invariant; the diffusion sampler precomputes and
        # caches it across steps. Compute inline when not supplied (e.g. uncached).
        if pair_bias is None:
            pair_bias = self.compute_pair_bias(z, bsz, num_diffusion_samples)
        logits = logits + pair_bias.to(dtype=logits.dtype)

        if attention_mask is not None:
            min_val = torch.finfo(logits.dtype).min
            mask_bias = torch.where(attention_mask.bool()[:, None, :, None], 0.0, min_val)
            logits = logits + mask_bias.to(dtype=logits.dtype)

        attn = torch.softmax(logits, dim=-2, dtype=torch.float32).to(dtype=v.dtype)
        ctx = torch.einsum("... i j h, ... j h d -> ... i h d", attn, v)
        ctx = g * ctx
        out = self.out_proj(ctx.reshape(bsz, n_queries, d_model).to(v.dtype))

        if s is not None:
            out = torch.sigmoid(self.out_gate(s)) * out
        return out


# ===========================================================================
# ESMFold2ConditionedTransitionBlock
# ===========================================================================


class ESMFold2ConditionedTransitionBlock(nn.Module):
    """Conditioned ESMFold2SwiGLU transition with adaptive layer norm."""

    def __init__(
        self,
        d_model: int,
        d_cond: int | None = None,
        transition_multiplier: int = 2,
        use_conditioning: bool = True,
    ) -> None:
        super().__init__()
        d_cond = d_cond or d_model
        hidden = transition_multiplier * d_model

        if use_conditioning:
            self.adaln = ESMFold2AdaptiveLayerNorm(d_model, d_cond, eps=1e-5)
            # adaln-Zero gate (weight 0, bias -2); init in ESMFold2PreTrainedModel._init_weights.
            self.output_gate = nn.Linear(d_cond, d_model, bias=True)
        else:
            self.pre_norm = ESMFold2LayerNorm(d_model, eps=1e-5)

        self.lin_swish = nn.Linear(d_model, 2 * hidden, bias=False)
        self.lin_out = nn.Linear(hidden, d_model, bias=False)

    def forward(self, a: Tensor, s: Tensor | None) -> Tensor:
        if s is not None:
            x = self.adaln(a, s)
        else:
            x = self.pre_norm(a)

        swish_a, swish_b = self.lin_swish(x).chunk(2, dim=-1)
        b = F.silu(swish_a) * swish_b
        out = self.lin_out(b)

        if s is not None:
            out = torch.sigmoid(self.output_gate(s)) * out
        return out


# ===========================================================================
# ESMFold2DiffusionTransformer (token transformer)
# ===========================================================================


class ESMFold2DiffusionTransformer(nn.Module):
    """Diffusion denoising transformer with attention pair bias."""

    def __init__(
        self,
        d_model: int,
        d_pair: int,
        num_heads: int,
        num_blocks: int,
        d_cond: int | None = None,
        transition_multiplier: int = 2,
        use_conditioning: bool = True,
    ) -> None:
        super().__init__()
        d_cond = d_cond or d_model

        self.attn_blocks = nn.ModuleList(
            [
                ESMFold2AttentionPairBias(
                    d_model=d_model,
                    d_pair=d_pair,
                    num_heads=num_heads,
                    d_cond=d_cond,
                    use_conditioning=use_conditioning,
                )
                for _ in range(num_blocks)
            ]
        )
        self.transition_blocks = nn.ModuleList(
            [
                ESMFold2ConditionedTransitionBlock(
                    d_model=d_model,
                    d_cond=d_cond,
                    transition_multiplier=transition_multiplier,
                    use_conditioning=use_conditioning,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        a: Tensor,
        s: Tensor | None,
        z: Tensor,
        attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
        return_intermediates: bool = False,
        inference_cache: dict | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        intermediates: list[Tensor] = []
        x = a
        bsz = a.shape[0]
        # Each block's pair bias depends only on the (step-invariant) conditioning
        # pair ``z`` and fixed weights, so compute it once per block and reuse it
        # across every diffusion sampling step. Bit-identical to recomputing it
        # each step; the cache lives in the sampler's per-fold ``inference_cache``.
        bias_cache = None if inference_cache is None else inference_cache.setdefault("token_pair_bias", {})
        for i, (attn, transition) in enumerate(zip(self.attn_blocks, self.transition_blocks)):
            if bias_cache is None:
                pair_bias = attn.compute_pair_bias(z, bsz, num_diffusion_samples)
            elif i in bias_cache:
                pair_bias = bias_cache[i]
            else:
                pair_bias = attn.compute_pair_bias(z, bsz, num_diffusion_samples)
                bias_cache[i] = pair_bias
            x = x + attn(
                x,
                s,
                z,
                attention_mask=attention_mask,
                num_diffusion_samples=num_diffusion_samples,
                pair_bias=pair_bias,
            )
            x = x + transition(x, s)
            if return_intermediates:
                intermediates.append(x)
        return x, intermediates


# ===========================================================================
# ESMFold2DiffusionConditioning
# ===========================================================================


class ESMFold2DiffusionConditioning(nn.Module):
    """Conditions pair and single representations on noise timestep."""

    def __init__(
        self,
        c_z: int = 256,
        c_s: int = 768,
        c_s_inputs: int = 451,
        sigma_data: float = 16.0,
        fourier_dim: int = 256,
        transition_multiplier: int = 2,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.sigma_data = float(sigma_data)
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs

        self.z_input_norm = ESMFold2LayerNorm(2 * c_z, eps=layer_norm_eps)
        self.z_proj = nn.Linear(2 * c_z, c_z, bias=False)
        self.z_transitions = nn.ModuleList(
            [ESMFold2TransitionLayer(c_z, n=transition_multiplier, eps=layer_norm_eps) for _ in range(2)]
        )

        self.s_input_norm = ESMFold2LayerNorm(c_s_inputs, eps=layer_norm_eps)
        self.s_proj = nn.Linear(c_s_inputs, c_s, bias=False)
        self.fourier = ESMFold2FourierEmbedding(fourier_dim)
        self.noise_norm = ESMFold2LayerNorm(fourier_dim, eps=layer_norm_eps)
        self.noise_proj = nn.Linear(fourier_dim, c_s, bias=False)
        self.s_transitions = nn.ModuleList(
            [ESMFold2TransitionLayer(c_s, n=transition_multiplier, eps=layer_norm_eps) for _ in range(2)]
        )

    def forward(
        self,
        t_hat: Tensor,
        s_inputs: Tensor,
        z_trunk: Tensor,
        relative_position_encoding: Tensor,
        sigma_data: float | None = None,
        num_diffusion_samples: int = 1,
        inference_cache: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        sigma = self.sigma_data if sigma_data is None else float(sigma_data)
        base_batch = z_trunk.shape[0]
        target_batch = base_batch * num_diffusion_samples

        # z conditioning (cached across diffusion steps — independent of t_hat)
        if inference_cache is not None and "z" in inference_cache:
            z = inference_cache["z"]
        else:
            z_rel = relative_position_encoding.to(dtype=torch.float32)
            z = torch.cat([z_trunk.to(dtype=torch.float32), z_rel], dim=-1)
            # The relpos/coords conditioning is fp32; z_input_norm keeps it fp32,
            # then we hand off to z_proj in the model's compute dtype.
            z = self.z_proj(self.z_input_norm(z).to(self.z_proj.weight.dtype))
            for block in self.z_transitions:
                z = z + block(z)
            if inference_cache is not None:
                inference_cache["z"] = z

        # s conditioning
        s_inputs_eff = s_inputs
        if s_inputs_eff.shape[0] != target_batch:
            s_inputs_eff = s_inputs_eff.repeat_interleave(num_diffusion_samples, 0)

        s = self.s_proj(self.s_input_norm(s_inputs_eff.to(dtype=torch.float32)).to(self.s_proj.weight.dtype))

        # Noise embedding
        t = torch.as_tensor(t_hat, dtype=torch.float32, device=s.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(target_batch)
        elif t.shape[0] != target_batch:
            t = t.repeat_interleave(num_diffusion_samples, 0)
        t_noise = 0.25 * torch.log((t / sigma).clamp(min=1e-20))
        n = self.fourier(t_noise)
        n = self.noise_proj(self.noise_norm(n.float()).to(self.noise_proj.weight.dtype))
        s = s + n.unsqueeze(1)

        for block in self.s_transitions:
            s = s + block(s)

        return s, z


# ===========================================================================
# ESMFold2DiffusionModule
# ===========================================================================


class ESMFold2DiffusionModule(nn.Module):
    """Diffusion denoising module for structure prediction."""

    def __init__(
        self,
        c_atom: int = 128,
        c_token: int = 768,
        c_z: int = 256,
        c_s_inputs: int = 451,
        sigma_data: float = 16.0,
        fourier_dim: int = 256,
        atom_num_blocks: int = 3,
        atom_num_heads: int = 4,
        token_num_blocks: int = 12,
        token_num_heads: int = 16,
        transition_multiplier: int = 2,
        swa_window_size: int = 128,
        spatial_rope_base_frequency: float = 20.0,
        n_spatial_rope_pairs_per_axis: int = 2,
        n_uid_rope_pairs: int = 10,
        uid_rope_base_frequency: float = 10000.0,
    ) -> None:
        super().__init__()
        self.sigma_data = float(sigma_data)

        self.conditioning = ESMFold2DiffusionConditioning(
            c_z=c_z,
            c_s=c_token,  # conditioning s output is c_token
            c_s_inputs=c_s_inputs,
            sigma_data=sigma_data,
            fourier_dim=fourier_dim,
            transition_multiplier=transition_multiplier,
        )

        # Atom encoder (structure_prediction=True, with coords_linear)
        self.atom_encoder = ESMFold2AtomEncoder(
            d_atom=c_atom,
            d_token=c_token,
            n_blocks=atom_num_blocks,
            n_heads=atom_num_heads,
            swa_window_size=swa_window_size,
            expansion_ratio=2,
            structure_prediction=True,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=n_uid_rope_pairs,
            uid_rope_base_frequency=uid_rope_base_frequency,
        )

        # Atom decoder
        self.atom_decoder = ESMFold2AtomDecoder(
            d_atom=c_atom,
            d_token=c_token,
            n_blocks=atom_num_blocks,
            n_heads=atom_num_heads,
            swa_window_size=swa_window_size,
            expansion_ratio=2,
            spatial_rope_base_frequency=spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=n_uid_rope_pairs,
            uid_rope_base_frequency=uid_rope_base_frequency,
        )

        # zero-init lives in ESMFold2PreTrainedModel._init_weights.
        self.s_to_token = nn.Linear(c_token, c_token, bias=False)

        # Token transformer (ESMFold2DiffusionTransformer with pair bias)
        self.token_transformer = ESMFold2DiffusionTransformer(
            d_model=c_token,
            d_pair=c_z,
            num_heads=token_num_heads,
            num_blocks=token_num_blocks,
            d_cond=c_token,
            transition_multiplier=transition_multiplier,
            use_conditioning=True,
        )

        self.s_step_norm = ESMFold2LayerNorm(c_token)
        self.token_norm = ESMFold2LayerNorm(c_token)

    def forward(
        self,
        x_noisy: Tensor,
        t_hat: Tensor,
        ref_pos: Tensor,
        ref_charge: Tensor,
        ref_mask: Tensor,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        tok_idx: Tensor,
        s_inputs: Tensor,
        z_trunk: Tensor,
        relative_position_encoding: Tensor,
        asym_id: Tensor,
        residue_index: Tensor,
        entity_id: Tensor,
        token_index: Tensor,
        sym_id: Tensor,
        sigma_data: float | None = None,
        token_attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
        return_atom_repr: bool = False,
        inference_cache: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor | None]:
        bsz = x_noisy.shape[0]
        sigma = self.sigma_data if sigma_data is None else float(sigma_data)
        t = torch.as_tensor(t_hat, dtype=torch.float32, device=x_noisy.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(bsz)

        # Step 1: conditioning (pair z is cached across diffusion steps)
        s, z = self.conditioning(
            t_hat=t,
            s_inputs=s_inputs,
            z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
            sigma_data=sigma,
            num_diffusion_samples=num_diffusion_samples,
            inference_cache=inference_cache,
        )

        # Step 2: normalize noisy coords
        denom = torch.sqrt(t * t + sigma * sigma)
        r_noisy = x_noisy / denom[:, None, None]

        # Step 3: atom encoder
        a, q_skip, c_skip, p_skip, enc_intermediates = self.atom_encoder(
            ref_pos=ref_pos,
            atom_attention_mask=ref_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            atom_to_token=tok_idx,
            r_l=r_noisy,
            num_diffusion_samples=num_diffusion_samples,
            return_intermediates=return_atom_repr,
            inference_cache=inference_cache,
        )

        # Step 4: add conditioned s
        a = a + self.s_to_token(self.s_step_norm(s))

        # Step 5: token transformer (pair bias is cached across steps via inference_cache)
        a, _ = self.token_transformer(
            a,
            s,
            z,
            attention_mask=token_attention_mask,
            num_diffusion_samples=num_diffusion_samples,
            inference_cache=inference_cache,
        )

        # Step 6: token norm
        a = self.token_norm(a)

        # Step 7: atom decoder
        r_update, dec_intermediates = self.atom_decoder(
            a_i=a,
            q_l=q_skip,
            c_l=c_skip,
            p_lm=p_skip,
            atom_to_token=tok_idx,
            atom_attention_mask=ref_mask,
            num_diffusion_samples=num_diffusion_samples,
            return_intermediates=return_atom_repr,
        )

        # Step 8: compute denoised output
        sigma2 = sigma * sigma
        t2 = t * t
        out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
        out = out + ((sigma * t) / torch.sqrt(sigma2 + t2))[:, None, None] * r_update

        # Collect atom intermediates from encoder + decoder
        atom_intermediates: Tensor | None = None
        if return_atom_repr:
            all_ints = enc_intermediates + dec_intermediates
            if all_ints:
                atom_intermediates = torch.stack(all_ints, dim=2)

        return {
            "x_denoised": out,
            "atom_intermediates": atom_intermediates,
        }


# ===========================================================================
# ESMFold2DiffusionStructureHead
# ===========================================================================


class ESMFold2DiffusionStructureHead(nn.Module):
    """Wrapper around ESMFold2DiffusionModule with diffusion sampling."""

    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__()
        dm = config.structure_head.diffusion_module
        swa_cfg = config.inputs.atom_encoder
        sh = config.structure_head

        self.diffusion_module = ESMFold2DiffusionModule(
            c_atom=dm.c_atom,
            c_token=dm.c_token,
            c_z=dm.c_z,
            c_s_inputs=dm.c_s_inputs,
            sigma_data=dm.sigma_data,
            fourier_dim=dm.fourier_dim,
            atom_num_blocks=dm.atom_num_blocks,
            atom_num_heads=dm.atom_num_heads,
            token_num_blocks=dm.token_num_blocks,
            token_num_heads=dm.token_num_heads,
            transition_multiplier=dm.transition_multiplier,
            swa_window_size=swa_cfg.swa_window_size,
            spatial_rope_base_frequency=swa_cfg.spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=swa_cfg.n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=swa_cfg.n_uid_rope_pairs,
            uid_rope_base_frequency=swa_cfg.uid_rope_base_frequency,
        )

        # Sampling hyperparameters
        self.sigma_data = dm.sigma_data
        self.gamma_0 = sh.gamma_0
        self.gamma_min = sh.gamma_min
        self.noise_scale = sh.noise_scale
        self.step_scale = sh.step_scale
        self.inference_s_max = sh.inference_s_max
        self.inference_s_min = sh.inference_s_min
        self.inference_p = sh.inference_p
        self.inference_num_steps = sh.inference_num_steps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def inference_noise_schedule(self, num_steps: int | None = None, device: torch.device | None = None) -> Tensor:
        """Karras power-law noise schedule."""
        steps = self.inference_num_steps if num_steps is None else int(num_steps)
        if steps == 1:
            return torch.tensor(
                [self.inference_s_max * self.sigma_data, 0.0],
                device=device,
                dtype=torch.float32,
            )
        p = float(self.inference_p)
        inv_p = 1.0 / p
        k = torch.arange(steps, device=device, dtype=torch.float32)
        base = self.inference_s_max**inv_p + (k / (steps - 1)) * (
            self.inference_s_min**inv_p - self.inference_s_max**inv_p
        )
        schedule = self.sigma_data * base.pow(p)
        return F.pad(schedule, (0, 1), value=0.0)

    @staticmethod
    def _random_rotations(n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        q = torch.randn((n, 4), dtype=dtype, device=device)
        scale = torch.sqrt((q * q).sum(dim=1))
        signs = torch.where(q[:, 0] < 0, -scale, scale)
        q = q / signs[:, None]
        r, i, j, k = torch.unbind(q, dim=-1)
        two_s = 2.0 / (q * q).sum(dim=-1)
        return torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            dim=-1,
        ).reshape(n, 3, 3)

    def _center_random_augmentation(
        self, x: Tensor, atom_mask: Tensor, second_coords: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Algorithm 19: center + random rotation + translation."""
        bsz = x.shape[0]
        mask = atom_mask.unsqueeze(-1)  # [B, A, 1]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (x * mask).sum(dim=1, keepdim=True) / denom
        x = x - mean
        if second_coords is not None:
            second_coords = second_coords - mean

        r = self._random_rotations(bsz, x.dtype, x.device)
        x = torch.einsum("bmd,bds->bms", x, r)
        if second_coords is not None:
            second_coords = torch.einsum("bmd,bds->bms", second_coords, r)

        t = torch.randn_like(x[:, 0:1, :])
        x = x + t
        if second_coords is not None:
            second_coords = second_coords + t
        return x, second_coords

    @staticmethod
    def _weighted_rigid_align(x: Tensor, x_gt: Tensor, w: Tensor, mask: Tensor) -> Tensor:
        """Kabsch alignment: align x to x_gt with weights w."""
        w = (mask * w).unsqueeze(-1)  # [B, N, 1]
        denom = w.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        mu = (x * w).sum(dim=-2, keepdim=True) / denom
        mu_gt = (x_gt * w).sum(dim=-2, keepdim=True) / denom
        x_c = x - mu
        xgt_c = x_gt - mu_gt
        H = torch.einsum("bni,bnj->bij", w * xgt_c, x_c)
        H32 = H.float()
        U, _, Vh = torch.linalg.svd(H32, driver="gesvd" if H32.is_cuda else None)
        det = torch.linalg.det(U @ Vh)
        ones = torch.ones_like(det)
        R = (U @ torch.diag_embed(torch.stack([ones, ones, det], dim=-1)) @ Vh).to(H.dtype)
        return x_c @ R.transpose(-1, -2) + mu_gt

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample(
        self,
        z_trunk: Tensor,
        s_inputs: Tensor,
        relative_position_encoding: Tensor,
        ref_pos: Tensor,
        ref_charge: Tensor,
        ref_mask: Tensor,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        tok_idx: Tensor,
        asym_id: Tensor,
        residue_index: Tensor,
        entity_id: Tensor,
        token_index: Tensor,
        sym_id: Tensor,
        token_attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
        num_sampling_steps: int | None = None,
        max_inference_sigma: float | None = 256.0,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        return_atom_repr: bool = False,
        use_inference_cache: bool = True,
        denoising_early_exit_rmsd: float | None = None,
    ) -> dict[str, Tensor | None]:
        """Diffusion sampling (Algorithm 18).

        ``num_sampling_steps`` is the number of denoising steps actually run.
        When ``max_inference_sigma`` is set, the Karras schedule built with
        ``num_sampling_steps`` entries would lose its high-σ tail to the cap,
        so we inflate the underlying schedule length here to land back at the
        requested step count post-truncation.
        """
        n_atoms = tok_idx.shape[1]
        device = s_inputs.device
        target_batch = s_inputs.shape[0] * num_diffusion_samples

        inference_cache: dict[str, Tensor] | None = {} if use_inference_cache else None

        steps = self.inference_num_steps if num_sampling_steps is None else int(num_sampling_steps)

        schedule = self.inference_noise_schedule(steps, device)
        if max_inference_sigma is not None:
            schedule = schedule[schedule <= float(max_inference_sigma)]
            schedule = F.pad(schedule, (1, 0), value=float(max_inference_sigma))

        lam = self.noise_scale if noise_scale is None else float(noise_scale)
        eta = self.step_scale if step_scale is None else float(step_scale)

        x = schedule[0] * torch.randn(target_batch, n_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = ref_mask.repeat_interleave(num_diffusion_samples, 0).float()

        gammas = torch.where(
            schedule > self.gamma_min,
            torch.full_like(schedule, self.gamma_0),
            torch.zeros_like(schedule),
        )

        x_denoised_prev: Tensor | None = None
        diff_atom_intermediates: Tensor | None = None

        step_pairs = list(zip(schedule[:-1], schedule[1:], gammas[1:]))
        num_steps = len(step_pairs)

        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(step_pairs):
            x, x_denoised_prev = self._center_random_augmentation(x, atom_mask, second_coords=x_denoised_prev)

            sigma_tm_val = float(sigma_tm.item())
            t_hat_val = sigma_tm_val * (1.0 + float(gamma.item()))
            eps_std = lam * max(t_hat_val**2 - sigma_tm_val**2, 0.0) ** 0.5
            x_noisy = x + eps_std * torch.randn_like(x)

            is_last_step = step_idx == num_steps - 1
            request_atom_repr = return_atom_repr and (is_last_step or denoising_early_exit_rmsd is not None)

            dm_out = self.diffusion_module(
                x_noisy=x_noisy,
                t_hat=torch.full((target_batch,), t_hat_val, device=device, dtype=torch.float32),
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_mask=ref_mask,
                ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars,
                ref_space_uid=ref_space_uid,
                tok_idx=tok_idx,
                s_inputs=s_inputs,
                z_trunk=z_trunk,
                relative_position_encoding=relative_position_encoding,
                asym_id=asym_id,
                residue_index=residue_index,
                entity_id=entity_id,
                token_index=token_index,
                sym_id=sym_id,
                token_attention_mask=token_attention_mask,
                num_diffusion_samples=num_diffusion_samples,
                return_atom_repr=request_atom_repr,
                inference_cache=inference_cache,
            )

            x_denoised = dm_out["x_denoised"]
            if request_atom_repr:
                diff_atom_intermediates = dm_out.get("atom_intermediates")

            # Reverse diffusion alignment (Kabsch). _weighted_rigid_align upcasts
            # to fp32 internally for the SVD/det.
            x_noisy = self._weighted_rigid_align(x_noisy.float(), x_denoised.float(), atom_mask, atom_mask)
            x_noisy = x_noisy.to(dtype=x_denoised.dtype)

            # ODE/SDE step
            sigma_t_val = float(sigma_t.item())
            denoised_over_sigma = (x_noisy - x_denoised) / t_hat_val
            x = x_noisy + eta * (sigma_t_val - t_hat_val) * denoised_over_sigma

            # Denoising early-exit: stop when consecutive predictions converge
            if denoising_early_exit_rmsd is not None and x_denoised_prev is not None and step_idx >= 1:
                aligned = self._weighted_rigid_align(
                    x_denoised_prev.float(),
                    x_denoised.float(),
                    atom_mask,
                    atom_mask,
                )
                diff = (x_denoised.float() - aligned) * atom_mask.unsqueeze(-1)
                per_sample_rmsd = (diff.pow(2).sum(dim=(-1, -2)) / atom_mask.sum(dim=-1).clamp(min=1)).sqrt()
                if per_sample_rmsd.max().item() < denoising_early_exit_rmsd:
                    x = x_denoised
                    x_denoised_prev = x_denoised
                    break

            x_denoised_prev = x_denoised

        result: dict[str, Tensor | None] = {
            "sample_atom_coords": x,
        }
        if return_atom_repr:
            result["diff_atom_intermediates"] = diff_atom_intermediates
        return result


# ===========================================================================
# ESMFold2RowAttentionPooling
# ===========================================================================


class ESMFold2RowAttentionPooling(nn.Module):
    """Row-wise attention pooling: attn_proj, out_proj."""

    def __init__(self, d_pair: int, d_single: int) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(d_pair, 1, bias=False)
        self.out_proj = nn.Linear(d_pair, d_single, bias=False)

    def forward(self, z: Tensor, mask: Tensor) -> Tensor:
        scores = self.attn_proj(z).squeeze(-1)
        mask_bias = torch.where(
            mask[:, None, :].bool(),
            torch.zeros_like(scores),
            torch.full_like(scores, -1e9),
        )
        scores = scores + mask_bias
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        pooled = torch.einsum("bnm,bnmd->bnd", weights, z)
        return self.out_proj(pooled)


# ===========================================================================
# ESMFold2InputsEmbedder
# ===========================================================================


class ESMFold2InputsEmbedder(nn.Module):
    """Embeds input features including atom-level encoding via SWA attention."""

    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__()
        swa_cfg = config.inputs.atom_encoder

        self.atom_attention_encoder = ESMFold2AtomEncoder(
            d_atom=swa_cfg.d_atom,
            d_token=swa_cfg.d_token,
            n_blocks=swa_cfg.n_blocks,
            n_heads=swa_cfg.n_heads,
            swa_window_size=swa_cfg.swa_window_size,
            expansion_ratio=swa_cfg.expansion_ratio,
            structure_prediction=False,  # no coords_linear
            spatial_rope_base_frequency=swa_cfg.spatial_rope_base_frequency,
            n_spatial_rope_pairs_per_axis=swa_cfg.n_spatial_rope_pairs_per_axis,
            n_uid_rope_pairs=swa_cfg.n_uid_rope_pairs,
            uid_rope_base_frequency=swa_cfg.uid_rope_base_frequency,
        )

    def forward(
        self,
        aatype: Tensor,
        profile: Tensor,
        deletion_mean: Tensor,
        ref_pos: Tensor,
        atom_attention_mask: Tensor,
        ref_space_uid: Tensor,
        ref_charge: Tensor,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        atom_to_token: Tensor,
    ) -> Tensor:
        """Embed inputs into per-token features.

        Returns:
            [B, L, d_inputs] concatenation of atom encoding, aatype, profile,
            and deletion_mean.
        """
        a, _q, _c, _attn_params, _intermediates = self.atom_attention_encoder(
            ref_pos=ref_pos,
            atom_attention_mask=atom_attention_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            atom_to_token=atom_to_token,
        )
        # The continuous input features are fp32; fold them into the atom
        # encoding's (compute) dtype so the single representation is one dtype.
        dtype = a.dtype
        return torch.cat(
            [a, aatype.to(dtype), profile.to(dtype), deletion_mean.unsqueeze(-1).to(dtype)],
            dim=-1,
        )


# ===========================================================================
# ESMFold2ResIdxAsymIdSymIdEntityIdEncoding (trunk relative position)
# ===========================================================================


class ESMFold2ResIdxAsymIdSymIdEntityIdEncoding(nn.Module):
    """embed.weight [d_pair, n_features] where n_features = 2*(2*r_bins+2) + 1 + (2*c_bins+2).

    For default r_bins=32, c_bins=2: 2*66 + 1 + 6 = 139.
    """

    def __init__(
        self,
        n_relative_residx_bins: int = 32,
        n_relative_chain_bins: int = 2,
        d_pair: int = 256,
    ) -> None:
        super().__init__()
        self.n_relative_residx_bins = n_relative_residx_bins
        self.n_relative_chain_bins = n_relative_chain_bins
        self.d_pair = d_pair

        n_feats_residue = 2 * n_relative_residx_bins + 2
        n_feats_token = 2 * n_relative_residx_bins + 2
        n_feats_chain = 2 * n_relative_chain_bins + 2
        n_feats_same_entity = 1
        total_feats = n_feats_residue + n_feats_token + n_feats_chain + n_feats_same_entity
        self.embed = nn.Linear(total_feats, d_pair, bias=False)

    def forward(
        self,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        token_index: Tensor,
    ) -> Tensor:
        bij_same_chain = asym_id.unsqueeze(2) == asym_id.unsqueeze(1)
        bij_same_residue = residue_index.unsqueeze(2) == residue_index.unsqueeze(1)
        bij_same_entity = entity_id.unsqueeze(2) == entity_id.unsqueeze(1)

        dij_residue = residue_index.unsqueeze(2) - residue_index.unsqueeze(1)
        dij_residue = torch.clip(
            dij_residue + self.n_relative_residx_bins,
            0,
            2 * self.n_relative_residx_bins,
        )
        dij_residue = torch.where(bij_same_chain, dij_residue, 2 * self.n_relative_residx_bins + 1)
        aij_rel_pos = F.one_hot(dij_residue, 2 * self.n_relative_residx_bins + 2)

        dij_token = torch.clip(
            token_index.unsqueeze(2) - token_index.unsqueeze(1) + self.n_relative_residx_bins,
            0,
            2 * self.n_relative_residx_bins,
        )
        dij_token = torch.where(
            bij_same_chain & bij_same_residue,
            dij_token,
            2 * self.n_relative_residx_bins + 1,
        )
        aij_rel_token = F.one_hot(dij_token, 2 * self.n_relative_residx_bins + 2)

        dij_chain = torch.clip(
            sym_id.unsqueeze(2) - sym_id.unsqueeze(1) + self.n_relative_chain_bins,
            0,
            2 * self.n_relative_chain_bins,
        )
        dij_chain = torch.where(bij_same_chain, 2 * self.n_relative_chain_bins + 1, dij_chain)
        aij_rel_chain = F.one_hot(dij_chain, 2 * self.n_relative_chain_bins + 2)

        feats = torch.cat(
            [
                aij_rel_pos.float(),
                aij_rel_token.float(),
                bij_same_entity.float().unsqueeze(-1),
                aij_rel_chain.float(),
            ],
            dim=-1,
        )

        return self.embed(feats.to(self.embed.weight.dtype))


# ===========================================================================
# ESMFold2SingleToPair (for ESMFold2LanguageModelShim)
# ===========================================================================


class ESMFold2SingleToPair(nn.Module):
    """downproject, output_mlp (Sequential of Linear, GELU, Linear)."""

    def __init__(self, input_dim: int, downproject_dim: int, output_dim: int) -> None:
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * downproject_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.downproject(x)
        x = torch.cat(
            [(x.unsqueeze(2) * x.unsqueeze(1)), (x.unsqueeze(2) - x.unsqueeze(1))],
            dim=3,
        )
        return self.output_mlp(x)


# ===========================================================================
# ESMFold2LanguageModelShim
# ===========================================================================


class ESMFold2LanguageModelShim(nn.Module):
    """Shim holding the trainable projection weights for LM integration.

    Contains:
    - base_z_combine: nn.Parameter [num_layers+1]
    - base_z_linear: Sequential(ESMFold2LayerNorm(d_model), Linear(d_model, d_z, bias=False))
    - base_z_mlp: Sequential(ESMFold2SingleToPair(d_z, d_z, d_z), ESMFold2LayerNorm(d_z))
    """

    def __init__(self, d_z: int = 256, d_model: int = 2560, num_layers: int = 80) -> None:
        super().__init__()

        self.base_z_mlp = nn.Sequential(ESMFold2SingleToPair(d_z, d_z, d_z), ESMFold2LayerNorm(d_z))
        self.base_z_linear = nn.Sequential(ESMFold2LayerNorm(d_model), nn.Linear(d_model, d_z, bias=False))
        self.base_z_combine = nn.Parameter(torch.zeros(num_layers + 1))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project pre-computed ESMC hidden states to pair representation.

        Args:
            hidden_states: [B, L, num_layers+1, d_model] from ESMC 6B.

        Returns:
            [B, L, L, d_pair] pair representation.
        """
        # The ESMC backbone may be loaded at a different precision than the trunk
        # (e.g. bf16 backbone with an fp32 trunk); align to the projection dtype.
        hidden_states = hidden_states.to(self.base_z_linear[1].weight.dtype)
        # base_z_linear[0] is an fp32-pinned LayerNorm; upcast in, downcast out.
        normed = self.base_z_linear[0](hidden_states)
        lm_z = self.base_z_linear[1](normed)  # [B, L, 81, d_z]
        weights = self.base_z_combine.softmax(0)  # [81]
        lm_z = (weights @ lm_z).squeeze(-2)  # [B, L, d_z]
        # base_z_mlp[1] is an fp32-pinned LayerNorm; upcast in, downcast out.
        pair = self.base_z_mlp[0](lm_z)
        lm_z = self.base_z_mlp[1](pair)  # [B, L, L, d_z]
        return lm_z


# ===========================================================================
# ESMFold2 — language-model backbone helpers
# ===========================================================================


def compute_lm_hidden_states(
    esmc: nn.Module,
    input_ids: Tensor,
    asym_id: Tensor,
    residue_index: Tensor,
    mol_type: Tensor,
    token_mask: Tensor,
    pad_to_multiple: int | None = None,
) -> Tensor:
    """Run ESMC with BOS/EOS wrapping, return hidden states [B, L, N, D] with N=81 layers.

    Atom-tokenized modified residues (HYP, MSE, ACE, NH2, ...) span multiple
    structure tokens but share a single ``(asym_id, residue_index)`` key —
    collapse them to one LM token per residue before running the LM (the LM
    was trained on per-residue inputs, not per-atom), then scatter the
    hidden states back to the per-token layout.
    """
    B, L = input_ids.shape
    device = input_ids.device
    protein_mask = (mol_type == 0) & token_mask

    lm_input_list = []
    lm_lengths = []
    # Per-batch maps from (original protein-token index) to (LM input position).
    expand_maps: list[Tensor] = []
    for b in range(B):
        mask_b = protein_mask[b]
        ids_b = input_ids[b][mask_b]
        asym_b = asym_id[b][mask_b]
        res_b = residue_index[b][mask_b]

        # Collapse: keep first token per (asym_id, residue_index) key, in
        # input order. ``inverse`` maps each original protein-token to its
        # collapsed residue index.
        keys = torch.stack((asym_b, res_b), dim=1)
        unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
        n_unique = unique_keys.size(0)
        token_positions = torch.arange(keys.size(0), device=device, dtype=torch.long)
        first_pos = torch.full((n_unique,), keys.size(0), device=device, dtype=torch.long)
        first_pos.scatter_reduce_(0, inverse, token_positions, reduce="amin", include_self=True)
        ordered = torch.argsort(first_pos)
        first_pos_ordered = first_pos[ordered]
        ids_collapsed = ids_b[first_pos_ordered]
        asym_collapsed = asym_b[first_pos_ordered]
        remap = torch.empty_like(ordered)
        remap[ordered] = torch.arange(n_unique, device=device, dtype=torch.long)
        inverse_ordered = remap[inverse]

        chain_ids = asym_collapsed.unique(sorted=True)
        # [BOS] chain1 [EOS BOS] chain2 ... [EOS]
        parts: list[Tensor] = [torch.tensor([0], device=device, dtype=ids_b.dtype)]
        # Per-chain LM positions accumulate; track them for the expand map.
        per_token_lm_pos = torch.empty(n_unique, device=device, dtype=torch.long)
        cursor = 1  # position 0 is the leading BOS
        for i, cid in enumerate(chain_ids):
            in_chain = (asym_collapsed == cid).nonzero(as_tuple=True)[0]
            parts.append(ids_collapsed[in_chain])
            per_token_lm_pos[in_chain] = torch.arange(
                cursor, cursor + in_chain.shape[0], device=device, dtype=torch.long
            )
            cursor += in_chain.shape[0]
            if i < len(chain_ids) - 1:
                parts.append(torch.tensor([2, 0], device=device, dtype=ids_b.dtype))
                cursor += 2  # EOS + BOS
        parts.append(torch.tensor([2], device=device, dtype=ids_b.dtype))
        lm_seq = torch.cat(parts)
        lm_input_list.append(lm_seq)
        lm_lengths.append(lm_seq.shape[0])

        # Original protein-token position → LM input position.
        prot_pos_b = mask_b.nonzero(as_tuple=True)[0]
        expand_map = torch.full((L,), -1, device=device, dtype=torch.long)
        expand_map[prot_pos_b] = per_token_lm_pos[inverse_ordered]
        expand_maps.append(expand_map)

    # Pad to longest LM input, optionally rounding up to ``pad_to_multiple``.
    max_len = max(lm_lengths)
    if pad_to_multiple is not None and pad_to_multiple > 1:
        max_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    lm_input_ids = torch.full(
        (B, max_len),
        1,
        device=device,
        dtype=input_ids.dtype,  # PAD=1
    )
    for b in range(B):
        lm_input_ids[b, : lm_lengths[b]] = lm_input_list[b]

    # sequence_id for chain-aware attention; PAD tokens get -1 (no attention).
    sequence_id = (lm_input_ids == 0).cumsum(dim=1) - 1  # BOS=0
    sequence_id = sequence_id.masked_fill(lm_input_ids == 1, -1)  # PAD=1

    with torch.inference_mode():
        esmc_out = esmc(input_ids=lm_input_ids, sequence_id=sequence_id, output_hidden_states=True)

    # ESMC returns hidden states as the standard tuple of per-layer tensors; stack
    # them into the single [n_layers+1, B, max_len, D] tensor the projection expects.
    hs = torch.stack(esmc_out.hidden_states, dim=0)  # [n_layers+1, B, max_len, D]
    n_layers_plus_1, _, _, D = hs.shape
    result = torch.zeros(B, L, n_layers_plus_1, D, device=device, dtype=hs.dtype)
    for b in range(B):
        mb = protein_mask[b]
        em = expand_maps[b][mb]  # [n_protein_tokens] LM positions
        # hs[:, b, em, :] -> [n_layers+1, n_protein_tokens, D]
        gathered = hs[:, b, em, :].permute(1, 0, 2)
        result[b, mb.nonzero(as_tuple=True)[0]] = gathered

    return result.detach()


# ===========================================================================
# ESMFold2TriangleMultiplicativeUpdate
# ===========================================================================
@use_kernel_forward_from_hub("ESMFold2TriangleMultiplication")
class ESMFold2TriangleMultiplicativeBlock(nn.Module):
    """Triangle multiplicative update block with gated signal routing.

    The O(N^3) triangular contraction below is the trunk's dominant cost. Loading
    with ``ESMFold2Model.from_pretrained(..., device_map="cuda", use_kernels=True)``
    (CUDA + inference) swaps the whole block forward for a fused Triton kernel from
    the Hub (see the ``hub_kernels`` mapping); the pure-PyTorch ``forward`` here stays
    as the reference/fallback. The kernel reads this module's parameters
    (``norm_start``/``norm_mix``/``proj_bundle``/``proj_emit``/``proj_gate``) and
    matches ``forward``'s ``(pair_grid, visibility)`` signature, returning the
    residual-free delta.
    """

    _FLOW_TO_EINSUM = {"outgoing": "bikd,bjkd->bijd", "incoming": "bkid,bkjd->bijd"}
    _VALID_FLOWS = ("outgoing", "incoming")

    def __init__(self, input_channels: int, latent_channels: int, flow: str) -> None:
        super().__init__()
        if flow not in self._FLOW_TO_EINSUM:
            raise ValueError(f"Invalid flow={flow!r}. Expected one of {self._VALID_FLOWS}.")

        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.flow = flow
        self._einsum_equation = self._FLOW_TO_EINSUM[flow]
        self.norm_start = ESMFold2LayerNorm(self.input_channels, eps=_EPS)
        self.norm_mix = ESMFold2LayerNorm(self.latent_channels, eps=_EPS)
        self.proj_bundle = nn.Linear(self.input_channels, 4 * self.latent_channels, bias=False)
        self.proj_emit = nn.Linear(self.latent_channels, self.input_channels, bias=False)
        self.proj_gate = nn.Linear(self.input_channels, self.input_channels, bias=False)

        # Default chunked for memory on long sequences; tests override with
        # ``set_chunk_size(None)`` for the unchunked path under bit-exact bf16
        # parity checks.
        self._chunk_size: int | None = _DEFAULT_CHUNK_SIZE

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def _triangular_contract(self, left_stream: Tensor, right_stream: Tensor) -> Tensor:
        return torch.einsum(self._einsum_equation, left_stream, right_stream)

    def _triangular_contract_chunked(self, left_stream: Tensor, right_stream: Tensor, chunk_size: int) -> Tensor:
        """Compute the triangular einsum in chunks along the output i-dimension."""
        L = left_stream.shape[1] if self.flow == "outgoing" else left_stream.shape[2]
        chunks = []
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            if self.flow == "outgoing":
                chunk = torch.einsum(self._einsum_equation, left_stream[:, start:end], right_stream)
            else:
                chunk = torch.einsum(self._einsum_equation, left_stream[:, :, start:end], right_stream)
            chunks.append(chunk)
        return torch.cat(chunks, dim=1)

    def forward(self, pair_grid: Tensor, visibility: Tensor | None = None) -> Tensor:
        if visibility is None:
            visibility = pair_grid.new_ones(pair_grid.shape[:-1])

        normalized_grid = self.norm_start(pair_grid)
        bundled = self.proj_bundle(normalized_grid)
        signal, gate_logits = bundled.split(2 * self.latent_channels, dim=-1)
        # Gates and the O(N^3) contraction run in the activation dtype (bf16). This
        # matches the reference: under its autocast the einsum is downcast to bf16,
        # and the fused Triton kernel likewise contracts in bf16 — the dtype the
        # checkpoint was trained with. Keeping these in fp32 was a (marginal)
        # precision *up* that diverges from training and is slower on the trunk's
        # dominant op. ``norm_start``/``norm_mix`` stay fp32. A no-op in fp32.
        routed = signal * torch.sigmoid(gate_logits)
        routed = routed * visibility.unsqueeze(-1)

        left_stream, right_stream = routed.chunk(2, dim=-1)
        if self._chunk_size is not None:
            contracted = self._triangular_contract_chunked(left_stream, right_stream, self._chunk_size)
        else:
            contracted = self._triangular_contract(left_stream, right_stream)
        mixed = self.proj_emit(self.norm_mix(contracted.float()).to(self.proj_emit.weight.dtype))
        output_gate = torch.sigmoid(self.proj_gate(normalized_grid))
        return mixed * output_gate


class ESMFold2TriangleMultiplicativeUpdate(nn.Module):
    """Thin wrapper exposing the triangular mixer with explicit orientation (v3)."""

    def __init__(self, dim: int = 128, _outgoing: bool = True) -> None:
        super().__init__()
        flow = "outgoing" if _outgoing else "incoming"
        self._engine = ESMFold2TriangleMultiplicativeBlock(input_channels=dim, latent_channels=dim, flow=flow)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._engine.set_chunk_size(chunk_size)

    def forward(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        return self._engine(z, visibility=mask)


# ===========================================================================
# ESMFold2FoldingTrunk: ESMFold2Transition, ESMFold2PairUpdateBlock, ESMFold2FoldingTrunk
# ===========================================================================


class ESMFold2Transition(nn.Module):
    """LayerNorm + ESMFold2SwiGLU feed-forward residual block, chunked along the token axis."""

    def __init__(self, d_model: int, expansion_ratio: int = 4) -> None:
        super().__init__()
        self.norm = ESMFold2LayerNorm(d_model)
        self.ffn = ESMFold2SwiGLUMLP(d_model, expansion_ratio=expansion_ratio, bias=False)
        # Default chunked; set_chunk_size(None) disables for bit-exact parity tests.
        self._chunk_size: int | None = _DEFAULT_CHUNK_SIZE

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def forward(self, x: Tensor) -> Tensor:
        if self._chunk_size is None or x.shape[1] <= self._chunk_size:
            return x + self.ffn(self.norm(x))
        out_list: list[Tensor] = []
        for s in range(0, x.shape[1], self._chunk_size):
            e = min(s + self._chunk_size, x.shape[1])
            sl = x[:, s:e]
            out_list.append(sl + self.ffn(self.norm(sl)))
        return torch.cat(out_list, dim=1)


class ESMFold2PairUpdateBlock(nn.Module):
    """tri_mul_out, tri_mul_in, pair_transition."""

    def __init__(self, d_pair: int = 256, expansion_ratio: int = 4) -> None:
        super().__init__()
        self.tri_mul_out = ESMFold2TriangleMultiplicativeUpdate(dim=d_pair, _outgoing=True)
        self.tri_mul_in = ESMFold2TriangleMultiplicativeUpdate(dim=d_pair, _outgoing=False)
        self.pair_transition = ESMFold2Transition(d_pair, expansion_ratio=expansion_ratio)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.tri_mul_out.set_chunk_size(chunk_size)
        self.tri_mul_in.set_chunk_size(chunk_size)
        self.pair_transition.set_chunk_size(chunk_size)

    def forward(self, pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        # HF model is inference-only, so the trained row-shared dropout (r=0) is a no-op.
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
        pair = self.pair_transition(pair)
        return pair


class ESMFold2FoldingTrunk(nn.Module):
    """ModuleList of PairUpdateBlocks."""

    def __init__(self, n_layers: int = 24, d_pair: int = 256, expansion_ratio: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ESMFold2PairUpdateBlock(d_pair=d_pair, expansion_ratio=expansion_ratio) for _ in range(n_layers)]
        )

    def set_chunk_size(self, chunk_size: int | None) -> None:
        for block in self.blocks:
            block.set_chunk_size(chunk_size)

    def forward(self, pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        for block in self.blocks:
            fn = partial(block, pair_attention_mask=pair_attention_mask)
            if torch.is_grad_enabled():
                pair = checkpoint(fn, pair, use_reentrant=False)
            else:
                pair = fn(pair)
        return pair


# ===========================================================================
# MSA Encoder
# ===========================================================================


class ESMFold2OuterProductMean(nn.Module):
    """Outer-product mean: maps an MSA representation into a pair update.

    The order of the ``/ n_valid`` divide vs. the ``Wout`` projection is
    selectable via ``divide_outer_before_proj`` because different ESMFold2
    checkpoints were trained with different orderings:

    * ``False`` (default): ``Wout(outer) / n_valid`` — the projection bias
      is scaled by 1/n_valid alongside the outer product.
    * ``True``: ``Wout(outer / n_valid)`` — the projection bias is added
      unscaled, post-divide.
    """

    def __init__(
        self,
        d_msa: int,
        d_hidden: int,
        d_pair: int,
        divide_outer_before_proj: bool = False,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.divide_outer_before_proj = divide_outer_before_proj
        self.norm = ESMFold2LayerNorm(d_msa)
        self.W = nn.Linear(d_msa, 2 * d_hidden, bias=False)
        self.Wout = nn.Linear(d_hidden * d_hidden, d_pair, bias=True)
        # Off for bit-exact bf16; ``set_chunk_size(64)`` for long sequences.
        self._chunk_size: int | None = None

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def forward(self, m: Tensor, msa_attention_mask: Tensor) -> Tensor:
        m_norm = self.norm(m)
        x = self.W(m_norm) * msa_attention_mask.unsqueeze(-1).to(m_norm.dtype)
        a, b = x.chunk(2, dim=-1)
        mask_f = msa_attention_mask.to(a.dtype)
        n_valid = (mask_f @ mask_f.transpose(-1, -2)).unsqueeze(-1).clamp(min=1.0)
        if self._chunk_size is None:
            outer = torch.einsum("bimc,bjmd->bijcd", a, b).flatten(-2)
            if self.divide_outer_before_proj:
                return self.Wout(outer / n_valid)
            return self.Wout(outer) / n_valid
        # Chunk along the left (i) axis so the peak einsum intermediate is
        # [B, chunk, L, c, d] instead of [B, L, L, c, d].
        L = a.shape[1]
        out_chunks: list[Tensor] = []
        for s in range(0, L, self._chunk_size):
            e = min(s + self._chunk_size, L)
            outer_chunk = torch.einsum("bimc,bjmd->bijcd", a[:, s:e], b).flatten(-2)
            if self.divide_outer_before_proj:
                out_chunks.append(self.Wout(outer_chunk / n_valid[:, s:e]))
            else:
                out_chunks.append(self.Wout(outer_chunk) / n_valid[:, s:e])
        return torch.cat(out_chunks, dim=1)


class ESMFold2MSAPairWeightedAveraging(nn.Module):
    """Pair-biased MSA row update (AF3 Supplement Algorithm 10)."""

    def __init__(self, d_msa: int, d_pair: int, n_heads: int = 8, head_width: int = 32) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_width = head_width
        self.norm_single = ESMFold2LayerNorm(d_msa)
        self.compute_bias = nn.Sequential(ESMFold2LayerNorm(d_pair), nn.Linear(d_pair, n_heads, bias=False))
        self.Wv = nn.Linear(d_msa, n_heads * head_width, bias=False)
        self.Wgate = nn.Linear(d_msa, n_heads * head_width, bias=False)
        self.Wout = nn.Linear(n_heads * head_width, d_msa, bias=False)

    def forward(self, msa_repr: Tensor, pair_repr: Tensor, pair_attention_mask: Tensor) -> Tensor:
        """
        Args:
            msa_repr:           [B, L, M, d_msa]
            pair_repr:          [B, L, L, d_pair]
            pair_attention_mask:[B, L, L]
        Returns:
            [B, L, M, d_msa]
        """
        B, L, M, _ = msa_repr.shape
        h, dh = self.n_heads, self.head_width

        msa_normed = self.norm_single(msa_repr)
        bias = self.compute_bias[1](self.compute_bias[0](pair_repr))  # [B, L, L, n_heads]
        bias.masked_fill_(~pair_attention_mask.unsqueeze(-1).bool(), -1e5)
        attn = torch.softmax(bias, dim=-2, dtype=torch.float32).to(bias.dtype)  # softmax over j

        v = self.Wv(msa_normed).reshape(B, L, M, h, dh)
        gate = torch.sigmoid(self.Wgate(msa_normed)).reshape(B, L, M, h, dh)

        output = torch.einsum("bijh,bjmhd,bimhd->bimhd", attn, v, gate)
        return self.Wout(output.reshape(B, L, M, h * dh))


_CONFIDENCE_EPS = 1e-6
_NONPOLYMER_ID = 4


@dataclass
class ESMFold2Output(ModelOutput):
    """
    Output of [`ESMFold2Model`]. All confidence scores are on a 0-1 scale; per-sample tensors
    have a leading `num_diffusion_samples` axis.

    Args:
        distogram_logits (`torch.FloatTensor` of shape `(batch_size, num_tokens, num_tokens, distogram_bins)`):
            Predicted distance-distribution logits over residue pairs (RNG-independent; no diffusion sampling).
        sample_atom_coords (`torch.FloatTensor` of shape `(num_diffusion_samples, num_atoms, 3)`):
            Predicted all-atom Cartesian coordinates for each diffusion sample.
        plddt_logits (`torch.FloatTensor` of shape `(num_diffusion_samples, num_atoms, num_plddt_bins)`):
            Per-atom pLDDT bin logits.
        plddt (`torch.FloatTensor` of shape `(num_diffusion_samples, num_tokens)`):
            Per-residue predicted lDDT confidence.
        plddt_per_atom (`torch.FloatTensor` of shape `(num_diffusion_samples, num_atoms)`):
            Per-atom predicted lDDT confidence.
        plddt_ca (`torch.FloatTensor` of shape `(num_diffusion_samples, num_tokens)`):
            Predicted lDDT at the representative (Cα) atom of each token.
        complex_plddt (`torch.FloatTensor` of shape `(num_diffusion_samples,)`):
            Mean pLDDT over all atoms of the complex.
        complex_iplddt (`torch.FloatTensor` of shape `(num_diffusion_samples,)`):
            Interface-weighted complex pLDDT.
        pae_logits (`torch.FloatTensor` of shape `(num_diffusion_samples, num_tokens, num_tokens, num_pae_bins)`):
            Predicted-aligned-error bin logits.
        pae (`torch.FloatTensor` of shape `(num_diffusion_samples, num_tokens, num_tokens)`):
            Expected predicted aligned error (Å) for each residue pair.
        pde_logits (`torch.FloatTensor` of shape `(num_diffusion_samples, num_tokens, num_tokens, num_pde_bins)`):
            Predicted-distance-error bin logits.
        pde (`torch.FloatTensor` of shape `(num_diffusion_samples, num_tokens, num_tokens)`):
            Expected predicted distance error (Å) for each residue pair.
        resolved_logits (`torch.FloatTensor` of shape `(num_diffusion_samples, num_atoms, 2)`):
            Per-atom resolved/unresolved logits.
        ptm (`torch.FloatTensor` of shape `(num_diffusion_samples,)`):
            Predicted TM-score for each sample.
        iptm (`torch.FloatTensor` of shape `(num_diffusion_samples,)`):
            Predicted interface TM-score for each sample.
        pair_chains_iptm (`torch.FloatTensor` of shape `(num_diffusion_samples, num_chains, num_chains)`):
            Predicted interface TM-score for each ordered chain pair.
    """

    distogram_logits: Tensor | None = None
    sample_atom_coords: Tensor | None = None
    plddt_logits: Tensor | None = None
    plddt: Tensor | None = None
    plddt_per_atom: Tensor | None = None
    plddt_ca: Tensor | None = None
    complex_plddt: Tensor | None = None
    complex_iplddt: Tensor | None = None
    pae_logits: Tensor | None = None
    pae: Tensor | None = None
    pde_logits: Tensor | None = None
    pde: Tensor | None = None
    resolved_logits: Tensor | None = None
    ptm: Tensor | None = None
    iptm: Tensor | None = None
    pair_chains_iptm: Tensor | None = None


class ESMFold2ConfidenceHead(nn.Module):
    """Predicts pLDDT, PAE, PDE, resolved-atom probability and distogram bins."""

    boundaries: Tensor

    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__()
        ch = config.confidence_head
        d_single = config.d_single
        d_pair = config.d_pair
        d_inputs = config.inputs.d_inputs

        boundaries = torch.linspace(ch.min_dist, ch.max_dist, ch.distogram_bins - 1)
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(ch.distogram_bins, d_pair)

        self.s_norm = ESMFold2LayerNorm(d_single)
        self.s_inputs_to_single = nn.Linear(d_inputs, d_single, bias=False)
        self.s_to_z = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_transpose = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_in1 = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_in2 = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_out = nn.Linear(d_pair, d_pair, bias=False)
        self.s_input_to_s = nn.Linear(d_inputs, d_single, bias=False)
        self.s_inputs_norm = ESMFold2LayerNorm(d_inputs)
        self.z_norm = ESMFold2LayerNorm(d_pair)

        self.row_attention_pooling = ESMFold2RowAttentionPooling(d_pair=d_pair, d_single=d_single)

        pf = ch.folding_trunk
        self.folding_trunk = ESMFold2FoldingTrunk(n_layers=pf.n_layers, d_pair=d_pair, expansion_ratio=4)

        # Heads.
        self.plddt_ln = ESMFold2LayerNorm(d_single)
        max_atoms_per_token = 23
        self.plddt_weight = nn.Parameter(torch.zeros(max_atoms_per_token, d_single, ch.num_plddt_bins))

        self.pae_ln = ESMFold2LayerNorm(d_pair)
        self.pae_head = nn.Linear(d_pair, ch.num_pae_bins, bias=False)

        self.pde_ln = ESMFold2LayerNorm(d_pair)
        self.pde_head = nn.Linear(d_pair, ch.num_pde_bins, bias=False)

        self.resolved_ln = ESMFold2LayerNorm(d_single)
        # 2 = resolved logits ([unresolved, resolved]).
        self.resolved_weight = nn.Parameter(torch.zeros(max_atoms_per_token, d_single, 2))

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)

    @staticmethod
    def _repeat_batch(x: Tensor, num_diffusion_samples: int) -> Tensor:
        return x if num_diffusion_samples == 1 else x.repeat_interleave(num_diffusion_samples, 0)

    @staticmethod
    def _flatten_sample_axis(x: Tensor) -> Tensor:
        if x.ndim == 4:
            b, mult, n, c = x.shape
            return x.reshape(b * mult, n, c)
        return x

    def forward(
        self,
        s_inputs: Tensor,
        z: Tensor,
        x_pred: Tensor,
        distogram_atom_idx: Tensor,
        token_attention_mask: Tensor,
        atom_to_token: Tensor,
        atom_attention_mask: Tensor,
        asym_id: Tensor,
        mol_type: Tensor,
        num_diffusion_samples: int = 1,
        relative_position_encoding: Tensor | None = None,
        token_bonds_encoding: Tensor | None = None,
    ) -> dict[str, Tensor]:
        s_inputs_normed = self.s_inputs_norm(s_inputs)

        z_base = self.z_norm(z)
        if relative_position_encoding is not None:
            z_base = z_base + relative_position_encoding
        if token_bonds_encoding is not None:
            z_base = z_base + token_bonds_encoding
        z_base = z_base + self.s_to_z(s_inputs_normed).unsqueeze(2)
        z_base = z_base + self.s_to_z_transpose(s_inputs_normed).unsqueeze(1)
        z_base = z_base + self.s_to_z_prod_out(
            self.s_to_z_prod_in1(s_inputs_normed)[:, :, None, :] * self.s_to_z_prod_in2(s_inputs_normed)[:, None, :, :]
        )

        pair = self._repeat_batch(z_base, num_diffusion_samples)
        x_pred_flat = self._flatten_sample_axis(x_pred)
        atom_to_token_m = self._repeat_batch(atom_to_token, num_diffusion_samples)
        atom_mask_m = self._repeat_batch(atom_attention_mask, num_diffusion_samples)
        rep_idx_m = self._repeat_batch(distogram_atom_idx, num_diffusion_samples).long()
        mask = self._repeat_batch(token_attention_mask, num_diffusion_samples)
        Bm = pair.shape[0]

        rep_coords = gather_rep_atom_coords(x_pred_flat, rep_idx_m)
        rep_distances = torch.cdist(rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist")
        distogram_bins = (rep_distances.unsqueeze(-1) > self.boundaries).sum(dim=-1).long()
        pair = pair + self.dist_bin_pairwise_embed(distogram_bins)

        pair_mask = mask[:, :, None].float() * mask[:, None, :].float()

        # `pair` is fp32 here (built from the fp32 trunk output `z`); run the
        # folding trunk in the model's compute dtype, then accumulate in fp32.
        pair_delta = self.folding_trunk(pair.to(self.pae_head.weight.dtype), pair_attention_mask=pair_mask)
        pair.add_(pair_delta.float())
        del pair_delta
        # Accumulated in fp32; hand the downstream confidence heads the compute dtype.
        pair = pair.to(self.pae_head.weight.dtype)
        single = self.row_attention_pooling(pair, mask)

        atom_mask_f = atom_mask_m.float()
        s_at_atoms = gather_token_to_atom(single, atom_to_token_m)
        s_at_atoms_ln = self.plddt_ln(s_at_atoms)

        intra_idx = _compute_intra_token_idx(atom_to_token_m)
        intra_idx = intra_idx.clamp(max=self.plddt_weight.shape[0] - 1)
        w_plddt = self.plddt_weight[intra_idx]
        plddt_logits = torch.einsum("...c,...cb->...b", s_at_atoms_ln, w_plddt)
        plddt_per_atom = _categorical_mean(plddt_logits, start=0.0, end=1.0)

        L = single.shape[1]
        plddt_sum = torch.zeros(Bm, L, device=single.device, dtype=plddt_per_atom.dtype)
        atom_count = torch.zeros(Bm, L, device=single.device, dtype=plddt_per_atom.dtype)
        atom_mask_t = atom_mask_f.to(plddt_per_atom.dtype)
        plddt_sum.scatter_add_(1, atom_to_token_m, plddt_per_atom * atom_mask_t)
        atom_count.scatter_add_(1, atom_to_token_m, atom_mask_t)
        plddt = plddt_sum / atom_count.clamp(min=1e-6)

        complex_plddt = (plddt_per_atom * atom_mask_f).sum(dim=-1) / (atom_mask_f.sum(dim=-1) + _CONFIDENCE_EPS)

        expanded_type = self._repeat_batch(mol_type, num_diffusion_samples)
        expanded_asym = self._repeat_batch(asym_id, num_diffusion_samples)
        is_ligand = (expanded_type == _NONPOLYMER_ID).float()
        inter_chain = (expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)).float()
        near_contact = (rep_distances < 8).float()
        interface_per_token = (near_contact * inter_chain * (1.0 - is_ligand).unsqueeze(-1)).amax(dim=-1)
        iplddt_weight = torch.where(
            is_ligand.bool(),
            torch.full_like(interface_per_token, 2.0),
            interface_per_token,
        )
        iplddt_weight_atoms = gather_token_to_atom(iplddt_weight.unsqueeze(-1), atom_to_token_m).squeeze(-1)
        atom_iplddt_w = atom_mask_f * iplddt_weight_atoms
        complex_iplddt = (plddt_per_atom * atom_iplddt_w).sum(dim=-1) / (atom_iplddt_w.sum(dim=-1) + _CONFIDENCE_EPS)

        plddt_ca = plddt_per_atom.gather(1, rep_idx_m)

        # PAE
        pae_logits = self.pae_head(self.pae_ln(pair))
        pae = _categorical_mean(pae_logits, start=0.0, end=32.0).detach()

        # PDE
        pde_logits = self.pde_head(self.pde_ln(pair))
        pde = _categorical_mean(pde_logits, start=0.0, end=32.0).detach()

        # Resolved (per-atom binary).
        s_at_atoms_res = self.resolved_ln(s_at_atoms)
        w_res = self.resolved_weight[intra_idx]
        resolved_logits = torch.einsum("...c,...cb->...b", s_at_atoms_res, w_res)

        # pTM / ipTM from pae_logits.
        n_bins = pae_logits.shape[-1]
        bin_width = 32.0 / n_bins
        bin_centers = torch.arange(0.5 * bin_width, 32.0, bin_width, device=pae_logits.device)
        mask_f = mask.float()
        N_res = mask_f.sum(dim=-1, keepdim=True)
        d0 = 1.24 * (N_res.clamp(min=19) - 15) ** (1 / 3) - 1.8
        tm_per_bin = 1 / (1 + (bin_centers / d0) ** 2)
        pae_probs = F.softmax(pae_logits, dim=-1, dtype=torch.float32)
        tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(dim=-1)

        pair_mask_2d = mask_f.unsqueeze(-1) * mask_f.unsqueeze(-2)
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (pair_mask_2d.sum(dim=-1) + _CONFIDENCE_EPS)
        ptm = ptm_per_row.max(dim=-1).values

        inter_chain_mask = (expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)).float() * pair_mask_2d
        iptm_per_row = (tm_expected * inter_chain_mask).sum(dim=-1) / (inter_chain_mask.sum(dim=-1) + _CONFIDENCE_EPS)
        iptm = iptm_per_row.max(dim=-1).values

        max_chain_id = int(expanded_asym.max().item()) if Bm > 0 else 0
        n_chains = max_chain_id + 1
        pair_chains_iptm = torch.zeros(Bm, n_chains, n_chains, device=tm_expected.device, dtype=tm_expected.dtype)
        for c1 in range(n_chains):
            chain_c1 = (expanded_asym == c1).float() * mask_f
            if chain_c1.sum() == 0:
                continue
            for c2 in range(n_chains):
                chain_c2 = (expanded_asym == c2).float() * mask_f
                pair_m = chain_c1.unsqueeze(-1) * chain_c2.unsqueeze(-2)
                denom = pair_m.sum(dim=(-1, -2)) + _CONFIDENCE_EPS
                pair_chains_iptm[:, c1, c2] = (tm_expected * pair_m).sum(dim=(-1, -2)) / denom

        return {
            "plddt_logits": plddt_logits,
            "plddt": plddt.detach(),
            "plddt_per_atom": plddt_per_atom.detach(),
            "plddt_ca": plddt_ca.detach(),
            "complex_plddt": complex_plddt.detach(),
            "complex_iplddt": complex_iplddt.detach(),
            "pae_logits": pae_logits,
            "pae": pae,
            "pde_logits": pde_logits,
            "pde": pde,
            "resolved_logits": resolved_logits,
            "ptm": ptm.detach(),
            "iptm": iptm.detach(),
            "pair_chains_iptm": pair_chains_iptm.detach(),
        }


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


class ESMFold2PreTrainedModel(PreTrainedModel):
    """Base class for ESMFold2 — declares the loading and weight-initialization behaviour."""

    config_class = ESMFold2Config
    base_model_prefix = "esmfold2"
    main_input_name = "token_index"
    _no_split_modules = [
        "ESMCLayer",
        "ESMFold2PairUpdateBlock",
        "ESMFold2AtomEncoder",
        "ESMFold2AtomDecoder",
        "ESMFold2DiffusionTransformer",
    ]
    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$"]
    # The Fourier noise-embedding frequencies/phases are random Gaussian features whose
    # precision drives the diffusion conditioning; keep them fp32 even under dtype=bf16.
    _keep_in_fp32_modules_strict = ["fourier"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        # The base initializer handles Linear/Embedding/LayerNorm; below we (re)apply the
        # few non-default inits the architecture needs (adaLN-Zero gates, identity/recurrent
        # parcae params, zeroed projections). These live here, not in submodule __init__s,
        # so they survive `post_init()` and are not wastefully run before weight loading.
        # The `init.*` helpers are load-flag aware (they no-op on already-loaded weights).
        super()._init_weights(module)
        if isinstance(module, ESMFold2Model):
            init.eye_(module.parcae_readout.weight)
            init.eye_(module.parcae_b_cont)
            init.zeros_(module.parcae_log_a)
            parcae_delta_init = -math.log(math.sqrt(1.0 / 5.0))
            init.constant_(module.parcae_log_delta, _inverse_softplus(parcae_delta_init))
        elif isinstance(module, ESMFold2ConfidenceHead):
            init.zeros_(module.plddt_weight)
            init.zeros_(module.resolved_weight)
        elif isinstance(module, ESMFold2AdaptiveLayerNorm):
            init.ones_(module.s_scale)
        elif isinstance(module, ESMFold2SWAAtomBlock):
            init.zeros_(module.adaln_modulation[1].weight)
        elif isinstance(module, ESMFold2AttentionPairBias):
            if getattr(module, "out_gate", None) is not None:
                init.zeros_(module.out_gate.weight)
                init.constant_(module.out_gate.bias, -2.0)
        elif isinstance(module, ESMFold2ConditionedTransitionBlock):
            if getattr(module, "output_gate", None) is not None:
                init.zeros_(module.output_gate.weight)
                init.constant_(module.output_gate.bias, -2.0)
        elif isinstance(module, ESMFold2DiffusionModule):
            init.zeros_(module.s_to_token.weight)
        elif isinstance(module, ESMFold2LanguageModelShim):
            init.zeros_(module.base_z_combine)


class ESMFold2Model(ESMFold2PreTrainedModel):
    """ESMFold2 — all-atom structure prediction with an ESMC PLM backbone.

    This is the standard released ESMFold2 architecture (uses a linear-
    recurrent trunk, internally referred to as "parcae").

    Forward kwargs that callers commonly override:

    * ``num_loops`` (default ``config.num_loops``): trunk refinement
      loops.
    * ``num_diffusion_samples`` (default ``config.num_diffusion_samples``):
      parallel structure samples; the confidence head re-runs once per
      sample, so memory scales linearly. Pass ``1`` for cheap inference.
    * ``num_sampling_steps`` (default ``config.structure_head.inference_num_steps``):
      diffusion ODE solver steps. Lower for speed, higher for quality.

    Memory / perf knobs:

    * ``model.set_chunk_size(int|None)``: caps L² ops (triangle / OPM /
      pair transition) at this token-axis chunk. Default 64 — fits
      L≈2k on an 80 GB GPU. Pass ``None`` for faster inference at L<600.
    """

    def __init__(self, config: ESMFold2Config) -> None:
        super().__init__(config)
        d_inputs = config.inputs.d_inputs
        d_pair = config.d_pair

        self.inputs_embedder = ESMFold2InputsEmbedder(config)
        self.z_init_1 = nn.Linear(d_inputs, d_pair, bias=False)
        self.z_init_2 = nn.Linear(d_inputs, d_pair, bias=False)
        self.rel_pos = ESMFold2ResIdxAsymIdSymIdEntityIdEncoding(
            n_relative_residx_bins=config.n_relative_residx_bins,
            n_relative_chain_bins=config.n_relative_chain_bins,
            d_pair=d_pair,
        )
        self.token_bonds = nn.Linear(1, d_pair, bias=False)
        self.language_model = ESMFold2LanguageModelShim(
            d_z=d_pair, d_model=config.esmc_config.d_model, num_layers=config.esmc_config.n_layers
        )
        # The ESMC backbone is a bundled submodule: built here with random weights
        # (instant, no I/O), then populated by the parent's from_pretrained like any
        # other composite sub-model (its weights live under ``esmc.*`` in the ESMFold2
        # checkpoint). It is frozen in effect — ``forward`` detaches its hidden states
        # before they enter the trunk, so no gradient ever reaches it — and the bf16
        # autocast in ``_compute_lm_hidden_states`` reproduces the LM precision regime.
        self.esmc = AutoModel.from_config(config.esmc_config)

        pf = config.folding_trunk
        self.folding_trunk = ESMFold2FoldingTrunk(n_layers=pf.n_layers, d_pair=d_pair, expansion_ratio=4)
        if config.lm_encoder.enabled:
            self.lm_encoder: ESMFold2FoldingTrunk | None = ESMFold2FoldingTrunk(
                n_layers=config.lm_encoder.n_layers, d_pair=d_pair, expansion_ratio=4
            )
        else:
            self.lm_encoder = None

        # parcae linear-recurrence params (allocated here; initialized in _init_weights):
        # log_a -> 0, log_delta -> a fixed decay constant, b_cont -> identity, readout -> identity.
        self.parcae_input_norm = ESMFold2LayerNorm(d_pair)
        self.parcae_log_a = nn.Parameter(torch.zeros(d_pair))
        self.parcae_log_delta = nn.Parameter(torch.empty(d_pair, dtype=torch.float32))
        self.parcae_b_cont = nn.Parameter(torch.empty(d_pair, d_pair))
        self.parcae_readout = nn.Linear(d_pair, d_pair, bias=False)
        self.parcae_coda = ESMFold2FoldingTrunk(n_layers=config.parcae.coda_n_layers, d_pair=d_pair, expansion_ratio=4)

        # Heads --------------------------------------------------------------
        self.structure_head = ESMFold2DiffusionStructureHead(config)
        self.distogram_head = nn.Linear(d_pair, config.structure_head.distogram_bins, bias=True)
        self.confidence_head = ESMFold2ConfidenceHead(config)

        msa_cfg = config.msa_encoder
        self.msa_encoder = None
        if msa_cfg.enabled:
            self.msa_encoder = ESMFold2MSAEncoder(
                d_msa=msa_cfg.d_msa,
                d_pair=d_pair,
                d_inputs=d_inputs,
                d_hidden=msa_cfg.d_hidden,
                n_layers=msa_cfg.n_layers,
                n_heads_msa=msa_cfg.n_heads_msa,
                msa_head_width=msa_cfg.msa_head_width,
            )

        # ESMFold2SWA3DRoPEAttention modules live deep in the atom encoders/decoders and
        # are built from explicit dims, so give each a handle to the model config:
        # their forward dispatches the plain-attention core through the v5
        # attention interface (config._attn_implementation), staying live under
        # set_attn_implementation() since the config object is shared.
        for module in self.modules():
            if isinstance(module, ESMFold2SWA3DRoPEAttention):
                module.config = self.config

        self.post_init()

    def apply_torch_compile(self, mode: str = "fixed_seqlen", dynamic: bool | None = None) -> None:
        """Compile L²-heavy blocks. ``mode='fixed_seqlen'`` recompiles per L; ``'dynamic_seqlen'`` compiles once."""
        import torch._dynamo

        torch._dynamo.config.cache_size_limit = 512
        torch._dynamo.config.accumulated_cache_size_limit = 512
        # capture_scalar_outputs avoids graph breaks at .item() in atom-attention path.
        torch._dynamo.config.capture_scalar_outputs = True

        if dynamic is None:
            dynamic = mode == "dynamic_seqlen"
        kwargs: dict = {"dynamic": dynamic}

        compile_targets = (
            ESMFold2PairUpdateBlock,
            ESMFold2DiffusionTransformer,
            ESMFold2DiffusionModule,
            ESMFold2MSAEncoderBlock,
        )

        def _maybe_compile(module: nn.Module) -> None:
            if isinstance(module, compile_targets):
                module.forward = torch.compile(module.forward, **kwargs)

        self.apply(_maybe_compile)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)
        if self.lm_encoder is not None:
            self.lm_encoder.set_chunk_size(chunk_size)
        self.parcae_coda.set_chunk_size(chunk_size)
        self.confidence_head.set_chunk_size(chunk_size)
        if self.msa_encoder is not None:
            self.msa_encoder.set_chunk_size(chunk_size)

    def _compute_lm_hidden_states(
        self,
        input_ids: Tensor,
        asym_id: Tensor,
        residue_index: Tensor,
        mol_type: Tensor,
        tok_mask: Tensor,
    ) -> Tensor:
        # Run the frozen ESMC backbone exactly as the reference does: plain weights
        # under a bf16 autocast (the fork's ``_lm_precision_context``). For a bf16
        # backbone this puts norms/softmax in fp32 and matmuls/rotary in bf16,
        # reproducing the checkpoint's training regime bit-for-bit; disabled (a
        # no-op) for an fp32 backbone. This autocast is scoped to the backbone only
        # — the trunk stays dtype-honest.
        use_amp = next(self.esmc.parameters()).dtype == torch.bfloat16
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
            return compute_lm_hidden_states(
                self.esmc,
                input_ids,
                asym_id,
                residue_index,
                mol_type,
                tok_mask,
                pad_to_multiple=None,
            )

    def _discretized_dynamics(self) -> tuple[Tensor, Tensor]:
        delta = F.softplus(self.parcae_log_delta)
        a = torch.exp(-delta * torch.exp(self.parcae_log_a))
        b = delta[:, None] * self.parcae_b_cont
        return a, b

    def _init_pair_state(self, ref: Tensor) -> Tensor:
        std = math.sqrt(2.0 / (5.0 * ref.shape[-1]))
        state = torch.empty_like(ref, dtype=torch.float32)
        nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return state.to(dtype=ref.dtype)

    def _run_one_loop(
        self,
        z: Tensor,
        z_init: Tensor,
        lm_z: Tensor | None,
        _msa_kwargs: dict | None,
        pair_mask: Tensor,
        a: Tensor,
        b_mat: Tensor,
        total_steps: int,
    ) -> Tensor:
        # Helper method (not inline) so per-iter locals free on return —
        # otherwise leaks ~2 GB L²×c_z into distogram/sample scope.
        # training=True forces dropout under eval(), matching the per-loop
        # dropout strategy used at train time.
        lm_cfg = self.config.lm_encoder
        _per_loop_lm_dropout = (
            lm_z is not None
            and getattr(lm_cfg, "per_loop_lm_dropout", False)
            and getattr(lm_cfg, "lm_dropout", 0.0) > 0.0
        )
        _lm_dropout_p = getattr(lm_cfg, "lm_dropout", 0.0)

        for _ in range(total_steps):
            if _per_loop_lm_dropout:
                assert lm_z is not None  # narrowed by _per_loop_lm_dropout
                lm_z_i: Tensor | None = F.dropout(lm_z, p=_lm_dropout_p, training=True)
            else:
                lm_z_i = lm_z

            refined_lm_z: Tensor | None = None
            if lm_z_i is not None and self.lm_encoder is not None:
                refined_lm_z = self.lm_encoder(lm_z_i.to(z_init.dtype), pair_attention_mask=pair_mask)

            z_inject_pair = z_init
            if lm_z_i is not None and self.lm_encoder is None:
                z_inject_pair = z_inject_pair + lm_z_i.to(z_inject_pair.dtype)

            if self.msa_encoder is not None and _msa_kwargs is not None:
                msa_pair = self.msa_encoder(x_pair=z_inject_pair, **_msa_kwargs).to(z_inject_pair.dtype)
                z_inject_pair = msa_pair if self.config.msa_encoder_overwrite else (z_inject_pair + msa_pair)

            if refined_lm_z is not None:
                z_inject_pair = z_inject_pair + refined_lm_z.to(z_inject_pair.dtype)

            injected_pair = self.parcae_input_norm(z_inject_pair)
            z = a * z + F.linear(injected_pair.to(z.dtype), b_mat)
            z = self.folding_trunk(z, pair_attention_mask=pair_mask)

        return z

    def forward(
        self,
        token_index: Tensor,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        mol_type: Tensor,
        res_type: Tensor,
        token_bonds: Tensor,
        token_attention_mask: Tensor,
        ref_pos: Tensor,
        ref_element: Tensor,
        ref_charge: Tensor,
        ref_atom_name_chars: Tensor,
        ref_space_uid: Tensor,
        atom_attention_mask: Tensor,
        atom_to_token: Tensor,
        distogram_atom_idx: Tensor,
        deletion_mean: Tensor | None = None,
        msa: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        lm_hidden_states: Tensor | None = None,
        num_loops: int | None = None,
        num_diffusion_samples: int | None = None,
        num_sampling_steps: int | None = None,
        **kwargs,
    ) -> ESMFold2Output:
        tok_mask = token_attention_mask
        atm_mask = atom_attention_mask
        disto_idx = distogram_atom_idx

        n_loops: int = num_loops if num_loops is not None else self.config.num_loops
        n_samples: int = (
            num_diffusion_samples if num_diffusion_samples is not None else self.config.num_diffusion_samples
        )
        total_steps = max(1, n_loops + 1)

        if res_type.dim() == 2:
            res_type_oh = F.one_hot(res_type.long(), num_classes=NUM_RES_TYPES).float()
            res_type_oh = res_type_oh * tok_mask.unsqueeze(-1).float()
        else:
            res_type_oh = res_type.float()

        if msa is not None:
            msa_oh_profile = F.one_hot(msa.long(), num_classes=NUM_RES_TYPES).float()
            if msa_attention_mask is not None:
                mask_f = msa_attention_mask.float().unsqueeze(-1)
                msa_oh_profile = msa_oh_profile * mask_f
                valid_seq_count = msa_attention_mask.float().sum(dim=1).clamp(min=1)
                profile = msa_oh_profile.sum(dim=1) / valid_seq_count.unsqueeze(-1)
            else:
                profile = msa_oh_profile.mean(dim=1)
        else:
            profile = res_type_oh

        if deletion_mean is None:
            deletion_mean = torch.zeros(res_type.shape[0], res_type.shape[1], device=res_type.device)

        ref_element_oh = F.one_hot(ref_element.long(), num_classes=MAX_ATOMIC_NUMBER).float()
        ref_atom_name_chars_oh = F.one_hot(ref_atom_name_chars.long(), num_classes=CHAR_VOCAB_SIZE).float()
        # Bias-free downstream Linears require zeroed padding.
        atm_mask_f = atm_mask.float()
        ref_element_oh = ref_element_oh * atm_mask_f.unsqueeze(-1)
        ref_atom_name_chars_oh = ref_atom_name_chars_oh * atm_mask_f.unsqueeze(-1).unsqueeze(-1)
        atom_to_token = atom_to_token * atm_mask.long()

        x_inputs = self.inputs_embedder(
            aatype=res_type_oh,
            profile=profile.float(),
            deletion_mean=deletion_mean.float(),
            ref_pos=ref_pos,
            atom_attention_mask=atm_mask,
            ref_space_uid=ref_space_uid,
            ref_charge=ref_charge,
            ref_element=ref_element_oh,
            ref_atom_name_chars=ref_atom_name_chars_oh,
            atom_to_token=atom_to_token,
        )

        z_init = self.z_init_1(x_inputs).unsqueeze(2) + self.z_init_2(x_inputs).unsqueeze(1)

        relative_position_encoding = self.rel_pos(
            residue_index=residue_index,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            token_index=token_index,
        )
        token_bonds_encoding = self.token_bonds(token_bonds.to(self.token_bonds.weight.dtype))
        z_init = z_init + relative_position_encoding + token_bonds_encoding

        if lm_hidden_states is None and input_ids is not None:
            lm_hidden_states = self._compute_lm_hidden_states(input_ids, asym_id, residue_index, mol_type, tok_mask)
        lm_z: Tensor | None = None
        if lm_hidden_states is not None:
            lm_z = self.language_model(lm_hidden_states.detach())
        del lm_hidden_states

        pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()

        z = self._init_pair_state(z_init)

        a, b = self._discretized_dynamics()
        a = a.view(1, 1, 1, -1).to(device=z.device, dtype=z.dtype)
        b_mat = b.to(device=z.device, dtype=z.dtype)

        _msa_kwargs: dict | None = None
        if self.msa_encoder is not None and msa is not None:
            B_msa, M, L_msa = msa.shape
            msa_oh = F.one_hot(msa.permute(0, 2, 1).long(), num_classes=NUM_RES_TYPES).float()
            msa_attn = (
                msa_attention_mask.permute(0, 2, 1).float()
                if msa_attention_mask is not None
                else tok_mask[:, :, None].expand(-1, -1, M).float()
            )
            # Bias-free ESMFold2MSAEncoder.embed requires zeroed padding.
            msa_oh = msa_oh * msa_attn.unsqueeze(-1)
            hd = (
                has_deletion.permute(0, 2, 1).float()
                if has_deletion is not None
                else torch.zeros(B_msa, L_msa, M, device=msa.device)
            )
            dv = (
                deletion_value.permute(0, 2, 1).float()
                if deletion_value is not None
                else torch.zeros(B_msa, L_msa, M, device=msa.device)
            )
            _msa_kwargs = {
                "x_inputs": x_inputs,
                "msa_oh": msa_oh,
                "has_deletion": hd,
                "deletion_value": dv,
                "msa_attention_mask": msa_attn,
            }

        # Method call (not inline loop) frees per-iter L²×c_z locals.
        z = self._run_one_loop(
            z=z,
            z_init=z_init,
            lm_z=lm_z,
            _msa_kwargs=_msa_kwargs,
            pair_mask=pair_mask,
            a=a,
            b_mat=b_mat,
            total_steps=total_steps,
        )
        del z_init, lm_z, _msa_kwargs, a, b_mat

        z = self.parcae_readout(z)
        z = self.parcae_coda(z, pair_attention_mask=pair_mask)

        z = z.float()
        distogram_logits = self.distogram_head((z + z.transpose(-2, -3)).to(self.distogram_head.weight.dtype))

        structure_output = self.structure_head.sample(
            z_trunk=z,
            s_inputs=x_inputs,
            relative_position_encoding=relative_position_encoding,
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            ref_mask=atm_mask,
            ref_element=ref_element_oh,
            ref_atom_name_chars=ref_atom_name_chars_oh,
            ref_space_uid=ref_space_uid,
            tok_idx=atom_to_token,
            asym_id=asym_id,
            residue_index=residue_index,
            entity_id=entity_id,
            token_index=token_index,
            sym_id=sym_id,
            token_attention_mask=tok_mask,
            num_diffusion_samples=n_samples,
            num_sampling_steps=num_sampling_steps,
            return_atom_repr=False,
            denoising_early_exit_rmsd=None,
        )

        sample_coords = structure_output["sample_atom_coords"]
        if sample_coords is None:
            raise RuntimeError("The diffusion structure head did not return sampled coordinates.")

        confidence_output = self.confidence_head(
            s_inputs=x_inputs.detach(),
            z=z.detach().float(),
            x_pred=sample_coords.detach(),
            distogram_atom_idx=disto_idx,
            token_attention_mask=tok_mask,
            atom_to_token=atom_to_token,
            atom_attention_mask=atm_mask,
            asym_id=asym_id,
            mol_type=mol_type,
            num_diffusion_samples=n_samples,
            relative_position_encoding=relative_position_encoding.detach(),
            token_bonds_encoding=token_bonds_encoding.detach(),
        )

        return ESMFold2Output(
            distogram_logits=distogram_logits,
            sample_atom_coords=sample_coords,
            **confidence_output,
        )

    @torch.no_grad()
    def infer_protein(self, seq: str, **forward_kwargs) -> ESMFold2Output:
        from .protein_utils import prepare_protein_features

        features = prepare_protein_features(seq)
        features = {k: v.to(self.device) for k, v in features.items()}
        return self(**features, **forward_kwargs)

    @torch.no_grad()
    def infer_protein_as_pdb(self, seq: str, **forward_kwargs) -> str:
        from .protein_utils import output_to_pdb, prepare_protein_features

        features = prepare_protein_features(seq)
        features = {k: v.to(self.device) for k, v in features.items()}
        output = self(**features, **forward_kwargs)
        return output_to_pdb(output, features)

    @staticmethod
    def output_to_pdb(output: ESMFold2Output, features: dict[str, Tensor]) -> str:
        """Render a PDB string from an [`ESMFold2Output`] and the input ``features`` it was produced from."""
        from .protein_utils import output_to_pdb as _output_to_pdb

        return _output_to_pdb(output, features)


class ESMFold2MSAEncoderBlock(nn.Module):
    """One MSA encoder block: OPM into pair, MSA pair-weighted averaging, triangle update."""

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_hidden: int,
        n_heads_msa: int,
        msa_head_width: int,
        is_final_block: bool = False,
    ) -> None:
        super().__init__()
        self.is_final_block = is_final_block
        self.outer_product_mean = ESMFold2OuterProductMean(d_msa, d_hidden, d_pair)
        if not is_final_block:
            self.msa_pair_weighted_averaging = ESMFold2MSAPairWeightedAveraging(
                d_msa, d_pair, n_heads_msa, msa_head_width
            )
            self.msa_transition = ESMFold2Transition(d_msa, expansion_ratio=4)
        self.tri_mul_out = ESMFold2TriangleMultiplicativeUpdate(dim=d_pair, _outgoing=True)
        self.tri_mul_in = ESMFold2TriangleMultiplicativeUpdate(dim=d_pair, _outgoing=False)
        self.pair_transition = ESMFold2Transition(d_pair, expansion_ratio=4)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.outer_product_mean.set_chunk_size(chunk_size)
        self.tri_mul_out.set_chunk_size(chunk_size)
        self.tri_mul_in.set_chunk_size(chunk_size)
        if not self.is_final_block:
            self.msa_transition.set_chunk_size(chunk_size)
        self.pair_transition.set_chunk_size(chunk_size)

    def forward(
        self,
        m: Tensor,
        pair: Tensor,
        msa_attention_mask: Tensor,
        pair_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pair = pair + self.outer_product_mean(m, msa_attention_mask)
        if not self.is_final_block:
            m = m + self.msa_pair_weighted_averaging(m, pair, pair_attention_mask)
            m = self.msa_transition(m)
        pair = pair + self.tri_mul_out(pair, mask=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, mask=pair_attention_mask)
        pair = self.pair_transition(pair)
        return m, pair


class ESMFold2MSAEncoder(nn.Module):
    """Stack of [`ESMFold2MSAEncoderBlock`] layers that conditions the pair on an MSA."""

    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        d_inputs: int,
        d_hidden: int = 32,
        n_layers: int = 4,
        n_heads_msa: int = 8,
        msa_head_width: int = 16,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(35, d_msa, bias=False)
        self.project_inputs = nn.Linear(d_inputs, d_msa, bias=False)
        self.blocks = nn.ModuleList(
            [
                ESMFold2MSAEncoderBlock(
                    d_msa=d_msa,
                    d_pair=d_pair,
                    d_hidden=d_hidden,
                    n_heads_msa=n_heads_msa,
                    msa_head_width=msa_head_width,
                    is_final_block=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )

    def set_chunk_size(self, chunk_size: int | None) -> None:
        for block in self.blocks:
            block.set_chunk_size(chunk_size)

    def forward(
        self,
        x_pair: Tensor,
        x_inputs: Tensor,
        msa_oh: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        msa_attention_mask: Tensor,
    ) -> Tensor:
        # All inputs are pre-transposed to [B, L, M, ...] before calling.
        m_feat = torch.cat([msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1)
        m = self.embed(m_feat.to(self.embed.weight.dtype)) + self.project_inputs(x_inputs).unsqueeze(2)
        tok_mask = msa_attention_mask[:, :, 0].bool()
        pair_attention_mask = tok_mask.unsqueeze(2) & tok_mask.unsqueeze(1)
        for block in self.blocks:
            m, x_pair = block(m, x_pair, msa_attention_mask, pair_attention_mask)
        return x_pair


__all__ = ["ESMFold2Model", "ESMFold2PreTrainedModel"]
