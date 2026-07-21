# Copyright 2026 BioHub and The HuggingFace Inc. team. All rights reserved.
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

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ... import initialization as init
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_bidirectional_mask
from ...modeling_outputs import ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..auto import AutoModel
from .configuration_esmfold2 import EsmFold2Config


@dataclass
class EsmFold2AtomInputs:
    """Reference-conformer atom features, threaded together through the inputs embedder,
    atom encoder/decoder, and diffusion module."""

    ref_pos: Tensor
    ref_charge: Tensor
    atom_attention_mask: Tensor
    ref_element: Tensor
    ref_atom_name_chars: Tensor
    ref_space_uid: Tensor
    atom_to_token: Tensor


@dataclass
class EsmFold2AtomAttention:
    """Per-fold, layer-invariant inputs for the SWA atom-stack attention: the 3D-RoPE ``cos``/``sin``,
    the valid-token indices used to zero padded outputs, and the sliding-window attention mask.
    Computed once in ``EsmFold2AtomEncoder._compute_step_invariants`` and threaded through every
    atom-stack layer (replaces the old positional ``attention_params`` tuple)."""

    cos: Tensor
    sin: Tensor
    valid_indices: Tensor
    swa_mask: Tensor


@dataclass
class EsmFold2AtomEncoderCache:
    """Step-invariant atom-encoder tensors cached across diffusion sampling steps (identical for every
    step of a fold): the atom base embedding, the SWA attention inputs, the expanded atom mask, the
    token count and the expanded atom->token map. Stored on ``EsmFold2DiffusionCache.atom_encoder``."""

    c_base: Tensor
    attention: EsmFold2AtomAttention
    mask_exp: Tensor
    n_tokens: int
    atom_to_token_exp: Tensor


@dataclass
class EsmFold2DiffusionCache:
    """Per-fold cache of the diffusion module's step-invariant tensors, reused across every denoising
    step of the sampler (see ``EsmFold2DiffusionStructureHead.sample``).

    This is **not** a generation/KV cache: it depends only on the fixed inputs and frozen weights (not
    on the noisy coordinates or noise level that change each step), it is fully populated on the first
    step and read-only thereafter, and caching is bit-identical to recomputing. It holds the
    atom-encoder tensors, the conditioning pair representation ``z``, and the diffusion-transformer's
    per-block attention pair biases (keyed by block index).
    """

    atom_encoder: EsmFold2AtomEncoderCache | None = None
    pair_repr: Tensor | None = None
    token_pair_bias: dict[int, Tensor] = field(default_factory=dict)


class EsmFold2LayerNorm(nn.LayerNorm):
    """LayerNorm that always computes in fp32, with its weight stored at the model dtype.

    A bf16 load rounds the weight to bf16, but the norm itself still computes in fp32
    (upcast here, cast back); an fp32 load is a no-op.
    """

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(x.dtype)


class EsmFold2TransitionLayer(nn.Module):
    """EsmFold2SwiGLU transition: norm -> a_proj/b_proj -> silu(gate) * up -> out_proj."""

    def __init__(self, d_model: int, n: int, eps: float = 1e-5) -> None:
        super().__init__()
        hidden = n * d_model
        self.norm = EsmFold2LayerNorm(d_model, eps=eps)
        self.a_proj = nn.Linear(d_model, hidden, bias=False)
        self.b_proj = nn.Linear(d_model, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        gate = self.a_proj(x)
        up = self.b_proj(x)
        return self.out_proj(F.silu(gate) * up)


class EsmFold2AdaptiveLayerNorm(nn.Module):
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
        # gate/shift in bf16; a_norm is fp32 so the affine promotes to fp32, then
        # downcast for the next op.
        gate = torch.sigmoid(self.s_gate(s_norm))
        shift = self.s_shift(s_norm)
        return (gate * a_norm + shift).to(a.dtype)


class EsmFold2FourierEmbedding(nn.Module):
    """Fourier embedding ``cos(2*pi*(t*frequencies + phases))`` with fixed (non-learnable) random
    frequencies and phases sampled once and stored in the checkpoint."""

    frequencies: Tensor
    phases: Tensor

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.randn(embedding_dim))
        self.register_buffer("phases", torch.randn(embedding_dim))

    def forward(self, t_hat: Tensor) -> Tensor:
        # frequencies/phases are kept fp32 (see EsmFold2Model._keep_in_fp32_modules_strict) so the
        # cos embedding is full precision.
        t = t_hat.to(dtype=self.frequencies.dtype).reshape(-1)
        return torch.cos(2.0 * torch.pi * (t[:, None] * self.frequencies[None, :] + self.phases[None, :]))


class EsmFold2SwiGLU(nn.Module):
    """SwiGLU feed-forward with a fused gate+up projection (``gate_up_proj``) and output ``down_proj``.

    ``intermediate_size`` is supplied by the caller and registered on the config, so every
    ESMFold2 SwiGLU feed-forward is this one module regardless of how its width is derived.
    """

    def __init__(
        self,
        in_features: int,
        intermediate_size: int,
        out_features: int | None = None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.gate_up_proj = nn.Linear(in_features, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, out_features, bias=False)
        self.intermediate_size = intermediate_size

    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def apply_rotary_emb_3d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply 3D RoPE with batch-dependent cos/sin.

    Args:
        x: [B, L, H, D]
        cos: [B, L, D] (full head dim, built by ``EsmFold2RotaryEmbedding3D``)
        sin: [B, L, D]
    """
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return x * cos + rotate_half(x) * sin


def qk_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def _resolve_atom_config(config: EsmFold2Config, structure_prediction: bool):
    """Resolve the SWA atom-stack hyperparameters for a given call site.

    The atom transformer is the same architecture in two places: the inputs
    embedder (``structure_prediction=False``) and the diffusion module
    (``structure_prediction=True``). Its I/O dims, block/head counts and FFN width
    come from the call-site's ``atom_encoder_*`` / ``diffusion_*`` config fields; the
    window and 3D-RoPE settings always come from the ``atom_encoder_*`` / ``sliding_window``
    fields (the diffusion module reused those — its atom encoder was built with the same
    window/RoPE), so call sites read those directly off ``config``.

    Every module in the atom stack (attention/block/transformer/encoder/decoder)
    takes only ``(config, structure_prediction)`` and derives its own dims from
    this, so no scalar dims are threaded between them.

    Returns ``(d_atom, d_token, n_blocks, n_heads, ffn_intermediate_size)`` — the SwiGLU
    FFN width for this call site's atom stack.
    """
    if structure_prediction:
        return (
            config.diffusion_atom_hidden_size,
            config.diffusion_token_hidden_size,
            config.diffusion_atom_num_blocks,
            config.diffusion_atom_num_heads,
            config.diffusion_atom_ffn_intermediate_size,
        )
    return (
        config.atom_encoder_hidden_size,
        config.atom_encoder_token_hidden_size,
        config.atom_encoder_num_hidden_layers,
        config.atom_encoder_num_attention_heads,
        config.atom_encoder_ffn_intermediate_size,
    )


def _swa_window_mask_function(valid: Tensor, half_window: int) -> Callable:
    """Sliding-window ``and`` mask over token index, with self-attention always allowed.

    A token attends to another iff both are valid (non-padding) and their indices differ by at
    most ``half_window`` (inputs are right-padded, so index distance is the window distance). The
    diagonal is always allowed so that fully padded query rows keep a single valid key and do not
    produce ``NaN`` in the softmax — which is why this stays a small ``and`` mask rather than the
    stock ``create_bidirectional_sliding_window_mask``.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        within = abs(q_idx - kv_idx) <= half_window
        in_window = within & valid[batch_idx, q_idx] & valid[batch_idx, kv_idx]
        return in_window | (q_idx == kv_idx)

    return inner_mask


class EsmFold2SWA3DRoPEAttention(nn.Module):
    """Sliding window self-attention with 3D RoPE. Has q/k/v/gate/out projections.

    The plain ``softmax(QKᵀ)V`` core is dispatched through the v5 attention
    interface (``config._attn_implementation``: ``eager`` / ``sdpa`` / ...), with
    the sliding window expressed as an additive attention mask. The shared
    ``config`` is passed in at construction, so ``set_attn_implementation()`` stays
    live (it mutates the same object); dims/window are derived from
    ``(config, structure_prediction)`` via :func:`_resolve_atom_config`.
    """

    def __init__(self, config: EsmFold2Config, structure_prediction: bool = True) -> None:
        super().__init__()
        d_model, _d_token, _n_blocks, n_heads, _ffn = _resolve_atom_config(config, structure_prediction)
        self.config = config
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        # No grouped-query attention; identity repeat keeps the interface happy.
        self.num_key_value_groups = 1
        # Bidirectional encoder: never let the sdpa/flash interface default to
        # causal masking when attention_mask happens to be None.
        self.is_causal = False

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, attention: EsmFold2AtomAttention) -> Tensor:
        B, N = x.shape[:2]
        cos, sin = attention.cos, attention.sin

        x_input = x
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim)
        q, k = qk_norm(q), qk_norm(k)

        q = apply_rotary_emb_3d(q, cos, sin)
        k = apply_rotary_emb_3d(k, cos, sin)

        input_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        attn_impl = self.config._attn_implementation

        # Sliding-window mask is built once per fold in EsmFold2AtomEncoder._compute_step_invariants
        # (it is identical across all atom-stack layers); reuse it here. ``valid`` is still needed to
        # zero out padded-token outputs below.
        valid = torch.zeros(B * N, dtype=torch.bool, device=q.device)
        valid[attention.valid_indices] = True
        valid = valid.view(B, N)
        attn_mask = attention.swa_mask

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(attn_impl, eager_attention_forward)
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


class EsmFold2SWAAtomLayer(nn.Module):
    """adaLN-Zero + SWA attention + EsmFold2SwiGLU FFN.

    The adaLN-Zero modulation is ``adaln_linear`` applied to ``silu(atom_cond)`` (zero-init gate).
    """

    def __init__(self, config: EsmFold2Config, structure_prediction: bool = True) -> None:
        super().__init__()
        d_atom, _d_token, _n_blocks, _n_heads, ffn_intermediate_size = _resolve_atom_config(
            config, structure_prediction
        )
        # adaln-Zero gate; zero-init lives in EsmFold2PreTrainedModel._init_weights.
        self.adaln_linear = nn.Linear(d_atom, 6 * d_atom, bias=False)

        self.attn = EsmFold2SWA3DRoPEAttention(config, structure_prediction)
        self.ffn = EsmFold2SwiGLU(d_atom, ffn_intermediate_size, d_atom)

    def forward(self, x: Tensor, atom_cond: Tensor, attention: EsmFold2AtomAttention) -> Tensor:
        modulation = self.adaln_linear(F.silu(atom_cond))
        if modulation.dim() == 2:
            modulation = modulation.unsqueeze(1)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.chunk(6, dim=-1)

        attn_input = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_a) + shift_a
        attn_out = self.attn(attn_input, attention)
        x = x + gate_a * attn_out

        ffn_input = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_f) + shift_f
        ffn_out = self.ffn(ffn_input)
        x = x + gate_f * ffn_out
        return x


class EsmFold2RotaryEmbedding3D(nn.Module):
    """Rotary embedding over continuous 3D atom coordinates plus a discrete space UID.

    Unlike sequence RoPE, this encodes physical position (``ref_pos`` = x/y/z) and a
    per-atom space UID rather than token indices, with separate spatial and UID base
    frequencies. ``forward`` returns cos/sin already at the full head dim, so the caller
    applies plain rotate-half RoPE (:func:`apply_rotary_emb_3d`). The inverse frequencies
    are cheap to rebuild each call and are computed on the input device to stay bit-exact.
    """

    def __init__(self, config: EsmFold2Config, structure_prediction: bool = True) -> None:
        super().__init__()
        d_atom, _d_token, _n_blocks, n_heads, _ffn = _resolve_atom_config(config, structure_prediction)
        self.head_dim = d_atom // n_heads
        self.n_spatial_per_axis = config.atom_encoder_n_spatial_rope_pairs_per_axis
        self.n_uid_pairs = config.atom_encoder_n_uid_rope_pairs
        self.spatial_base_freq = config.atom_encoder_spatial_rope_base_frequency
        self.uid_base_freq = config.atom_encoder_uid_rope_base_frequency

    def forward(self, ref_pos: Tensor, ref_space_uid: Tensor) -> tuple[Tensor, Tensor]:
        device = ref_pos.device
        B, N = ref_pos.shape[:2]
        half_dim = self.head_dim // 2
        n_spatial_total = 3 * self.n_spatial_per_axis

        spatial_inv_freq = 1.0 / (
            self.spatial_base_freq
            ** (torch.arange(0, self.n_spatial_per_axis, dtype=torch.float32, device=device) / self.n_spatial_per_axis)
        )
        uid_inv_freq = 1.0 / (
            self.uid_base_freq
            ** (torch.arange(0, self.n_uid_pairs, dtype=torch.float32, device=device) / self.n_uid_pairs)
        )

        spatial_freqs = (ref_pos.float().unsqueeze(-1) * spatial_inv_freq).reshape(B, N, n_spatial_total)
        uid_freqs = ref_space_uid.float().unsqueeze(-1) * uid_inv_freq

        freqs = torch.cat([spatial_freqs, uid_freqs], dim=-1)
        n_active = n_spatial_total + self.n_uid_pairs
        if n_active < half_dim:
            freqs = torch.cat([freqs, freqs.new_zeros(B, N, half_dim - n_active)], dim=-1)

        # Duplicate to the full head dim so the caller applies standard rotate-half RoPE.
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


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


class EsmFold2AtomEncoder(nn.Module):
    """SWA atom encoder with atom_linear, atom_norm, atom_to_token_linear, [coords_linear], rotary_emb, blocks.

    ``structure_prediction=True`` (diffusion module) adds ``coords_linear`` and
    aggregates to the full ``d_token``; ``False`` (inputs embedder) omits coords
    and aggregates to ``d_token // 2``. All dims/hyperparameters are read from
    ``config`` via :func:`_resolve_atom_config`.
    """

    def __init__(self, config: EsmFold2Config, structure_prediction: bool = True) -> None:
        super().__init__()
        d_atom, d_token, n_blocks, _n_heads, _ffn = _resolve_atom_config(config, structure_prediction)
        self.d_atom = d_atom
        self.d_token = d_token
        self.structure_prediction = structure_prediction

        # Atom feature width (`config.atom_feature_dim`) = 3 (xyz) + 1 (charge) + 1 (mask) + element +
        # atom-name-char one-hots; `char_feature_dim` is the atom-name-char slice used by the featurizer.
        self.char_feature_dim = config.char_vocab_size * config.max_chars
        self.atom_linear = nn.Linear(config.atom_feature_dim, d_atom, bias=False)
        self.atom_norm = EsmFold2LayerNorm(d_atom)

        if structure_prediction:
            self.coords_linear = nn.Linear(6, d_atom, bias=False)

        self.config = config
        self.rotary_emb = EsmFold2RotaryEmbedding3D(config, structure_prediction)
        self.layers = nn.ModuleList([EsmFold2SWAAtomLayer(config, structure_prediction) for _ in range(n_blocks)])

        # Output aggregation: d_token for structure prediction, d_token//2 for inputs
        out_dim = d_token if structure_prediction else d_token // 2
        self.atom_to_token_linear = nn.Linear(d_atom, out_dim, bias=False)

    def _compute_step_invariants(
        self,
        atom_inputs: EsmFold2AtomInputs,
        num_diffusion_samples: int,
    ) -> tuple[Tensor, tuple, Tensor, int]:
        """Tensors that don't change across diffusion steps (cached per fold): the atom base
        embedding ``c_base``, the attention params ``(cos, sin, indices, swa_mask)`` (3D-RoPE plus the
        sliding-window mask, which is identical across all atom-stack layers), the expanded atom mask,
        and n_tokens."""
        ref_pos = atom_inputs.ref_pos
        B, N = ref_pos.shape[:2]
        atom_feats = torch.cat(
            [
                ref_pos,
                atom_inputs.ref_charge.unsqueeze(-1),
                atom_inputs.atom_attention_mask.unsqueeze(-1),
                atom_inputs.ref_element,
                atom_inputs.ref_atom_name_chars.reshape(B, N, self.char_feature_dim),
            ],
            dim=-1,
        )
        c_base = self.atom_norm(self.atom_linear(atom_feats.to(self.atom_linear.weight.dtype)).float()).to(
            self.atom_linear.weight.dtype
        )
        cos, sin = self.rotary_emb(ref_pos, atom_inputs.ref_space_uid)
        cos = cos.repeat_interleave(num_diffusion_samples, 0)
        sin = sin.repeat_interleave(num_diffusion_samples, 0)
        mask_exp = atom_inputs.atom_attention_mask.repeat_interleave(num_diffusion_samples, 0)
        indices = torch.nonzero(mask_exp.flatten(), as_tuple=False).flatten()
        # The SWA mask depends only on the (step- and layer-invariant) valid-token mask and window,
        # so build it once here rather than in every atom-stack layer's forward. ``cos`` (bf16, same
        # batch/seq as the attention queries) supplies the mask metadata (dtype/device/shape).
        valid = mask_exp.bool()
        swa_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=cos,
            attention_mask=None,
            and_mask_function=_swa_window_mask_function(valid, self.config.sliding_window // 2),
        )
        attention = EsmFold2AtomAttention(cos, sin, indices, swa_mask)
        n_tokens = int(atom_inputs.atom_to_token.max().item()) + 1
        return c_base, attention, mask_exp, n_tokens

    def forward(
        self,
        atom_inputs: EsmFold2AtomInputs,
        atom_coords: Tensor | None = None,
        num_diffusion_samples: int = 1,
        inference_cache: EsmFold2DiffusionCache | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, EsmFold2AtomAttention]:
        """Returns (token_acts, atom_queries, atom_cond, attention).

        ``inference_cache.atom_encoder`` caches the step-invariant tensors (see
        :class:`EsmFold2AtomEncoderCache`) across diffusion sampling steps.
        """
        cache = inference_cache.atom_encoder if inference_cache is not None else None
        if cache is None:
            c_base, attention, mask_exp, n_tokens = self._compute_step_invariants(atom_inputs, num_diffusion_samples)
            atom_to_token_exp = atom_inputs.atom_to_token.repeat_interleave(num_diffusion_samples, 0)
            if inference_cache is not None:
                cache = EsmFold2AtomEncoderCache(c_base, attention, mask_exp, n_tokens, atom_to_token_exp)
                inference_cache.atom_encoder = cache
        else:
            c_base, attention, mask_exp, n_tokens = cache.c_base, cache.attention, cache.mask_exp, cache.n_tokens
            atom_to_token_exp = cache.atom_to_token_exp

        atom_cond = c_base

        atom_queries = atom_cond

        if self.structure_prediction:
            atom_queries = atom_queries.repeat_interleave(num_diffusion_samples, 0)
            # The second coord slot (a predicted-coords channel in the research model) is unused in
            # this release — always zeros — so coords_linear sees [atom_coords, 0].
            coord_input = torch.cat([atom_coords, torch.zeros_like(atom_coords)], dim=-1)
            coords_to_queries = self.coords_linear(coord_input.to(self.coords_linear.weight.dtype))
            atom_queries = atom_queries + coords_to_queries

        atom_cond = atom_cond.repeat_interleave(num_diffusion_samples, 0)

        for layer in self.layers:
            atom_queries = layer(atom_queries, atom_cond, attention)

        queries_to_acts = F.relu(self.atom_to_token_linear(atom_queries))
        token_acts = scatter_atom_to_token(queries_to_acts, atom_to_token_exp, n_tokens, atom_mask=mask_exp.bool())

        return token_acts, atom_queries, atom_cond, attention


def _gather_along_dim1(source: Tensor, index: Tensor) -> Tensor:
    """Gather ``source`` (``[B, N, d]``) along dim 1 with a ``[B, M]`` index, returning ``[B, M, d]``."""
    idx = index.unsqueeze(-1).expand(-1, -1, source.size(-1))
    return torch.gather(source, 1, idx)


class EsmFold2AtomDecoder(nn.Module):
    """SWA atom decoder with token_to_atom_linear, blocks, norm, output_linear.

    Only used inside the diffusion module, so its atom dims are always the
    structure-prediction (diffusion) ones (:func:`_resolve_atom_config`).
    """

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        d_atom, d_token, n_blocks, _n_heads, _ffn = _resolve_atom_config(config, structure_prediction=True)
        self.token_to_atom_linear = nn.Linear(d_token, d_atom, bias=False)

        self.layers = nn.ModuleList([EsmFold2SWAAtomLayer(config, structure_prediction=True) for _ in range(n_blocks)])

        self.norm = EsmFold2LayerNorm(d_atom)
        self.output_linear = nn.Linear(d_atom, 3, bias=False)  # (x, y, z) coordinates

    def forward(
        self,
        token_acts: Tensor,
        atom_queries: Tensor,
        atom_cond: Tensor,
        attention: EsmFold2AtomAttention,
        atom_inputs: EsmFold2AtomInputs,
        num_diffusion_samples: int = 1,
    ) -> Tensor:
        """Returns coord_update."""
        atom_to_token_exp = atom_inputs.atom_to_token.repeat_interleave(num_diffusion_samples, 0)
        a_to_q = self.token_to_atom_linear(token_acts)
        a_to_q = _gather_along_dim1(a_to_q, atom_to_token_exp)
        atom_queries = atom_queries + a_to_q

        for layer in self.layers:
            atom_queries = layer(atom_queries, atom_cond, attention)

        atom_coords = self.output_linear(self.norm(atom_queries))
        return atom_coords


class EsmFold2AttentionPairBias(nn.Module):
    """Gated multi-head attention with pair bias conditioning."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.config = config
        d_model = config.diffusion_token_hidden_size
        d_pair = config.pairwise_hidden_size  # the trunk pair rep flows in at this width
        num_heads = config.diffusion_token_num_heads
        d_cond = config.diffusion_token_hidden_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        # No grouped-query attention; identity repeat keeps the attention interface happy.
        self.num_key_value_groups = 1
        self.is_causal = False

        self.adaln = EsmFold2AdaptiveLayerNorm(d_model, d_cond)
        self.out_gate = nn.Linear(d_cond, d_model, bias=True)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.pair_norm = EsmFold2LayerNorm(d_pair)
        self.pair_bias_proj = nn.Linear(d_pair, num_heads, bias=False)

    def compute_pair_bias(self, pair_repr: Tensor, bsz: int, num_diffusion_samples: int = 1) -> Tensor:
        """Project the (normed) pair representation to per-head attention biases.

        Depends only on ``pair_repr`` and this block's fixed weights, so it is invariant
        across diffusion sampling steps — the sampler computes it once and reuses
        it (see ``EsmFold2DiffusionTransformer.forward``). Bit-identical to computing it
        inline every step.
        """
        if pair_repr.dim() == 4 and pair_repr.shape[0] != bsz and num_diffusion_samples > 1:
            pair_repr = pair_repr.repeat_interleave(num_diffusion_samples, dim=0)
        if pair_repr.dim() == 4:
            return self.pair_bias_proj(self.pair_norm(pair_repr))
        return pair_repr.unsqueeze(-1)

    def forward(
        self,
        token_acts: Tensor,
        single_repr: Tensor,
        pair_repr: Tensor,
        attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
        pair_bias: Tensor | None = None,
    ) -> Tensor:
        bsz, n_queries, d_model = token_acts.shape

        x = self.adaln(token_acts, single_repr)

        n_keys = x.shape[1]
        q = self.q_proj(x).view(bsz, n_queries, self.num_heads, self.head_dim)
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(bsz, n_keys, self.num_heads, self.head_dim)
        v = v.view(bsz, n_keys, self.num_heads, self.head_dim)

        if attention_mask is not None and attention_mask.shape[0] != bsz and num_diffusion_samples > 1:
            attention_mask = attention_mask.repeat_interleave(num_diffusion_samples, dim=0)

        gate = torch.sigmoid(self.g_proj(x)).view(bsz, n_queries, self.num_heads, self.head_dim)

        # ``pair_bias`` is step-invariant; the diffusion sampler precomputes and
        # caches it across steps. Compute inline when not supplied (e.g. uncached).
        if pair_bias is None:
            pair_bias = self.compute_pair_bias(pair_repr, bsz, num_diffusion_samples)

        attn_bias = pair_bias.permute(0, 3, 1, 2)  # [B,Q,K,H]->[B,H,Q,K] (H may be 1)
        if attention_mask is not None:
            min_val = torch.finfo(q.dtype).min
            mask_bias = torch.where(attention_mask.bool()[:, None, None, :], 0.0, min_val)
            attn_bias = attn_bias + mask_bias
        qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))  # [B,H,Q,D]
        # Route through the attention interface (respects config._attn_implementation) with the
        # per-head pair bias as the additive attention mask. Returns [B, Q, H, D].
        attn_impl = self.config._attn_implementation
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(attn_impl, eager_attention_forward)
        context, _ = attention_interface(self, qh, kh, vh, attn_bias.to(qh.dtype), dropout=0.0, scaling=self.scale)

        context = gate * context
        out = self.out_proj(context.reshape(bsz, n_queries, d_model).to(v.dtype))
        out = torch.sigmoid(self.out_gate(single_repr)) * out
        return out


class EsmFold2ConditionedTransitionBlock(nn.Module):
    """Conditioned EsmFold2SwiGLU transition with adaptive layer norm."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        d_model = config.diffusion_token_hidden_size
        d_cond = config.diffusion_token_hidden_size
        intermediate_size = config.diffusion_token_transition_intermediate_size

        self.adaln = EsmFold2AdaptiveLayerNorm(d_model, d_cond)
        self.output_gate = nn.Linear(d_cond, d_model, bias=True)

        self.ffn = EsmFold2SwiGLU(d_model, intermediate_size, d_model)

    def forward(self, token_acts: Tensor, single_repr: Tensor) -> Tensor:
        x = self.adaln(token_acts, single_repr)
        out = self.ffn(x)
        return torch.sigmoid(self.output_gate(single_repr)) * out


class EsmFold2DiffusionTransformer(nn.Module):
    """Diffusion denoising transformer with attention pair bias."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        num_blocks = config.diffusion_token_num_blocks
        self.attn_blocks = nn.ModuleList([EsmFold2AttentionPairBias(config) for _ in range(num_blocks)])
        self.transition_blocks = nn.ModuleList([EsmFold2ConditionedTransitionBlock(config) for _ in range(num_blocks)])

    def forward(
        self,
        token_acts: Tensor,
        single_repr: Tensor,
        pair_repr: Tensor,
        attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
        inference_cache: EsmFold2DiffusionCache | None = None,
    ) -> Tensor:
        x = token_acts
        bsz = token_acts.shape[0]
        # Each block's pair bias depends only on the (step-invariant) conditioning
        # pair repr and fixed weights, so compute it once per block and reuse it
        # across every diffusion sampling step. Bit-identical to recomputing it
        # each step; the cache lives in the sampler's per-fold ``inference_cache``.
        bias_cache = None if inference_cache is None else inference_cache.token_pair_bias
        for i, (attn, transition) in enumerate(zip(self.attn_blocks, self.transition_blocks)):
            if bias_cache is None:
                pair_bias = attn.compute_pair_bias(pair_repr, bsz, num_diffusion_samples)
            elif i in bias_cache:
                pair_bias = bias_cache[i]
            else:
                pair_bias = attn.compute_pair_bias(pair_repr, bsz, num_diffusion_samples)
                bias_cache[i] = pair_bias
            x = x + attn(
                x,
                single_repr,
                pair_repr,
                attention_mask=attention_mask,
                num_diffusion_samples=num_diffusion_samples,
                pair_bias=pair_bias,
            )
            x = x + transition(x, single_repr)
        return x


class EsmFold2DiffusionConditioning(nn.Module):
    """Conditions pair and single representations on noise timestep."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        # The conditioning's pair/single-inputs widths are the parent's pairwise_hidden_size /
        # single_inputs_size (the trunk pair rep and the embedder single-inputs tensor flow straight
        # into the norms below); the conditioning's single output is sized to the diffusion token width.
        c_z = config.pairwise_hidden_size
        c_s = config.diffusion_token_hidden_size
        c_s_inputs = config.single_inputs_size
        fourier_dim = config.diffusion_fourier_dim
        transition_multiplier = config.diffusion_transition_multiplier
        # The norms/transitions use their default eps (1e-5), as before.
        self.sigma_data = float(config.diffusion_sigma_data)

        self.z_input_norm = EsmFold2LayerNorm(2 * c_z)
        self.z_proj = nn.Linear(2 * c_z, c_z, bias=False)
        self.z_transitions = nn.ModuleList([EsmFold2TransitionLayer(c_z, n=transition_multiplier) for _ in range(2)])

        self.s_input_norm = EsmFold2LayerNorm(c_s_inputs)
        self.s_proj = nn.Linear(c_s_inputs, c_s, bias=False)
        self.fourier = EsmFold2FourierEmbedding(fourier_dim)
        self.noise_norm = EsmFold2LayerNorm(fourier_dim)
        self.noise_proj = nn.Linear(fourier_dim, c_s, bias=False)
        self.s_transitions = nn.ModuleList([EsmFold2TransitionLayer(c_s, n=transition_multiplier) for _ in range(2)])

    def forward(
        self,
        t_hat: Tensor,
        single_inputs: Tensor,
        pair_trunk: Tensor,
        relative_position_encoding: Tensor,
        sigma_data: float | None = None,
        num_diffusion_samples: int = 1,
        inference_cache: EsmFold2DiffusionCache | None = None,
    ) -> tuple[Tensor, Tensor]:
        sigma = self.sigma_data if sigma_data is None else float(sigma_data)
        base_batch = pair_trunk.shape[0]
        target_batch = base_batch * num_diffusion_samples

        # pair conditioning (cached across diffusion steps — independent of t_hat)
        if inference_cache is not None and inference_cache.pair_repr is not None:
            pair_repr = inference_cache.pair_repr
        else:
            rel_pos = relative_position_encoding.to(dtype=torch.float32)
            pair_repr = torch.cat([pair_trunk.to(dtype=torch.float32), rel_pos], dim=-1)
            # The relpos/coords conditioning is fp32; z_input_norm keeps it fp32,
            # then we hand off to z_proj in the model's compute dtype.
            pair_repr = self.z_proj(self.z_input_norm(pair_repr).to(self.z_proj.weight.dtype))
            for block in self.z_transitions:
                pair_repr = pair_repr + block(pair_repr)
            if inference_cache is not None:
                inference_cache.pair_repr = pair_repr

        # single conditioning
        single_inputs_eff = single_inputs
        if single_inputs_eff.shape[0] != target_batch:
            single_inputs_eff = single_inputs_eff.repeat_interleave(num_diffusion_samples, 0)

        single_repr = self.s_proj(
            self.s_input_norm(single_inputs_eff.to(dtype=torch.float32)).to(self.s_proj.weight.dtype)
        )

        # Noise embedding. ``t_hat`` already arrives at ``target_batch`` length — the diffusion
        # module (the only caller) normalizes it once before this call — so no re-expansion here.
        t = t_hat.to(dtype=torch.float32).reshape(-1)
        t_noise = 0.25 * torch.log((t / sigma).clamp(min=1e-20))
        noise_emb = self.fourier(t_noise)
        noise_emb = self.noise_proj(self.noise_norm(noise_emb.float()).to(self.noise_proj.weight.dtype))
        single_repr = single_repr + noise_emb.unsqueeze(1)

        for block in self.s_transitions:
            single_repr = single_repr + block(single_repr)

        return single_repr, pair_repr


class EsmFold2DiffusionModule(nn.Module):
    """Diffusion denoising module for structure prediction."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        c_token = config.diffusion_token_hidden_size
        self.sigma_data = float(config.diffusion_sigma_data)

        self.conditioning = EsmFold2DiffusionConditioning(config)
        self.atom_encoder = EsmFold2AtomEncoder(config, structure_prediction=True)
        self.atom_decoder = EsmFold2AtomDecoder(config)
        self.s_to_token = nn.Linear(c_token, c_token, bias=False)
        self.token_transformer = EsmFold2DiffusionTransformer(config)
        self.s_step_norm = EsmFold2LayerNorm(c_token)
        self.token_norm = EsmFold2LayerNorm(c_token)

    def forward(
        self,
        x_noisy: Tensor,
        t_hat: Tensor,
        atom_inputs: EsmFold2AtomInputs,
        single_inputs: Tensor,
        pair_trunk: Tensor,
        relative_position_encoding: Tensor,
        sigma_data: float | None = None,
        token_attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
        inference_cache: EsmFold2DiffusionCache | None = None,
    ) -> Tensor:
        bsz = x_noisy.shape[0]
        sigma = self.sigma_data if sigma_data is None else float(sigma_data)
        t = t_hat.to(dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(bsz)

        # Step 1: conditioning (pair repr is cached across diffusion steps)
        single_repr, pair_repr = self.conditioning(
            t_hat=t,
            single_inputs=single_inputs,
            pair_trunk=pair_trunk,
            relative_position_encoding=relative_position_encoding,
            sigma_data=sigma,
            num_diffusion_samples=num_diffusion_samples,
            inference_cache=inference_cache,
        )

        # Step 2: normalize noisy coords
        denominator = torch.sqrt(t * t + sigma * sigma)
        normalized_coords = x_noisy / denominator[:, None, None]

        # Step 3: atom encoder
        token_acts, atom_queries_skip, atom_cond_skip, atom_attention = self.atom_encoder(
            atom_inputs,
            atom_coords=normalized_coords,
            num_diffusion_samples=num_diffusion_samples,
            inference_cache=inference_cache,
        )

        # Step 4: add conditioned single repr
        token_acts = token_acts + self.s_to_token(self.s_step_norm(single_repr))

        # Step 5: token transformer (pair bias is cached across steps via inference_cache)
        token_acts = self.token_transformer(
            token_acts,
            single_repr,
            pair_repr,
            attention_mask=token_attention_mask,
            num_diffusion_samples=num_diffusion_samples,
            inference_cache=inference_cache,
        )

        # Step 6: token norm
        token_acts = self.token_norm(token_acts)

        # Step 7: atom decoder
        coord_update = self.atom_decoder(
            token_acts=token_acts,
            atom_queries=atom_queries_skip,
            atom_cond=atom_cond_skip,
            attention=atom_attention,
            atom_inputs=atom_inputs,
            num_diffusion_samples=num_diffusion_samples,
        )

        # Step 8: compute denoised output
        sigma2 = sigma * sigma
        t2 = t * t
        out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
        out = out + ((sigma * t) / torch.sqrt(sigma2 + t2))[:, None, None] * coord_update

        return out


class EsmFold2DiffusionStructureHead(nn.Module):
    """Wrapper around EsmFold2DiffusionModule with diffusion sampling."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.diffusion_module = EsmFold2DiffusionModule(config)

        # Sampling hyperparameters
        self.sigma_data = config.diffusion_sigma_data
        self.gamma_0 = config.structure_head_gamma_0
        self.gamma_min = config.structure_head_gamma_min
        self.noise_scale = config.structure_head_noise_scale
        self.step_scale = config.structure_head_step_scale
        self.inference_s_max = config.structure_head_inference_s_max
        self.inference_s_min = config.structure_head_inference_s_min
        self.inference_p = config.structure_head_inference_p
        self.inference_num_steps = config.structure_head_inference_num_steps
        self.max_inference_sigma = config.structure_head_max_inference_sigma

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
        ramp = torch.arange(steps, device=device, dtype=torch.float32)
        base = self.inference_s_max**inv_p + (ramp / (steps - 1)) * (
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
        denominator = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (x * mask).sum(dim=1, keepdim=True) / denominator
        x = x - mean
        if second_coords is not None:
            second_coords = second_coords - mean

        r = self._random_rotations(bsz, x.dtype, x.device)
        x = x @ r
        if second_coords is not None:
            second_coords = second_coords @ r

        t = torch.randn_like(x[:, 0:1, :])
        x = x + t
        if second_coords is not None:
            second_coords = second_coords + t
        return x, second_coords

    @staticmethod
    def _weighted_rigid_align(x: Tensor, x_gt: Tensor, w: Tensor, mask: Tensor) -> Tensor:
        """Kabsch alignment: align x to x_gt with weights w."""
        w = (mask * w).unsqueeze(-1)  # [B, N, 1]
        denominator = w.sum(dim=-2, keepdim=True).clamp(min=1e-8)
        centroid = (x * w).sum(dim=-2, keepdim=True) / denominator
        centroid_gt = (x_gt * w).sum(dim=-2, keepdim=True) / denominator
        x_centered = x - centroid
        x_gt_centered = x_gt - centroid_gt
        H = (w * x_gt_centered).transpose(-1, -2) @ x_centered
        H32 = H.float()
        U, _, Vh = torch.linalg.svd(H32, driver="gesvd" if H32.is_cuda else None)
        determinant = torch.linalg.det(U @ Vh)
        ones = torch.ones_like(determinant)
        R = (U @ torch.diag_embed(torch.stack([ones, ones, determinant], dim=-1)) @ Vh).to(H.dtype)
        return x_centered @ R.transpose(-1, -2) + centroid_gt

    @torch.inference_mode()
    def _build_noise_schedule(self, num_sampling_steps: int | None, device: torch.device) -> tuple[Tensor, Tensor]:
        """Karras σ schedule (Algorithm 18) + per-step γ churn factors.

        The schedule is capped at ``self.max_inference_sigma`` (from config): the high-σ tail above
        the cap is truncated and the cap re-prepended so sampling still starts from it.
        """
        steps = self.inference_num_steps if num_sampling_steps is None else int(num_sampling_steps)
        schedule = self.inference_noise_schedule(steps, device)
        max_inference_sigma = self.max_inference_sigma
        if max_inference_sigma is not None:
            schedule = schedule[schedule <= float(max_inference_sigma)]
            schedule = F.pad(schedule, (1, 0), value=float(max_inference_sigma))
        gammas = torch.where(
            schedule > self.gamma_min,
            torch.full_like(schedule, self.gamma_0),
            torch.zeros_like(schedule),
        )
        return schedule, gammas

    def sample(
        self,
        pair_trunk: Tensor,
        single_inputs: Tensor,
        relative_position_encoding: Tensor,
        atom_inputs: EsmFold2AtomInputs,
        token_attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
        num_sampling_steps: int | None = None,
    ) -> dict[str, Tensor | None]:
        """Diffusion sampling (Algorithm 18).

        ``num_sampling_steps`` is the number of denoising steps actually run. The remaining sampling
        hyperparameters (noise/step scales, the ``max_inference_sigma`` schedule cap) are read from
        config; see ``_build_noise_schedule``.
        """
        n_atoms = atom_inputs.atom_to_token.shape[1]
        device = single_inputs.device
        target_batch = single_inputs.shape[0] * num_diffusion_samples

        inference_cache = EsmFold2DiffusionCache()

        schedule, gammas = self._build_noise_schedule(num_sampling_steps, device)

        lam = self.noise_scale
        eta = self.step_scale

        x = schedule[0] * torch.randn(target_batch, n_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = atom_inputs.atom_attention_mask.repeat_interleave(num_diffusion_samples, 0).float()

        x_denoised_prev: Tensor | None = None

        step_pairs = list(zip(schedule[:-1], schedule[1:], gammas[1:]))

        for sigma_tm, sigma_t, gamma in step_pairs:
            x, x_denoised_prev = self._center_random_augmentation(x, atom_mask, second_coords=x_denoised_prev)

            sigma_tm_val = float(sigma_tm.item())
            t_hat_val = sigma_tm_val * (1.0 + float(gamma.item()))
            eps_std = lam * max(t_hat_val**2 - sigma_tm_val**2, 0.0) ** 0.5
            x_noisy = x + eps_std * torch.randn_like(x)

            x_denoised = self.diffusion_module(
                x_noisy=x_noisy,
                t_hat=torch.full((target_batch,), t_hat_val, device=device, dtype=torch.float32),
                atom_inputs=atom_inputs,
                single_inputs=single_inputs,
                pair_trunk=pair_trunk,
                relative_position_encoding=relative_position_encoding,
                token_attention_mask=token_attention_mask,
                num_diffusion_samples=num_diffusion_samples,
                inference_cache=inference_cache,
            )

            # Reverse diffusion alignment (Kabsch).
            x_noisy = self._weighted_rigid_align(x_noisy.float(), x_denoised.float(), atom_mask, atom_mask)
            x_noisy = x_noisy.to(dtype=x_denoised.dtype)

            # ODE/SDE step
            sigma_t_val = float(sigma_t.item())
            denoised_over_sigma = (x_noisy - x_denoised) / t_hat_val
            x = x_noisy + eta * (sigma_t_val - t_hat_val) * denoised_over_sigma

            x_denoised_prev = x_denoised

        return {"sample_atom_coords": x}


class EsmFold2RowAttentionPooling(nn.Module):
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
            torch.full_like(scores, torch.finfo(scores.dtype).min),
        )
        scores = scores + mask_bias
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        pooled = torch.einsum("bnm,bnmd->bnd", weights, z)
        return self.out_proj(pooled)


def _relative_position_one_hot(diff: Tensor, n_bins: int, keep_mask: Tensor) -> Tensor:
    """One-hot encode a relative index difference into ``2 * n_bins + 2`` classes.

    Classes ``[0, 2 * n_bins]`` hold the clipped relative offset; the final class
    ``2 * n_bins + 1`` is the "out-of-context" bin assigned wherever ``keep_mask`` is False
    (e.g. a pair spanning two chains).
    """
    binned = torch.clip(diff + n_bins, 0, 2 * n_bins)
    binned = torch.where(keep_mask, binned, 2 * n_bins + 1)
    return F.one_hot(binned, 2 * n_bins + 2)


class EsmFold2ResIdxAsymIdSymIdEntityIdEncoding(nn.Module):
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

        # Three relative encodings, all clip -> mask-out -> one-hot (see _relative_position_one_hot):
        # residue offset within a chain, token offset within a residue, and chain (sym_id) offset
        # across chains. The chain encoding keeps *cross*-chain pairs, so its mask is inverted.
        residx_bins, chain_bins = self.n_relative_residx_bins, self.n_relative_chain_bins
        aij_rel_pos = _relative_position_one_hot(
            residue_index.unsqueeze(2) - residue_index.unsqueeze(1), residx_bins, bij_same_chain
        )
        aij_rel_token = _relative_position_one_hot(
            token_index.unsqueeze(2) - token_index.unsqueeze(1), residx_bins, bij_same_chain & bij_same_residue
        )
        aij_rel_chain = _relative_position_one_hot(
            sym_id.unsqueeze(2) - sym_id.unsqueeze(1), chain_bins, ~bij_same_chain
        )

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


class EsmFold2SingleToPair(nn.Module):
    """downproject -> outer product/difference -> two-layer MLP (fc1, GELU, fc2)."""

    def __init__(self, input_dim: int, downproject_dim: int, output_dim: int) -> None:
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim)
        self.output_fc1 = nn.Linear(2 * downproject_dim, output_dim)
        self.output_fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.downproject(x)
        x = torch.cat(
            [(x.unsqueeze(2) * x.unsqueeze(1)), (x.unsqueeze(2) - x.unsqueeze(1))],
            dim=3,
        )
        return self.output_fc2(F.gelu(self.output_fc1(x)))


class EsmFold2LanguageModelShim(nn.Module):
    """Shim holding the trainable projection weights for LM integration.

    Contains:
    - base_z_combine: nn.Parameter [num_layers+1]
    - base_z_input_norm -> base_z_proj: EsmFold2LayerNorm(d_model) then Linear(d_model, d_z, bias=False)
    - base_z_to_pair -> base_z_output_norm: EsmFold2SingleToPair(d_z, d_z, d_z) then EsmFold2LayerNorm(d_z)
    """

    def __init__(self, d_z: int = 256, d_model: int = 2560, num_layers: int = 80) -> None:
        super().__init__()

        self.base_z_to_pair = EsmFold2SingleToPair(d_z, d_z, d_z)
        self.base_z_output_norm = EsmFold2LayerNorm(d_z)
        self.base_z_input_norm = EsmFold2LayerNorm(d_model)
        self.base_z_proj = nn.Linear(d_model, d_z, bias=False)
        self.base_z_combine = nn.Parameter(torch.zeros(num_layers + 1))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project pre-computed ESMC hidden states to pair representation.

        Args:
            hidden_states: [B, L, num_layers+1, d_model] from ESMC 6B.

        Returns:
            [B, L, L, d_pair] pair representation.
        """
        hidden_states = hidden_states.to(self.base_z_proj.weight.dtype)
        # base_z_input_norm is an fp32-pinned LayerNorm; upcast in, downcast out.
        normed = self.base_z_input_norm(hidden_states)
        lm_z = self.base_z_proj(normed)  # [B, L, 81, d_z]
        weights = self.base_z_combine.softmax(0)  # [81]
        lm_z = (weights @ lm_z).squeeze(-2)  # [B, L, d_z]
        # base_z_output_norm is an fp32-pinned LayerNorm; upcast in, downcast out.
        pair = self.base_z_to_pair(lm_z)
        lm_z = self.base_z_output_norm(pair)  # [B, L, L, d_z]
        return lm_z


@use_kernel_forward_from_hub("EsmFold2TriangleMultiplication")
class EsmFold2TriangleMultiplicativeUpdate(nn.Module):
    """Triangle multiplicative update with gated signal routing and explicit orientation.

    The O(N^3) contraction is the trunk's dominant cost; ``use_kernels=True`` (CUDA +
    inference) swaps this whole forward for a fused Triton Hub kernel matching the
    ``(pair_grid, visibility)`` signature and returning the residual-free delta.
    """

    _FLOW_TO_EINSUM = {"outgoing": "bikd,bjkd->bijd", "incoming": "bkid,bkjd->bijd"}

    def __init__(self, dim: int, outgoing: bool = True, chunk_size: int | None = 64) -> None:
        super().__init__()
        self.dim = dim
        self.flow = "outgoing" if outgoing else "incoming"
        self._einsum_equation = self._FLOW_TO_EINSUM[self.flow]
        self.norm_start = EsmFold2LayerNorm(dim)
        self.norm_mix = EsmFold2LayerNorm(dim)
        self.proj_bundle = nn.Linear(dim, 4 * dim, bias=False)
        self.proj_emit = nn.Linear(dim, dim, bias=False)
        self.proj_gate = nn.Linear(dim, dim, bias=False)

        # Chunk the O(N^3) contraction for memory on long sequences (chunk_size=None disables).
        self._chunk_size: int | None = chunk_size

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
        signal, gate_logits = bundled.split(2 * self.dim, dim=-1)
        # Gates and the O(N^3) contraction run in the activation dtype (bf16, the
        # training dtype); ``norm_start``/``norm_mix`` stay fp32.
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


class EsmFold2Transition(nn.Module):
    """LayerNorm + EsmFold2SwiGLU feed-forward residual block, chunked along the token axis."""

    def __init__(self, d_model: int, intermediate_size: int, chunk_size: int | None = 64) -> None:
        super().__init__()
        self.norm = EsmFold2LayerNorm(d_model)
        self.ffn = EsmFold2SwiGLU(d_model, intermediate_size, d_model)
        # Chunk along the token axis on long sequences (chunk_size=None disables).
        self._chunk_size: int | None = chunk_size

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def forward(self, x: Tensor) -> Tensor:
        if self._chunk_size is None or x.shape[1] <= self._chunk_size:
            return x + self.ffn(self.norm(x))
        out_list: list[Tensor] = []
        for start in range(0, x.shape[1], self._chunk_size):
            end = min(start + self._chunk_size, x.shape[1])
            chunk_x = x[:, start:end]
            out_list.append(chunk_x + self.ffn(self.norm(chunk_x)))
        return torch.cat(out_list, dim=1)


class EsmFold2PairUpdateBlock(nn.Module):
    """tri_mul_out, tri_mul_in, pair_transition."""

    def __init__(self, d_pair: int, intermediate_size: int, chunk_size: int | None = 64) -> None:
        super().__init__()
        self.tri_mul_out = EsmFold2TriangleMultiplicativeUpdate(dim=d_pair, outgoing=True, chunk_size=chunk_size)
        self.tri_mul_in = EsmFold2TriangleMultiplicativeUpdate(dim=d_pair, outgoing=False, chunk_size=chunk_size)
        self.pair_transition = EsmFold2Transition(d_pair, intermediate_size, chunk_size=chunk_size)

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.tri_mul_out.set_chunk_size(chunk_size)
        self.tri_mul_in.set_chunk_size(chunk_size)
        self.pair_transition.set_chunk_size(chunk_size)

    def forward(self, pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        # Inference-only: trained row-shared dropout omitted.
        pair = pair + self.tri_mul_out(pair, visibility=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, visibility=pair_attention_mask)
        pair = self.pair_transition(pair)
        return pair


class EsmFold2FoldingTrunk(nn.Module):
    """ModuleList of PairUpdateBlocks."""

    def __init__(self, n_layers: int, d_pair: int, intermediate_size: int, chunk_size: int | None = 64) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EsmFold2PairUpdateBlock(d_pair=d_pair, intermediate_size=intermediate_size, chunk_size=chunk_size)
                for _ in range(n_layers)
            ]
        )

    def set_chunk_size(self, chunk_size: int | None) -> None:
        for layer in self.layers:
            layer.set_chunk_size(chunk_size)

    def forward(self, pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            layer_fn = partial(layer, pair_attention_mask=pair_attention_mask)
            if torch.is_grad_enabled():
                pair = checkpoint(layer_fn, pair, use_reentrant=False)
            else:
                pair = layer_fn(pair)
        return pair


class EsmFold2OuterProductMean(nn.Module):
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
        self.norm = EsmFold2LayerNorm(d_msa)
        self.W = nn.Linear(d_msa, 2 * d_hidden, bias=False)
        self.Wout = nn.Linear(d_hidden * d_hidden, d_pair, bias=True)
        # Off for bit-exact bf16; ``set_chunk_size(64)`` for long sequences.
        self._chunk_size: int | None = None

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self._chunk_size = chunk_size

    def forward(self, msa_repr: Tensor, msa_attention_mask: Tensor) -> Tensor:
        msa_normed = self.norm(msa_repr)
        x = self.W(msa_normed) * msa_attention_mask.unsqueeze(-1).to(msa_normed.dtype)
        left, right = x.chunk(2, dim=-1)
        mask_f = msa_attention_mask.to(left.dtype)
        n_valid = (mask_f @ mask_f.transpose(-1, -2)).unsqueeze(-1).clamp(min=1.0)
        if self._chunk_size is None:
            outer = torch.einsum("bimc,bjmd->bijcd", left, right).flatten(-2)
            if self.divide_outer_before_proj:
                return self.Wout(outer / n_valid)
            return self.Wout(outer) / n_valid
        # Chunk along the left (i) axis so the peak einsum intermediate is
        # [B, chunk, L, c, d] instead of [B, L, L, c, d].
        L = left.shape[1]
        out_chunks: list[Tensor] = []
        for start in range(0, L, self._chunk_size):
            end = min(start + self._chunk_size, L)
            outer_chunk = torch.einsum("bimc,bjmd->bijcd", left[:, start:end], right).flatten(-2)
            if self.divide_outer_before_proj:
                out_chunks.append(self.Wout(outer_chunk / n_valid[:, start:end]))
            else:
                out_chunks.append(self.Wout(outer_chunk) / n_valid[:, start:end])
        return torch.cat(out_chunks, dim=1)


class EsmFold2MSAPairWeightedAveraging(nn.Module):
    """Pair-biased MSA row update (AF3 Supplement Algorithm 10)."""

    def __init__(self, d_msa: int, d_pair: int, n_heads: int = 8, head_width: int = 32) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_width = head_width
        self.norm_single = EsmFold2LayerNorm(d_msa)
        self.bias_norm = EsmFold2LayerNorm(d_pair)
        self.bias_proj = nn.Linear(d_pair, n_heads, bias=False)
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
        bias = self.bias_proj(self.bias_norm(pair_repr))  # [B, L, L, n_heads]
        bias.masked_fill_(~pair_attention_mask.unsqueeze(-1).bool(), -1e5)
        attn = torch.softmax(bias, dim=-2, dtype=torch.float32).to(bias.dtype)  # softmax over j

        v = self.Wv(msa_normed).reshape(B, L, M, h, dh)
        gate = torch.sigmoid(self.Wgate(msa_normed)).reshape(B, L, M, h, dh)

        output = torch.einsum("bijh,bjmhd,bimhd->bimhd", attn, v, gate)
        return self.Wout(output.reshape(B, L, M, h * dh))


@dataclass
class EsmFold2Output(ModelOutput):
    """
    Output of [`EsmFold2Model`]. All confidence scores are on a 0-1 scale; per-sample tensors
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


class EsmFold2ConfidenceInputEmbedder(nn.Module):
    """Builds the confidence head's base pair representation from the trunk pair representation and
    the single-inputs tensor (input norms + single->pair projections, including the outer product)."""

    def __init__(self, d_pair: int, d_inputs: int) -> None:
        super().__init__()
        self.s_inputs_norm = EsmFold2LayerNorm(d_inputs)
        self.z_norm = EsmFold2LayerNorm(d_pair)
        self.s_to_z = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_transpose = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_in1 = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_in2 = nn.Linear(d_inputs, d_pair, bias=False)
        self.s_to_z_prod_out = nn.Linear(d_pair, d_pair, bias=False)

    def forward(
        self,
        single_inputs: Tensor,
        z: Tensor,
        relative_position_encoding: Tensor | None,
        token_bonds_encoding: Tensor | None,
    ) -> Tensor:
        s_inputs_normed = self.s_inputs_norm(single_inputs)

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
        return z_base


class EsmFold2ConfidenceHead(nn.Module):
    """Predicts pLDDT, PAE, PDE, resolved-atom probability and distogram bins."""

    boundaries: Tensor
    # Additive guard for masked-mean denominators (empty chains / all-padding rows). Kept as an
    # explicit ``+ eps`` (rather than clamping the denominator to 1) to reproduce the reference numerics.
    _CONFIDENCE_EPS: float = 1e-6

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        d_single = config.hidden_size
        d_pair = config.pairwise_hidden_size
        d_inputs = config.single_inputs_size
        distogram_bins = config.confidence_head_distogram_bins

        boundaries = torch.linspace(
            config.confidence_head_min_dist, config.confidence_head_max_dist, distogram_bins - 1
        )
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(distogram_bins, d_pair)

        self.input_embedder = EsmFold2ConfidenceInputEmbedder(d_pair=d_pair, d_inputs=d_inputs)

        self.row_attention_pooling = EsmFold2RowAttentionPooling(d_pair=d_pair, d_single=d_single)

        self.folding_trunk = EsmFold2FoldingTrunk(
            n_layers=config.confidence_head_num_hidden_layers,
            d_pair=d_pair,
            intermediate_size=config.pair_transition_intermediate_size,
            chunk_size=config.chunk_size,
        )

        # Heads.
        self.plddt_ln = EsmFold2LayerNorm(d_single)
        max_atoms_per_token = config.max_atoms_per_token
        self.plddt_weight = nn.Parameter(
            torch.zeros(max_atoms_per_token, d_single, config.confidence_head_num_plddt_bins)
        )

        self.pae_ln = EsmFold2LayerNorm(d_pair)
        self.pae_head = nn.Linear(d_pair, config.confidence_head_num_pae_bins, bias=False)

        self.pde_ln = EsmFold2LayerNorm(d_pair)
        self.pde_head = nn.Linear(d_pair, config.confidence_head_num_pde_bins, bias=False)

        self.resolved_ln = EsmFold2LayerNorm(d_single)
        # 2 = resolved logits ([unresolved, resolved]).
        self.resolved_weight = nn.Parameter(torch.zeros(max_atoms_per_token, d_single, 2))

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)

    def _build_pair_and_single(
        self,
        single_inputs: Tensor,
        z: Tensor,
        x_pred: Tensor,
        distogram_atom_idx: Tensor,
        token_attention_mask: Tensor,
        atom_to_token: Tensor,
        atom_attention_mask: Tensor,
        num_diffusion_samples: int,
        relative_position_encoding: Tensor | None,
        token_bonds_encoding: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """Build the per-sample pair + single representations shared by every confidence head.

        Returns ``(single, pair, mask, rep_distances, rep_idx_m, atom_to_token_m, atom_mask_m, Bm)``
        where the ``*_m`` tensors are repeated across the diffusion-sample batch axis.
        """
        z_base = self.input_embedder(single_inputs, z, relative_position_encoding, token_bonds_encoding)

        pair = z_base.repeat_interleave(num_diffusion_samples, 0)
        x_pred_flat = x_pred.reshape(-1, *x_pred.shape[-2:]) if x_pred.ndim == 4 else x_pred
        atom_to_token_m = atom_to_token.repeat_interleave(num_diffusion_samples, 0)
        atom_mask_m = atom_attention_mask.repeat_interleave(num_diffusion_samples, 0)
        rep_idx_m = distogram_atom_idx.repeat_interleave(num_diffusion_samples, 0).long()
        mask = token_attention_mask.repeat_interleave(num_diffusion_samples, 0)
        Bm = pair.shape[0]

        rep_coords = _gather_along_dim1(x_pred_flat, rep_idx_m)
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

        return single, pair, mask, rep_distances, rep_idx_m, atom_to_token_m, atom_mask_m, Bm

    def _compute_atom_confidences(
        self,
        single: Tensor,
        atom_to_token_m: Tensor,
        atom_mask_m: Tensor,
        rep_idx_m: Tensor,
        rep_distances: Tensor,
        expanded_type: Tensor,
        expanded_asym: Tensor,
        Bm: int,
    ) -> dict[str, Tensor]:
        """Per-atom confidence outputs off the single representation (pLDDT family + resolved)."""
        atom_mask_f = atom_mask_m.float()
        s_at_atoms = _gather_along_dim1(single, atom_to_token_m)
        s_at_atoms_ln = self.plddt_ln(s_at_atoms)

        intra_idx = _compute_intra_token_idx(atom_to_token_m)
        intra_idx = intra_idx.clamp(max=self.plddt_weight.shape[0] - 1)
        w_plddt = self.plddt_weight[intra_idx]
        plddt_logits = torch.einsum("...c,...cb->...b", s_at_atoms_ln, w_plddt)

        # The pLDDT-family metrics are reported summaries derived from the logits, not loss
        # targets (only the logits are trained), so build them under no_grad rather than
        # detaching each one afterward.
        with torch.no_grad():
            plddt_per_atom = _categorical_mean(plddt_logits, start=0.0, end=1.0)

            L = single.shape[1]
            plddt_sum = torch.zeros(Bm, L, device=single.device, dtype=plddt_per_atom.dtype)
            atom_count = torch.zeros(Bm, L, device=single.device, dtype=plddt_per_atom.dtype)
            atom_mask_t = atom_mask_f.to(plddt_per_atom.dtype)
            plddt_sum.scatter_add_(1, atom_to_token_m, plddt_per_atom * atom_mask_t)
            atom_count.scatter_add_(1, atom_to_token_m, atom_mask_t)
            plddt = plddt_sum / atom_count.clamp(min=1e-6)

            complex_plddt = (plddt_per_atom * atom_mask_f).sum(dim=-1) / (
                atom_mask_f.sum(dim=-1) + self._CONFIDENCE_EPS
            )

            is_ligand = (expanded_type == 4).float()  # 4 = non-polymer (ligand) molecule type
            inter_chain = (expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)).float()
            near_contact = (rep_distances < 8).float()
            interface_per_token = (near_contact * inter_chain * (1.0 - is_ligand).unsqueeze(-1)).amax(dim=-1)
            iplddt_weight = torch.where(
                is_ligand.bool(),
                torch.full_like(interface_per_token, 2.0),
                interface_per_token,
            )
            iplddt_weight_atoms = _gather_along_dim1(iplddt_weight.unsqueeze(-1), atom_to_token_m).squeeze(-1)
            atom_iplddt_w = atom_mask_f * iplddt_weight_atoms
            complex_iplddt = (plddt_per_atom * atom_iplddt_w).sum(dim=-1) / (
                atom_iplddt_w.sum(dim=-1) + self._CONFIDENCE_EPS
            )

            plddt_ca = plddt_per_atom.gather(1, rep_idx_m)

        # Resolved (per-atom binary): same per-atom single features, its own weight.
        s_at_atoms_res = self.resolved_ln(s_at_atoms)
        w_res = self.resolved_weight[intra_idx]
        resolved_logits = torch.einsum("...c,...cb->...b", s_at_atoms_res, w_res)

        return {
            "plddt_logits": plddt_logits,
            "plddt": plddt,
            "plddt_per_atom": plddt_per_atom,
            "plddt_ca": plddt_ca,
            "complex_plddt": complex_plddt,
            "complex_iplddt": complex_iplddt,
            "resolved_logits": resolved_logits,
        }

    @torch.no_grad()
    def _compute_ptm_iptm(
        self, pae_logits: Tensor, mask: Tensor, expanded_asym: Tensor, Bm: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """pTM / ipTM / per-chain-pair ipTM derived from the PAE logits (reported metrics only)."""
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
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (pair_mask_2d.sum(dim=-1) + self._CONFIDENCE_EPS)
        ptm = ptm_per_row.max(dim=-1).values

        inter_chain_mask = (expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)).float() * pair_mask_2d
        iptm_per_row = (tm_expected * inter_chain_mask).sum(dim=-1) / (
            inter_chain_mask.sum(dim=-1) + self._CONFIDENCE_EPS
        )
        iptm = iptm_per_row.max(dim=-1).values

        max_chain_id = int(expanded_asym.max().item()) if Bm > 0 else 0
        n_chains = max_chain_id + 1
        pair_chains_iptm = torch.zeros(Bm, n_chains, n_chains, device=tm_expected.device, dtype=tm_expected.dtype)
        # pair_chains_iptm[c1, c2] = max over rows i in chain c2 of the mean over
        # columns j in chain c1 of tm_expected[i, j] (max-of-row-mean, as in the
        # global iptm above), so iptm equals the max off-diagonal entry.
        for c1 in range(n_chains):
            chain_c1 = (expanded_asym == c1).float() * mask_f
            if chain_c1.sum() == 0:
                continue
            col_mask = chain_c1.unsqueeze(-2)
            avg_tm = (tm_expected * col_mask).sum(dim=-1) / (col_mask.sum(dim=-1) + self._CONFIDENCE_EPS)
            for c2 in range(n_chains):
                chain_c2 = (expanded_asym == c2).float() * mask_f
                row_vals = avg_tm.masked_fill(chain_c2 == 0, float("-inf"))
                pair_chains_iptm[:, c1, c2] = row_vals.max(dim=-1).values.clamp(min=0.0)

        return ptm, iptm, pair_chains_iptm

    def forward(
        self,
        single_inputs: Tensor,
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
        single, pair, mask, rep_distances, rep_idx_m, atom_to_token_m, atom_mask_m, Bm = self._build_pair_and_single(
            single_inputs=single_inputs,
            z=z,
            x_pred=x_pred,
            distogram_atom_idx=distogram_atom_idx,
            token_attention_mask=token_attention_mask,
            atom_to_token=atom_to_token,
            atom_attention_mask=atom_attention_mask,
            num_diffusion_samples=num_diffusion_samples,
            relative_position_encoding=relative_position_encoding,
            token_bonds_encoding=token_bonds_encoding,
        )

        expanded_type = mol_type.repeat_interleave(num_diffusion_samples, 0)
        expanded_asym = asym_id.repeat_interleave(num_diffusion_samples, 0)
        atom_confidences = self._compute_atom_confidences(
            single=single,
            atom_to_token_m=atom_to_token_m,
            atom_mask_m=atom_mask_m,
            rep_idx_m=rep_idx_m,
            rep_distances=rep_distances,
            expanded_type=expanded_type,
            expanded_asym=expanded_asym,
            Bm=Bm,
        )

        pae_logits = self.pae_head(self.pae_ln(pair))
        pde_logits = self.pde_head(self.pde_ln(pair))
        # Expected-value pae/pde are reported metrics; only the logits are trained.
        with torch.no_grad():
            pae = _categorical_mean(pae_logits, start=0.0, end=32.0)
            pde = _categorical_mean(pde_logits, start=0.0, end=32.0)

        ptm, iptm, pair_chains_iptm = self._compute_ptm_iptm(pae_logits, mask, expanded_asym, Bm)

        return {
            **atom_confidences,
            "pae_logits": pae_logits,
            "pae": pae,
            "pde_logits": pde_logits,
            "pde": pde,
            "ptm": ptm,
            "iptm": iptm,
            "pair_chains_iptm": pair_chains_iptm,
        }


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


class EsmFold2MSAEncoderBlock(nn.Module):
    """One MSA encoder block: OPM into pair, MSA pair-weighted averaging, triangle update."""

    def __init__(self, config: EsmFold2Config, is_final_block: bool = False) -> None:
        super().__init__()
        d_msa = config.msa_encoder_hidden_size
        d_pair = config.pairwise_hidden_size
        d_hidden = config.msa_encoder_outer_hidden_size
        n_heads_msa = config.msa_encoder_num_attention_heads
        msa_head_width = config.msa_encoder_head_width
        self.is_final_block = is_final_block
        # Outer-product-mean chunking stays off by default (its chunked einsum is not bit-exact under
        # bf16); enable it explicitly via ``set_chunk_size`` / the model ``chunk_size`` forward argument.
        self.outer_product_mean = EsmFold2OuterProductMean(
            d_msa, d_hidden, d_pair, divide_outer_before_proj=config.msa_encoder_divide_outer_before_proj
        )
        if not is_final_block:
            self.msa_pair_weighted_averaging = EsmFold2MSAPairWeightedAveraging(
                d_msa, d_pair, n_heads_msa, msa_head_width
            )
            self.msa_transition = EsmFold2Transition(
                d_msa, config.msa_encoder_transition_intermediate_size, chunk_size=config.chunk_size
            )
        self.tri_mul_out = EsmFold2TriangleMultiplicativeUpdate(
            dim=d_pair, outgoing=True, chunk_size=config.chunk_size
        )
        self.tri_mul_in = EsmFold2TriangleMultiplicativeUpdate(
            dim=d_pair, outgoing=False, chunk_size=config.chunk_size
        )
        self.pair_transition = EsmFold2Transition(
            d_pair, config.pair_transition_intermediate_size, chunk_size=config.chunk_size
        )

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.outer_product_mean.set_chunk_size(chunk_size)
        self.tri_mul_out.set_chunk_size(chunk_size)
        self.tri_mul_in.set_chunk_size(chunk_size)
        if not self.is_final_block:
            self.msa_transition.set_chunk_size(chunk_size)
        self.pair_transition.set_chunk_size(chunk_size)

    def forward(
        self,
        msa_repr: Tensor,
        pair: Tensor,
        msa_attention_mask: Tensor,
        pair_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pair = pair + self.outer_product_mean(msa_repr, msa_attention_mask)
        if not self.is_final_block:
            msa_repr = msa_repr + self.msa_pair_weighted_averaging(msa_repr, pair, pair_attention_mask)
            msa_repr = self.msa_transition(msa_repr)
        pair = pair + self.tri_mul_out(pair, visibility=pair_attention_mask)
        pair = pair + self.tri_mul_in(pair, visibility=pair_attention_mask)
        pair = self.pair_transition(pair)
        return msa_repr, pair


class EsmFold2MSAEncoder(nn.Module):
    """Stack of [`EsmFold2MSAEncoderBlock`] layers that conditions the pair on an MSA."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        d_msa = config.msa_encoder_hidden_size
        d_inputs = config.single_inputs_size
        n_layers = config.msa_encoder_num_hidden_layers
        # num_res_types one-hot + has_deletion + deletion_value.
        self.embed = nn.Linear(config.num_res_types + 2, d_msa, bias=False)
        self.project_inputs = nn.Linear(d_inputs, d_msa, bias=False)
        self.layers = nn.ModuleList(
            [EsmFold2MSAEncoderBlock(config, is_final_block=(i == n_layers - 1)) for i in range(n_layers)]
        )

    def set_chunk_size(self, chunk_size: int | None) -> None:
        for layer in self.layers:
            layer.set_chunk_size(chunk_size)

    def forward(
        self,
        x_pair: Tensor,
        single_inputs: Tensor,
        msa_oh: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        msa_attention_mask: Tensor,
    ) -> Tensor:
        # All inputs are pre-transposed to [B, L, M, ...] before calling.
        msa_feat = torch.cat([msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1)
        msa_repr = self.embed(msa_feat.to(self.embed.weight.dtype)) + self.project_inputs(single_inputs).unsqueeze(2)
        tok_mask = msa_attention_mask[:, :, 0].bool()
        pair_attention_mask = tok_mask.unsqueeze(2) & tok_mask.unsqueeze(1)
        for layer in self.layers:
            msa_repr, x_pair = layer(msa_repr, x_pair, msa_attention_mask, pair_attention_mask)
        return x_pair


@auto_docstring
class EsmFold2PreTrainedModel(PreTrainedModel):
    config_class = EsmFold2Config
    base_model_prefix = "esmfold2"
    main_input_name = "token_index"
    _no_split_modules = [
        "EsmcLayer",
        "EsmFold2PairUpdateBlock",
        "EsmFold2AtomEncoder",
        "EsmFold2AtomDecoder",
        "EsmFold2DiffusionTransformer",
    ]
    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$"]
    # The Fourier noise-embedding frequencies/phases are random Gaussian features whose
    # precision drives the diffusion conditioning; keep them fp32 even under dtype=bf16.
    _keep_in_fp32_modules_strict = ["fourier"]
    _supports_sdpa = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        # The non-default weight inits (adaLN-Zero gates, the parcae recurrence, zeroed output
        # projections). They live here rather than in each submodule's __init__ so post_init()
        # applies them; on a from_pretrained load the init.* helpers below no-op, leaving the
        # checkpoint weights untouched.
        super()._init_weights(module)
        if isinstance(module, EsmFold2Model):
            init.eye_(module.parcae_readout.weight)
            init.eye_(module.parcae_b_cont)
            init.zeros_(module.parcae_log_a)
            parcae_delta_init = -math.log(math.sqrt(1.0 / 5.0))
            init.constant_(module.parcae_log_delta, _inverse_softplus(parcae_delta_init))
        elif isinstance(module, EsmFold2ConfidenceHead):
            init.zeros_(module.plddt_weight)
            init.zeros_(module.resolved_weight)
        elif isinstance(module, EsmFold2AdaptiveLayerNorm):
            init.ones_(module.s_scale)
        elif isinstance(module, EsmFold2SWAAtomLayer):
            init.zeros_(module.adaln_linear.weight)
        elif isinstance(module, EsmFold2AttentionPairBias):
            if getattr(module, "out_gate", None) is not None:
                init.zeros_(module.out_gate.weight)
                init.constant_(module.out_gate.bias, -2.0)
        elif isinstance(module, EsmFold2ConditionedTransitionBlock):
            if getattr(module, "output_gate", None) is not None:
                init.zeros_(module.output_gate.weight)
                init.constant_(module.output_gate.bias, -2.0)
        elif isinstance(module, EsmFold2DiffusionModule):
            init.zeros_(module.s_to_token.weight)
        elif isinstance(module, EsmFold2LanguageModelShim):
            init.zeros_(module.base_z_combine)


@auto_docstring(
    custom_intro="""
    ESMFold2 all-atom protein structure predictor with a bundled ESMC protein-language-model backbone. This is the
    standard released ESMFold2 architecture, whose trunk is a linear-recurrent stack (internally referred to as
    "parcae").
    """
)
class EsmFold2Model(EsmFold2PreTrainedModel):
    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__(config)
        d_inputs = config.single_inputs_size
        d_pair = config.pairwise_hidden_size

        # structure_prediction=False: no coords_linear, aggregates to d_token // 2.
        self.inputs_atom_encoder = EsmFold2AtomEncoder(config, structure_prediction=False)
        self.z_init_1 = nn.Linear(d_inputs, d_pair, bias=False)
        self.z_init_2 = nn.Linear(d_inputs, d_pair, bias=False)
        self.rel_pos = EsmFold2ResIdxAsymIdSymIdEntityIdEncoding(
            n_relative_residx_bins=config.n_relative_residx_bins,
            n_relative_chain_bins=config.n_relative_chain_bins,
            d_pair=d_pair,
        )
        self.token_bonds = nn.Linear(1, d_pair, bias=False)
        self.language_model = EsmFold2LanguageModelShim(
            d_z=d_pair, d_model=config.esmc_config.d_model, num_layers=config.esmc_config.n_layers
        )
        # ESMC backbone built here with random weights (no I/O), then populated by
        # from_pretrained from the checkpoint's ``esmc.*`` weights. Frozen in effect:
        # forward detaches its hidden states before they enter the trunk.
        self.esmc = AutoModel.from_config(config.esmc_config)

        self.folding_trunk = EsmFold2FoldingTrunk(
            n_layers=config.folding_trunk_num_hidden_layers,
            d_pair=d_pair,
            intermediate_size=config.pair_transition_intermediate_size,
            chunk_size=config.chunk_size,
        )
        self.lm_encoder = EsmFold2FoldingTrunk(
            n_layers=config.lm_encoder_num_hidden_layers,
            d_pair=d_pair,
            intermediate_size=config.pair_transition_intermediate_size,
            chunk_size=config.chunk_size,
        )

        # parcae linear-recurrence params (allocated here; initialized in _init_weights):
        # log_a -> 0, log_delta -> a fixed decay constant, b_cont -> identity, readout -> identity.
        self.parcae_input_norm = EsmFold2LayerNorm(d_pair)
        self.parcae_log_a = nn.Parameter(torch.zeros(d_pair))
        self.parcae_log_delta = nn.Parameter(torch.empty(d_pair, dtype=torch.float32))
        self.parcae_b_cont = nn.Parameter(torch.empty(d_pair, d_pair))
        self.parcae_readout = nn.Linear(d_pair, d_pair, bias=False)
        self.parcae_coda = EsmFold2FoldingTrunk(
            n_layers=config.parcae_num_coda_layers,
            d_pair=d_pair,
            intermediate_size=config.pair_transition_intermediate_size,
            chunk_size=config.chunk_size,
        )

        # Heads --------------------------------------------------------------
        self.structure_head = EsmFold2DiffusionStructureHead(config)
        self.distogram_head = nn.Linear(d_pair, config.structure_head_distogram_bins, bias=True)
        self.confidence_head = EsmFold2ConfidenceHead(config)

        self.msa_encoder = EsmFold2MSAEncoder(config)

        self.post_init()

    def set_chunk_size(self, chunk_size: int | None) -> None:
        self.folding_trunk.set_chunk_size(chunk_size)
        self.lm_encoder.set_chunk_size(chunk_size)
        self.parcae_coda.set_chunk_size(chunk_size)
        self.confidence_head.set_chunk_size(chunk_size)
        self.msa_encoder.set_chunk_size(chunk_size)

    @torch.no_grad()
    def _compute_lm_hidden_states(
        self,
        input_ids: Tensor,
        asym_id: Tensor,
        residue_index: Tensor,
        mol_type: Tensor,
        tok_mask: Tensor,
    ) -> Tensor:
        """Run ESMC with BOS/EOS wrapping, return hidden states [B, L, N, D] with N=81 layers.

        Atom-tokenized modified residues (HYP, MSE, ACE, NH2, ...) span multiple
        structure tokens but share a single ``(asym_id, residue_index)`` key —
        collapse them to one LM token per residue before running the LM (the LM
        was trained on per-residue inputs, not per-atom), then scatter the
        hidden states back to the per-token layout. The frozen backbone runs
        under ``no_grad``, so no gradients are tracked (no detach needed).
        """
        B, L = input_ids.shape
        device = input_ids.device
        protein_mask = (mol_type == 0) & tok_mask

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

        # Pad to the longest LM input.
        max_len = max(lm_lengths)
        lm_input_ids = torch.full((B, max_len), 1, device=device, dtype=input_ids.dtype)  # PAD=1
        for b in range(B):
            lm_input_ids[b, : lm_lengths[b]] = lm_input_list[b]

        # sequence_id for chain-aware attention; PAD tokens get -1 (no attention).
        sequence_id = (lm_input_ids == 0).cumsum(dim=1) - 1  # BOS=0
        sequence_id = sequence_id.masked_fill(lm_input_ids == 1, -1)  # PAD=1

        # bf16 autocast scoped to the ESMC backbone (norms/softmax fp32, matmuls/rotary
        # bf16); a no-op for an fp32 backbone, and the trunk stays dtype-honest.
        use_amp = next(self.esmc.parameters()).dtype == torch.bfloat16
        with (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp),
            torch.inference_mode(),
        ):
            esmc_out = self.esmc(input_ids=lm_input_ids, sequence_id=sequence_id, output_hidden_states=True)

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

        return result

    def _discretized_dynamics(self) -> tuple[Tensor, Tensor]:
        # Discretized linear state-space dynamics for the parcae recurrence: ``state_decay`` is
        # the per-channel state transition (Ā), ``input_matrix`` the discretized input projection (B̄).
        delta = F.softplus(self.parcae_log_delta)
        state_decay = torch.exp(-delta * torch.exp(self.parcae_log_a))
        input_matrix = delta[:, None] * self.parcae_b_cont
        return state_decay, input_matrix

    def _init_pair_state(self, ref: Tensor) -> Tensor:
        std = math.sqrt(2.0 / (5.0 * ref.shape[-1]))
        state = torch.empty_like(ref, dtype=torch.float32)
        nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return state.to(dtype=ref.dtype)

    def _prepare_features(
        self,
        res_type: Tensor,
        tok_mask: Tensor,
        msa: Tensor | None,
        msa_attention_mask: Tensor | None,
        deletion_mean: Tensor | None,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        atom_attention_mask: Tensor,
        atom_to_token: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One-hot / mask the raw structural inputs into embedder-ready features.

        Returns ``(res_type_oh, profile, deletion_mean, ref_element_oh,
        ref_atom_name_chars_oh, atom_to_token)`` with ``atom_to_token`` zeroed at padding.
        """
        if res_type.dim() == 2:
            res_type_oh = F.one_hot(res_type.long(), num_classes=self.config.num_res_types).float()
            res_type_oh = res_type_oh * tok_mask.unsqueeze(-1).float()
        else:
            res_type_oh = res_type.float()

        if msa is not None:
            msa_oh_profile = F.one_hot(msa.long(), num_classes=self.config.num_res_types).float()
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

        ref_element_oh = F.one_hot(ref_element.long(), num_classes=self.config.max_atomic_number).float()
        ref_atom_name_chars_oh = F.one_hot(ref_atom_name_chars.long(), num_classes=self.config.char_vocab_size).float()
        # Bias-free downstream Linears require zeroed padding.
        atm_mask_f = atom_attention_mask.float()
        ref_element_oh = ref_element_oh * atm_mask_f.unsqueeze(-1)
        ref_atom_name_chars_oh = ref_atom_name_chars_oh * atm_mask_f.unsqueeze(-1).unsqueeze(-1)
        atom_to_token = atom_to_token * atom_attention_mask.long()

        return res_type_oh, profile, deletion_mean, ref_element_oh, ref_atom_name_chars_oh, atom_to_token

    def _build_msa_kwargs(
        self,
        msa: Tensor | None,
        msa_attention_mask: Tensor | None,
        has_deletion: Tensor | None,
        deletion_value: Tensor | None,
        tok_mask: Tensor,
        single_inputs: Tensor,
    ) -> dict | None:
        """Assemble the transposed/padded one-hot MSA tensors the MSA encoder consumes."""
        if msa is None:
            return None
        B_msa, M, L_msa = msa.shape
        msa_oh = F.one_hot(msa.permute(0, 2, 1).long(), num_classes=self.config.num_res_types).float()
        msa_attn = (
            msa_attention_mask.permute(0, 2, 1).float()
            if msa_attention_mask is not None
            else tok_mask[:, :, None].expand(-1, -1, M).float()
        )
        # Bias-free EsmFold2MSAEncoder.embed requires zeroed padding.
        msa_oh = msa_oh * msa_attn.unsqueeze(-1)
        has_deletion_t = (
            has_deletion.permute(0, 2, 1).float()
            if has_deletion is not None
            else torch.zeros(B_msa, L_msa, M, device=msa.device)
        )
        deletion_value_t = (
            deletion_value.permute(0, 2, 1).float()
            if deletion_value is not None
            else torch.zeros(B_msa, L_msa, M, device=msa.device)
        )
        return {
            "single_inputs": single_inputs,
            "msa_oh": msa_oh,
            "has_deletion": has_deletion_t,
            "deletion_value": deletion_value_t,
            "msa_attention_mask": msa_attn,
        }

    def _run_one_loop(
        self,
        z: Tensor,
        z_init: Tensor,
        lm_z: Tensor | None,
        _msa_kwargs: dict | None,
        pair_mask: Tensor,
        state_decay: Tensor,
        input_matrix: Tensor,
        total_steps: int,
    ) -> Tensor:
        # Helper method (not inline) so per-iter L²×c_z locals free on return (else
        # ~2 GB leaks into distogram/sample scope). training=True forces the per-loop
        # dropout under eval().
        _lm_dropout_p = self.config.lm_encoder_lm_dropout
        _per_loop_lm_dropout = lm_z is not None and self.config.lm_encoder_per_loop_lm_dropout and _lm_dropout_p > 0.0

        for _ in range(total_steps):
            if _per_loop_lm_dropout:
                assert lm_z is not None  # narrowed by _per_loop_lm_dropout
                lm_z_i: Tensor | None = F.dropout(lm_z, p=_lm_dropout_p, training=True)
            else:
                lm_z_i = lm_z

            refined_lm_z: Tensor | None = None
            if lm_z_i is not None:
                refined_lm_z = self.lm_encoder(lm_z_i.to(z_init.dtype), pair_attention_mask=pair_mask)

            z_inject_pair = z_init
            if _msa_kwargs is not None:
                msa_pair = self.msa_encoder(x_pair=z_inject_pair, **_msa_kwargs).to(z_inject_pair.dtype)
                z_inject_pair = msa_pair if self.config.msa_encoder_overwrite else (z_inject_pair + msa_pair)

            if refined_lm_z is not None:
                z_inject_pair = z_inject_pair + refined_lm_z.to(z_inject_pair.dtype)

            injected_pair = self.parcae_input_norm(z_inject_pair)
            z = state_decay * z + F.linear(injected_pair.to(z.dtype), input_matrix)
            z = self.folding_trunk(z, pair_attention_mask=pair_mask)

        return z

    @auto_docstring
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
        chunk_size: int | None = None,
        **kwargs,
    ) -> EsmFold2Output:
        r"""
        token_index (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Per-token positional index within the full complex; feeds the relative-position encoding.
        residue_index (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Residue index within each chain; feeds the relative-position encoding.
        asym_id (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Asymmetric-unit (chain) ID for each token.
        sym_id (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Symmetry-copy ID distinguishing identical chains of a homomer.
        entity_id (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Entity ID grouping tokens that belong to the same molecular entity.
        mol_type (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Molecule-type code for each token (``0`` = protein).
        res_type (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Residue-type (amino-acid identity) index for each token.
        token_bonds (`torch.Tensor` of shape `(batch_size, num_tokens, num_tokens, 1)`):
            Pairwise inter-token covalent-bond feature.
        token_attention_mask (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Mask marking valid tokens (``1``) versus padding (``0``). Inputs must be right-padded.
        ref_pos (`torch.Tensor` of shape `(batch_size, num_atoms, 3)`):
            Reference-conformer Cartesian coordinates for each atom.
        ref_element (`torch.Tensor` of shape `(batch_size, num_atoms)`):
            Atomic number of each atom.
        ref_charge (`torch.Tensor` of shape `(batch_size, num_atoms)`):
            Formal charge of each atom.
        ref_atom_name_chars (`torch.Tensor` of shape `(batch_size, num_atoms, 4)`):
            Encoded four-character atom name for each atom.
        ref_space_uid (`torch.Tensor` of shape `(batch_size, num_atoms)`):
            Per-atom group ID (the atom's token index), used by the atom-encoder 3D RoPE.
        atom_attention_mask (`torch.Tensor` of shape `(batch_size, num_atoms)`):
            Mask marking valid atoms (``1``) versus padding (``0``).
        atom_to_token (`torch.Tensor` of shape `(batch_size, num_atoms)`):
            Index of the token each atom belongs to (a token's atoms are contiguous).
        distogram_atom_idx (`torch.Tensor` of shape `(batch_size, num_tokens)`):
            Index of the representative atom (Cβ, or Cα for glycine) of each token, used by the distogram head.
        deletion_mean (`torch.Tensor` of shape `(batch_size, num_tokens)`, *optional*):
            Mean MSA deletion count per column. Defaults to zeros (no MSA).
        msa (`torch.Tensor` of shape `(batch_size, msa_depth, num_tokens)`, *optional*):
            MSA residue-type tokens (row 0 is the query sequence). Defaults to a single-sequence MSA.
        has_deletion (`torch.Tensor` of shape `(batch_size, msa_depth, num_tokens)`, *optional*):
            Boolean flag marking MSA positions preceded by a deletion.
        deletion_value (`torch.Tensor` of shape `(batch_size, msa_depth, num_tokens)`, *optional*):
            Per-position MSA deletion counts.
        msa_attention_mask (`torch.Tensor` of shape `(batch_size, msa_depth, num_tokens)`, *optional*):
            Validity mask for the MSA rows/columns.
        input_ids (`torch.Tensor` of shape `(batch_size, num_tokens)`, *optional*):
            ESMC-vocabulary token ids for the sequence. Fed to the bundled ESMC backbone to produce
            `lm_hidden_states` when those are not passed directly; ignored when `lm_hidden_states` is given.
        lm_hidden_states (`torch.Tensor` of shape `(batch_size, num_tokens, hidden_size)`, *optional*):
            Precomputed ESMC backbone hidden states. When provided, the backbone is not run and `input_ids`
            is unused.
        num_loops (`int`, *optional*):
            Number of trunk refinement loops. Defaults to `config.num_loops`.
        num_diffusion_samples (`int`, *optional*):
            Number of parallel structure samples to draw; the confidence head re-runs once per sample.
            Defaults to `config.num_diffusion_samples`.
        num_sampling_steps (`int`, *optional*):
            Number of diffusion sampling steps. Defaults to `config.structure_head_inference_num_steps`.
        chunk_size (`int`, *optional*):
            Override for `config.chunk_size`, the chunk size used by the memory-heavy pair-/MSA-stream ops.
            When given, it is applied to the trunk/encoders for this and subsequent calls (`None` keeps the
            configured value). Lower it to save memory on long sequences; pass `0` to disable chunking.
        """
        if chunk_size is not None:
            # Persisted onto the chunking submodules (mirrors the reference set_chunk_size knob); 0 disables.
            self.set_chunk_size(chunk_size or None)

        tok_mask = token_attention_mask
        atm_mask = atom_attention_mask
        disto_idx = distogram_atom_idx

        n_loops: int = num_loops if num_loops is not None else self.config.num_loops
        n_samples: int = (
            num_diffusion_samples if num_diffusion_samples is not None else self.config.num_diffusion_samples
        )
        total_steps = max(1, n_loops + 1)

        res_type_oh, profile, deletion_mean, ref_element_oh, ref_atom_name_chars_oh, atom_to_token = (
            self._prepare_features(
                res_type=res_type,
                tok_mask=tok_mask,
                msa=msa,
                msa_attention_mask=msa_attention_mask,
                deletion_mean=deletion_mean,
                ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars,
                atom_attention_mask=atm_mask,
                atom_to_token=atom_to_token,
            )
        )

        atom_inputs = EsmFold2AtomInputs(
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            atom_attention_mask=atm_mask,
            ref_element=ref_element_oh,
            ref_atom_name_chars=ref_atom_name_chars_oh,
            ref_space_uid=ref_space_uid,
            atom_to_token=atom_to_token,
        )

        atom_encoding, _q, _c, _attn = self.inputs_atom_encoder(atom_inputs)
        # The continuous input features are fp32; fold them into the atom encoding's
        # (compute) dtype so the single representation is one dtype.
        dtype = atom_encoding.dtype
        single_inputs = torch.cat(
            [
                atom_encoding,
                res_type_oh.to(dtype),
                profile.float().to(dtype),
                deletion_mean.float().unsqueeze(-1).to(dtype),
            ],
            dim=-1,
        )

        z_init = self.z_init_1(single_inputs).unsqueeze(2) + self.z_init_2(single_inputs).unsqueeze(1)

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
            lm_z = self.language_model(lm_hidden_states)
        del lm_hidden_states

        pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()

        z = self._init_pair_state(z_init)

        state_decay, input_matrix = self._discretized_dynamics()
        state_decay = state_decay.view(1, 1, 1, -1).to(device=z.device, dtype=z.dtype)
        input_matrix = input_matrix.to(device=z.device, dtype=z.dtype)

        _msa_kwargs = self._build_msa_kwargs(
            msa=msa,
            msa_attention_mask=msa_attention_mask,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            tok_mask=tok_mask,
            single_inputs=single_inputs,
        )

        z = self._run_one_loop(
            z=z,
            z_init=z_init,
            lm_z=lm_z,
            _msa_kwargs=_msa_kwargs,
            pair_mask=pair_mask,
            state_decay=state_decay,
            input_matrix=input_matrix,
            total_steps=total_steps,
        )
        del z_init, lm_z, _msa_kwargs, state_decay, input_matrix

        z = self.parcae_readout(z)
        z = self.parcae_coda(z, pair_attention_mask=pair_mask)

        z = z.float()
        distogram_logits = self.distogram_head((z + z.transpose(-2, -3)).to(self.distogram_head.weight.dtype))

        structure_output = self.structure_head.sample(
            pair_trunk=z,
            single_inputs=single_inputs,
            relative_position_encoding=relative_position_encoding,
            atom_inputs=atom_inputs,
            token_attention_mask=tok_mask,
            num_diffusion_samples=n_samples,
            num_sampling_steps=num_sampling_steps,
        )

        sample_coords = structure_output["sample_atom_coords"]
        if sample_coords is None:
            raise RuntimeError("The diffusion structure head did not return sampled coordinates.")

        confidence_output = self.confidence_head(
            single_inputs=single_inputs.detach(),
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

        return EsmFold2Output(
            distogram_logits=distogram_logits,
            sample_atom_coords=sample_coords,
            **confidence_output,
        )

    @torch.no_grad()
    def infer_protein(self, seq: str, **forward_kwargs) -> EsmFold2Output:
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
    def output_to_pdb(output: EsmFold2Output, features: dict[str, Tensor]) -> str:
        """Render a PDB string from an [`EsmFold2Output`] and the input ``features`` it was produced from."""
        from .protein_utils import output_to_pdb as _output_to_pdb

        return _output_to_pdb(output, features)


__all__ = ["EsmFold2Model", "EsmFold2PreTrainedModel"]
