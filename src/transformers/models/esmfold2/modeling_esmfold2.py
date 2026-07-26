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
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ... import initialization as init
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_bidirectional_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..auto import AutoModel
from .configuration_esmfold2 import EsmFold2Config
from .generation_esmfold2 import EsmFold2GenerationMixin


@dataclass
class EsmFold2AtomInputs:
    """Featurized reference-conformer atom inputs, bundled so the atom stack takes one argument
    instead of seven.

    These are the raw `fold` inputs after `EsmFold2GenerationMixin._prepare_features` has one-hot
    encoded the categorical fields and zeroed the padding, so some shapes differ from the raw inputs.
    Consumed by the inputs embedder, the diffusion atom encoder/decoder, and the sampler.

    Args:
        ref_pos (`torch.Tensor` of shape `(batch_size, num_atoms, 3)`):
            Reference-conformer Cartesian coordinates for each atom.
        ref_charge (`torch.Tensor` of shape `(batch_size, num_atoms)`):
            Formal charge of each atom.
        atom_attention_mask (`torch.Tensor` of shape `(batch_size, num_atoms)`):
            Mask marking valid atoms (`1`) versus padding (`0`).
        ref_element (`torch.Tensor` of shape `(batch_size, num_atoms, max_atomic_number)`):
            One-hot atomic number of each atom, zeroed at padding.
        ref_atom_name_chars (`torch.Tensor` of shape `(batch_size, num_atoms, max_chars, char_vocab_size)`):
            One-hot encoding of each atom's four-character name, zeroed at padding. The atom encoder
            flattens the last two axes into its feature vector.
        ref_space_uid (`torch.Tensor` of shape `(batch_size, num_atoms)`):
            Per-atom group ID (the atom's token index), used by the atom-encoder 3D RoPE.
        atom_to_token (`torch.Tensor` of shape `(batch_size, num_atoms)`):
            Index of the token each atom belongs to, zeroed at padding. A token's atoms are contiguous.
    """

    ref_pos: Tensor
    ref_charge: Tensor
    atom_attention_mask: Tensor
    ref_element: Tensor
    ref_atom_name_chars: Tensor
    ref_space_uid: Tensor
    atom_to_token: Tensor


@dataclass
class EsmFold2DiffusionStepInvariants:
    """Denoiser conditioning that depends on neither the noise level nor the noisy coordinates.

    Built once per fold by ``EsmFold2DiffusionModule.prepare_step_invariants`` and read by every
    denoising step. Everything here is derived from the trunk outputs and the frozen weights, so
    hoisting it out of the module forwards is bit-identical to recomputing it each step.

    ``prepare_step_invariants`` is also where the batch is expanded from ``batch_size`` to
    ``batch_size * num_diffusion_samples``: each field below is already expanded, and every module
    downstream of it works on the expanded batch and never sees ``num_diffusion_samples``. The
    expansion happens on these *outputs* rather than on the raw inputs so that the per-atom
    featurization and the pair-bias projection still run once at the unexpanded batch size.
    """

    # Atom stack.
    c_base: Tensor
    attention_mask: Tensor
    position_embeddings: tuple[Tensor, Tensor]
    atom_mask: Tensor
    atom_to_token: Tensor
    n_tokens: int
    # Token stack.
    single_repr_inputs: Tensor
    token_pair_bias: list[Tensor]


class EsmFold2LayerNorm(nn.LayerNorm):
    """LayerNorm that always computes in fp32.

    The weight/bias are pinned to fp32 (see ``_keep_in_fp32_modules_strict``), so they stay full
    precision even under a bf16 load. The bf16 activation is upcast to compute the norm in fp32 and
    the result is cast back to the input dtype for the downstream (bf16) ops.
    """

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.layer_norm(hidden_states.float(), self.normalized_shape, self.weight, self.bias, self.eps).to(
            hidden_states.dtype
        )


class EsmFold2RMSNorm(torch.nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x)

    def extra_repr(self):
        return f"eps={self.eps}"


class EsmFold2TransitionLayer(nn.Module):
    """Norm + SwiGLU transition (no residual; the caller adds it).

    Reuses the single ``EsmFold2SwiGLU`` feed-forward op so there is one SwiGLU implementation across
    the model: the gate/up projections are the fused ``ffn.gate_up_proj`` and the output is ``ffn.down_proj``.
    """

    def __init__(self, d_model: int, n: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = EsmFold2LayerNorm(d_model, eps=eps)
        self.ffn = EsmFold2SwiGLU(d_model, n * d_model)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.ffn(self.norm(hidden_states))


class EsmFold2AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalization (adaLN-Zero)."""

    def __init__(self, config: EsmFold2Config, eps: float = 1e-5) -> None:
        super().__init__()
        # adaLN-Zero is diffusion-token-width in both call sites (attention pair-bias + conditioned transition).
        self.d_model = config.structure_head.diffusion_module.token_hidden_size
        self.d_cond = config.structure_head.diffusion_module.token_hidden_size
        self.eps = eps
        self.norm_scale = nn.Parameter(
            torch.empty(self.d_cond)
        )  # LayerNorm scale for the conditioning; ones-init in _init_weights
        self.gate_proj = nn.Linear(self.d_cond, self.d_model, bias=True)
        self.shift_proj = nn.Linear(self.d_cond, self.d_model, bias=False)

    def forward(self, hidden_states: Tensor, single_repr: Tensor) -> Tensor:
        normed = F.layer_norm(hidden_states.float(), (self.d_model,), None, None, self.eps)
        cond = F.layer_norm(single_repr.float(), (self.d_cond,), self.norm_scale, None, self.eps).to(single_repr.dtype)
        # gate/shift in bf16; ``normed`` is fp32 so the affine promotes to fp32, then
        # downcast for the next op.
        gate = torch.sigmoid(self.gate_proj(cond))
        shift = self.shift_proj(cond)
        return (gate * normed + shift).to(hidden_states.dtype)


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
        t = t_hat.to(dtype=self.frequencies.dtype).reshape(-1)
        return torch.cos(2.0 * torch.pi * (t[:, None] * self.frequencies[None, :] + self.phases[None, :]))


class EsmFold2SwiGLU(nn.Module):
    """SwiGLU feed-forward with a fused gate+up projection (``gate_up_proj``) and output ``down_proj``.

    ``intermediate_size`` is supplied by the caller and registered on the config, so every
    ESMFold2 SwiGLU feed-forward is this one module regardless of how its width is derived.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.intermediate_size = intermediate_size

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate_up = self.gate_up_proj(hidden_states)
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


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    The atom stack keeps heads at dim 2 (``[batch, atoms, heads, head_dim]``) and
    :class:`EsmFold2RotaryEmbedding3D` builds ``cos``/``sin`` at the full head dim over 3D
    coordinates plus a space UID, so it calls this with ``unsqueeze_dim=2``.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _swa_window_mask_function(valid: Tensor, half_window: int) -> Callable:
    """Sliding-window ``and`` mask over atom index.

    An atom attends to another iff their indices differ by at most ``half_window`` (inputs are
    right-padded, so index distance is the window distance).

    Padding is masked here rather than through ``create_bidirectional_mask``'s ``attention_mask``
    argument because that one masks *keys* only (see ``padding_mask_function``). The atom stack needs
    padded *queries* masked too, so that a fully padded row is all-masked and resolves to zeros via
    ``torch.nan_to_num`` in the attention forward, instead of emitting values that would then have to
    be cleaned up downstream.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        within = abs(q_idx - kv_idx) <= half_window
        return within & valid[batch_idx, q_idx] & valid[batch_idx, kv_idx]

    return inner_mask


class EsmFold2SWA3DRoPEAttention(nn.Module):
    """Sliding window self-attention with 3D RoPE. Has q/k/v/gate/out projections.

    The plain ``softmax(QKᵀ)V`` core is dispatched through the v5 attention
    interface (``config._attn_implementation``: ``eager`` / ``sdpa`` / ...), with
    the sliding window expressed as an additive attention mask. The shared
    ``config`` is passed in at construction, so ``set_attn_implementation()`` stays
    live (it mutates the same object); dims come from the atom sub-config of this call site.
    """

    def __init__(self, config: EsmFold2Config, structure_prediction: bool = True) -> None:
        super().__init__()
        atom_config = (
            config.structure_head.diffusion_module.atom_encoder if structure_prediction else config.atom_encoder
        )
        d_model = atom_config.hidden_size
        self.config = config
        self.n_heads = atom_config.num_attention_heads
        self.head_dim = atom_config.head_dim
        self.scale = self.head_dim**-0.5
        # No grouped-query attention; identity repeat keeps the interface happy.
        self.num_key_value_groups = 1
        # Bidirectional encoder: never let the sdpa/flash interface default to
        # causal masking when attention_mask happens to be None.
        self.is_causal = False

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = EsmFold2RMSNorm()
        self.k_norm = EsmFold2RMSNorm()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tensor:
        B, N = hidden_states.shape[:2]
        cos, sin = position_embeddings

        q = self.q_proj(hidden_states).view(B, N, self.n_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, N, self.n_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, N, self.n_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        out, _ = attention_interface(
            self,
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attention_mask,
            dropout=0.0,
            scaling=self.scale,
            **kwargs,
        )
        # Fully padded query rows are all-masked -> NaN in the softmax; zero them (their outputs are
        # dropped downstream by scatter_atom_to_token regardless). Replaces the old post-hoc valid mul.
        out = torch.nan_to_num(out)

        out = out.reshape(B, N, -1)
        out = out * torch.sigmoid(self.gate_proj(hidden_states))
        return self.o_proj(out)


class EsmFold2SWAAtomLayer(nn.Module):
    """adaLN-Zero + SWA attention + EsmFold2SwiGLU FFN.

    The adaLN-Zero modulation is ``adaln_linear`` applied to ``silu(atom_cond)`` (zero-init gate).
    """

    def __init__(self, config: EsmFold2Config, structure_prediction: bool = True) -> None:
        super().__init__()
        atom_config = (
            config.structure_head.diffusion_module.atom_encoder if structure_prediction else config.atom_encoder
        )
        d_atom = atom_config.hidden_size
        ffn_intermediate_size = atom_config.ffn_intermediate_size
        # adaln-Zero gate; zero-init lives in EsmFold2PreTrainedModel._init_weights.
        self.adaln_linear = nn.Linear(d_atom, 6 * d_atom, bias=False)

        self.attn = EsmFold2SWA3DRoPEAttention(config, structure_prediction)
        self.ffn = EsmFold2SwiGLU(d_atom, ffn_intermediate_size)
        self.attn_norm = EsmFold2RMSNorm()
        self.ffn_norm = EsmFold2RMSNorm()

    def forward(
        self,
        hidden_states: Tensor,
        atom_cond: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        modulation = self.adaln_linear(F.silu(atom_cond))
        if modulation.dim() == 2:
            modulation = modulation.unsqueeze(1)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.chunk(6, dim=-1)

        attn_input = self.attn_norm(hidden_states) * (1 + scale_a) + shift_a
        attn_out = self.attn(attn_input, attention_mask, position_embeddings)
        hidden_states = hidden_states + gate_a * attn_out

        ffn_input = self.ffn_norm(hidden_states) * (1 + scale_f) + shift_f
        ffn_out = self.ffn(ffn_input)
        hidden_states = hidden_states + gate_f * ffn_out
        return hidden_states


class EsmFold2RotaryEmbedding3D(nn.Module):
    """Rotary embedding over continuous 3D atom coordinates plus a discrete space UID.

    Unlike sequence RoPE, this encodes physical position (``ref_pos`` = x/y/z) and a
    per-atom space UID rather than token indices, with separate spatial and UID base
    frequencies. ``forward`` returns cos/sin already at the full head dim, so the caller
    applies plain rotate-half RoPE (:func:`apply_rotary_pos_emb`). The inverse frequencies
    are cheap to rebuild each call and are computed on the input device to stay bit-exact.
    """

    def __init__(self, config: EsmFold2Config, structure_prediction: bool = True) -> None:
        super().__init__()
        atom_config = (
            config.structure_head.diffusion_module.atom_encoder if structure_prediction else config.atom_encoder
        )
        self.head_dim = atom_config.head_dim
        self.n_spatial_per_axis = atom_config.n_spatial_rope_pairs_per_axis
        self.n_uid_pairs = atom_config.n_uid_rope_pairs
        self.spatial_base_freq = atom_config.spatial_rope_base_frequency
        self.uid_base_freq = atom_config.uid_rope_base_frequency

    def forward(self, ref_pos: Tensor, ref_space_uid: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
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

        # Duplicate to the full head dim so the caller applies standard rotate-half RoPE. The angles are
        # built in fp32 and returned at the activation dtype (the caller passes it), so ``cos``/``sin``
        # always match the queries they rotate -- the same contract as the sequence-RoPE modules.
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


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
    batch_size, _, hidden_dim = atom_features.shape
    idx = atom_to_token_idx
    if atom_mask is not None:
        idx = idx.masked_fill(~atom_mask, n_tokens)
    out = atom_features.new_zeros(batch_size, n_tokens + int(atom_mask is not None), hidden_dim)
    out.scatter_reduce_(
        dim=1,
        index=idx.unsqueeze(-1).expand_as(atom_features),
        src=atom_features,
        reduce="mean",
        include_self=False,
    )
    return out[:, :n_tokens]


class EsmFold2AtomEncoder(nn.Module):
    """SWA atom encoder with atom_linear, atom_norm, atom_to_token_linear, [coords_linear], rotary_emb, blocks.

    ``structure_prediction=True`` (diffusion module) adds ``coords_linear``. All dims
    come from the atom sub-config of this call site, including the atom->token
    aggregation width ``output_dim``.
    """

    def __init__(self, config: EsmFold2Config, structure_prediction: bool = True) -> None:
        super().__init__()
        atom_config = (
            config.structure_head.diffusion_module.atom_encoder if structure_prediction else config.atom_encoder
        )
        d_atom = atom_config.hidden_size
        n_blocks = atom_config.num_hidden_layers
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

        self.atom_to_token_linear = nn.Linear(d_atom, atom_config.output_dim, bias=False)

    def embed_atoms(self, atom_inputs: EsmFold2AtomInputs) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Per-atom base embedding and 3D-RoPE position embeddings.

        Depends only on the reference conformer, so the diffusion stack computes this once per fold
        (before the batch is expanded across diffusion samples) rather than once per denoising step.
        """
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
        # ``atom_feats`` is fp32 (one-hots and masks), so the downcast into the projection is real;
        # ``atom_norm`` already computes in fp32 internally and returns its input dtype.
        c_base = self.atom_norm(self.atom_linear(atom_feats.to(self.atom_linear.weight.dtype)))
        return c_base, self.rotary_emb(ref_pos, atom_inputs.ref_space_uid, dtype=c_base.dtype)

    def build_attention_mask(self, atom_mask: Tensor, position_embeddings: tuple[Tensor, Tensor]) -> Tensor:
        """Sliding-window attention mask for the atom stack.

        Depends only on the (step- and layer-invariant) valid-atom mask and the window, so it is built
        once per fold rather than in every atom-stack layer's forward. ``cos`` (bf16, same batch/seq as
        the attention queries) supplies the mask metadata (dtype/device/shape). ``attention_mask=None``:
        padding is folded into the ``and`` mask, which masks padded queries as well as padded keys
        (see :func:`_swa_window_mask_function`).
        """
        return create_bidirectional_mask(
            config=self.config,
            inputs_embeds=position_embeddings[0],
            attention_mask=None,
            and_mask_function=_swa_window_mask_function(atom_mask.bool(), self.config.sliding_window // 2),
        )

    def forward(
        self,
        c_base: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        atom_mask: Tensor,
        atom_to_token: Tensor,
        n_tokens: int,
        atom_coords: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (token_acts, atom_queries, atom_cond).

        Every argument is already at the caller's batch size — for the diffusion stack that is
        ``batch_size * num_diffusion_samples``, expanded once by
        :meth:`EsmFold2DiffusionModule.prepare_step_invariants`.
        """
        atom_cond = c_base
        atom_queries = atom_cond

        if self.structure_prediction:
            # The second coord slot (a predicted-coords channel in the research model) is unused in
            # this release — always zeros — so coords_linear sees [atom_coords, 0].
            coord_input = torch.cat([atom_coords, torch.zeros_like(atom_coords)], dim=-1)
            coords_to_queries = self.coords_linear(coord_input.to(self.coords_linear.weight.dtype))
            atom_queries = atom_queries + coords_to_queries

        for layer in self.layers:
            atom_queries = layer(atom_queries, atom_cond, attention_mask, position_embeddings)

        queries_to_acts = F.relu(self.atom_to_token_linear(atom_queries))
        token_acts = scatter_atom_to_token(
            queries_to_acts,
            atom_to_token,
            n_tokens,
            atom_mask=atom_mask.bool(),
        )

        return token_acts, atom_queries, atom_cond


def _gather_along_dim1(source: Tensor, index: Tensor) -> Tensor:
    """Gather ``source`` (``[B, N, d]``) along dim 1 with a ``[B, M]`` index, returning ``[B, M, d]``."""
    idx = index.unsqueeze(-1).expand(-1, -1, source.size(-1))
    return torch.gather(source, 1, idx)


class EsmFold2AtomDecoder(nn.Module):
    """SWA atom decoder with token_to_atom_linear, blocks, norm, output_linear.

    Only used inside the diffusion module, so its atom dims are always the
    structure-prediction (diffusion) ones.
    """

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        atom_config = config.structure_head.diffusion_module.atom_encoder
        d_atom = atom_config.hidden_size
        n_blocks = atom_config.num_hidden_layers
        # The decoder only exists inside the diffusion module, so the token-stream width it projects
        # from is its parent's -- the single owner of that value.
        self.token_to_atom_linear = nn.Linear(
            config.structure_head.diffusion_module.token_hidden_size, d_atom, bias=False
        )

        self.layers = nn.ModuleList([EsmFold2SWAAtomLayer(config, structure_prediction=True) for _ in range(n_blocks)])

        self.norm = EsmFold2LayerNorm(d_atom)
        self.output_linear = nn.Linear(d_atom, 3, bias=False)  # (x, y, z) coordinates

    def forward(
        self,
        token_acts: Tensor,
        atom_queries: Tensor,
        atom_cond: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        atom_to_token: Tensor,
    ) -> Tensor:
        """Returns coord_update."""
        a_to_q = self.token_to_atom_linear(token_acts)
        a_to_q = _gather_along_dim1(a_to_q, atom_to_token)
        atom_queries = atom_queries + a_to_q

        for layer in self.layers:
            atom_queries = layer(atom_queries, atom_cond, attention_mask, position_embeddings)

        atom_coords = self.output_linear(self.norm(atom_queries))
        return atom_coords


class EsmFold2AttentionPairBias(nn.Module):
    """Gated multi-head attention with pair bias conditioning."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.config = config
        d_model = config.structure_head.diffusion_module.token_hidden_size
        d_pair = config.pairwise_hidden_size  # the trunk pair rep flows in at this width
        num_heads = config.structure_head.diffusion_module.token_num_heads
        d_cond = config.structure_head.diffusion_module.token_hidden_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        # No grouped-query attention; identity repeat keeps the attention interface happy.
        self.num_key_value_groups = 1
        self.is_causal = False

        self.adaln = EsmFold2AdaptiveLayerNorm(config)
        self.out_gate = nn.Linear(d_cond, d_model, bias=True)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.pair_norm = EsmFold2LayerNorm(d_pair)
        self.pair_bias_proj = nn.Linear(d_pair, num_heads, bias=False)

    def compute_pair_bias(self, pair_repr: Tensor) -> Tensor:
        """Project the (normed) pair representation to per-head attention biases.

        Depends only on ``pair_repr`` and this block's fixed weights, so the sampler computes it once
        per fold and reuses it across denoising steps. Called before the batch is expanded across
        diffusion samples, so the norm and projection see ``pairwise_hidden_size`` channels at the
        unexpanded batch size and only the ``num_heads``-wide result is expanded.
        """
        return self.pair_bias_proj(self.pair_norm(pair_repr))

    def forward(
        self,
        token_acts: Tensor,
        single_repr: Tensor,
        pair_bias: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        bsz, n_queries, d_model = token_acts.shape

        hidden_states = self.adaln(token_acts, single_repr)

        n_keys = hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(bsz, n_queries, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, n_keys, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, n_keys, self.num_heads, self.head_dim)

        gate = torch.sigmoid(self.g_proj(hidden_states)).view(bsz, n_queries, self.num_heads, self.head_dim)

        # ``pair_bias`` is step-invariant, so the sampler precomputes it once per block
        # (see :meth:`compute_pair_bias`) and hands it in.
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
        out = self.o_proj(context.reshape(bsz, n_queries, d_model))
        out = torch.sigmoid(self.out_gate(single_repr)) * out
        return out


class EsmFold2ConditionedTransitionBlock(nn.Module):
    """Conditioned EsmFold2SwiGLU transition with adaptive layer norm."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        d_model = config.structure_head.diffusion_module.token_hidden_size
        d_cond = config.structure_head.diffusion_module.token_hidden_size
        intermediate_size = config.structure_head.diffusion_module.token_transition_intermediate_size

        self.adaln = EsmFold2AdaptiveLayerNorm(config)
        self.output_gate = nn.Linear(d_cond, d_model, bias=True)

        self.ffn = EsmFold2SwiGLU(d_model, intermediate_size)

    def forward(self, token_acts: Tensor, single_repr: Tensor) -> Tensor:
        hidden_states = self.adaln(token_acts, single_repr)
        out = self.ffn(hidden_states)
        return torch.sigmoid(self.output_gate(single_repr)) * out


class EsmFold2DiffusionTransformer(nn.Module):
    """Diffusion denoising transformer with attention pair bias."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        num_blocks = config.structure_head.diffusion_module.token_num_blocks
        self.attn_blocks = nn.ModuleList([EsmFold2AttentionPairBias(config) for _ in range(num_blocks)])
        self.transition_blocks = nn.ModuleList([EsmFold2ConditionedTransitionBlock(config) for _ in range(num_blocks)])

    def compute_pair_biases(self, pair_repr: Tensor) -> list[Tensor]:
        """Per-block attention pair biases. Each depends only on the (step-invariant) conditioning pair
        repr and this block's fixed weights, so the sampler builds them once per fold."""
        return [attn.compute_pair_bias(pair_repr) for attn in self.attn_blocks]

    def forward(
        self,
        token_acts: Tensor,
        single_repr: Tensor,
        pair_biases: list[Tensor],
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden_states = token_acts
        for attn, transition, pair_bias in zip(self.attn_blocks, self.transition_blocks, pair_biases):
            hidden_states = hidden_states + attn(
                hidden_states,
                single_repr,
                pair_bias,
                attention_mask=attention_mask,
            )
            hidden_states = hidden_states + transition(hidden_states, single_repr)
        return hidden_states


class EsmFold2DiffusionConditioning(nn.Module):
    """Conditions pair and single representations on noise timestep."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        # The conditioning's pair/single-inputs widths are the parent's pairwise_hidden_size /
        # single_inputs_size (the trunk pair rep and the embedder single-inputs tensor flow straight
        # into the norms below); the conditioning's single output is sized to the diffusion token width.
        c_z = config.pairwise_hidden_size
        c_s = config.structure_head.diffusion_module.token_hidden_size
        c_s_inputs = config.single_inputs_size
        fourier_dim = config.structure_head.diffusion_module.fourier_dim
        transition_multiplier = config.structure_head.diffusion_module.transition_multiplier
        # The norms/transitions use their default eps (1e-5), as before.
        self.sigma_data = config.structure_head.diffusion_module.sigma_data

        self.z_input_norm = EsmFold2LayerNorm(2 * c_z)
        self.z_proj = nn.Linear(2 * c_z, c_z, bias=False)
        self.z_transitions = nn.ModuleList([EsmFold2TransitionLayer(c_z, n=transition_multiplier) for _ in range(2)])

        self.s_input_norm = EsmFold2LayerNorm(c_s_inputs)
        self.s_proj = nn.Linear(c_s_inputs, c_s, bias=False)
        self.fourier = EsmFold2FourierEmbedding(fourier_dim)
        self.noise_norm = EsmFold2LayerNorm(fourier_dim)
        self.noise_proj = nn.Linear(fourier_dim, c_s, bias=False)
        self.s_transitions = nn.ModuleList([EsmFold2TransitionLayer(c_s, n=transition_multiplier) for _ in range(2)])

    def compute_pair_repr(self, pair_trunk: Tensor, relative_position_encoding: Tensor) -> Tensor:
        """The pair half of the conditioning. Independent of the noise level, so the sampler builds it
        once per fold rather than every denoising step."""
        # ``pair_trunk`` is already fp32 (see ``EsmFold2TrunkOutput.pair_states``); the relative-position
        # encoding is in the compute dtype, so upcast it to match before the concat. ``z_input_norm``
        # keeps the result fp32, then we hand off to z_proj in the compute dtype.
        rel_pos = relative_position_encoding.to(dtype=torch.float32)
        pair_repr = torch.cat([pair_trunk, rel_pos], dim=-1)
        pair_repr = self.z_proj(self.z_input_norm(pair_repr).to(self.z_proj.weight.dtype))
        for block in self.z_transitions:
            pair_repr = pair_repr + block(pair_repr)
        return pair_repr

    def compute_single_repr(self, single_inputs: Tensor) -> Tensor:
        """Project the single-inputs tensor into the diffusion token width.

        ``single_inputs`` comes from the trunk and does not depend on the noise level, so this half of
        the single conditioning is step-invariant too — only the Fourier noise embedding added to it in
        :meth:`forward` varies. Computed once per fold rather than once per denoising step.
        """
        return self.s_proj(self.s_input_norm(single_inputs))

    def forward(self, t_hat: Tensor, single_repr: Tensor) -> Tensor:
        """The noise-dependent half of the single conditioning: add the Fourier noise embedding to the
        precomputed ``single_repr`` (see :meth:`compute_single_repr`) and run the transitions."""
        # ``t_hat`` arrives fp32 and flat at the expanded batch length — the diffusion module, its only
        # caller, normalizes it once before this call.
        t_noise = 0.25 * torch.log((t_hat / self.sigma_data).clamp(min=1e-20))
        # ``fourier`` holds fp32-pinned buffers, so ``noise_emb`` is already fp32.
        noise_emb = self.fourier(t_noise)
        noise_emb = self.noise_proj(self.noise_norm(noise_emb).to(self.noise_proj.weight.dtype))
        single_repr = single_repr + noise_emb.unsqueeze(1)

        for block in self.s_transitions:
            single_repr = single_repr + block(single_repr)

        return single_repr


class EsmFold2DiffusionModule(nn.Module):
    """Diffusion denoising module for structure prediction."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        c_token = config.structure_head.diffusion_module.token_hidden_size
        self.sigma_data = config.structure_head.diffusion_module.sigma_data

        self.conditioning = EsmFold2DiffusionConditioning(config)
        self.atom_encoder = EsmFold2AtomEncoder(config, structure_prediction=True)
        self.atom_decoder = EsmFold2AtomDecoder(config)
        self.s_to_token = nn.Linear(c_token, c_token, bias=False)
        self.token_transformer = EsmFold2DiffusionTransformer(config)
        self.s_step_norm = EsmFold2LayerNorm(c_token)
        self.token_norm = EsmFold2LayerNorm(c_token)

    def prepare_step_invariants(
        self,
        atom_inputs: EsmFold2AtomInputs,
        pair_trunk: Tensor,
        relative_position_encoding: Tensor,
        single_inputs: Tensor,
        num_diffusion_samples: int = 1,
    ) -> EsmFold2DiffusionStepInvariants:
        """Precompute everything that is constant across denoising steps (see
        :class:`EsmFold2DiffusionStepInvariants`), and expand the batch across diffusion samples.

        Order matters: the per-atom featurization, the pair-bias projection and the single projection
        all run first, at the unexpanded batch size, and only their (much narrower) results are
        expanded. Expanding the raw inputs instead would multiply the atom one-hot featurization and
        the ``pairwise_hidden_size``-wide pair norm by ``num_diffusion_samples``.
        """
        samples = num_diffusion_samples

        # --- unexpanded: batch_size ---
        c_base, position_embeddings = self.atom_encoder.embed_atoms(atom_inputs)
        pair_repr = self.conditioning.compute_pair_repr(pair_trunk, relative_position_encoding)
        token_pair_bias = self.token_transformer.compute_pair_biases(pair_repr)

        # --- expansion boundary: batch_size -> batch_size * num_diffusion_samples ---
        cos, sin = position_embeddings
        atom_mask = atom_inputs.atom_attention_mask.repeat_interleave(samples, 0)
        position_embeddings = (cos.repeat_interleave(samples, 0), sin.repeat_interleave(samples, 0))
        return EsmFold2DiffusionStepInvariants(
            c_base=c_base.repeat_interleave(samples, 0),
            attention_mask=self.atom_encoder.build_attention_mask(atom_mask, position_embeddings),
            position_embeddings=position_embeddings,
            atom_mask=atom_mask,
            atom_to_token=atom_inputs.atom_to_token.repeat_interleave(samples, 0),
            n_tokens=pair_trunk.shape[1],
            # Projected *after* expanding, unlike the pair bias above: a Linear reassociates
            # differently at batch_size than at batch_size * num_diffusion_samples (~1 ULP), and the
            # win here is running this once per fold instead of once per denoising step, not running
            # it on a smaller batch. Projecting post-expansion keeps the sampler bit-exact.
            single_repr_inputs=self.conditioning.compute_single_repr(single_inputs.repeat_interleave(samples, 0)),
            token_pair_bias=[bias.repeat_interleave(samples, 0) for bias in token_pair_bias],
        )

    def forward(
        self,
        x_noisy: Tensor,
        t_hat: Tensor,
        step_invariants: EsmFold2DiffusionStepInvariants,
        token_attention_mask: Tensor | None = None,
    ) -> Tensor:
        bsz = x_noisy.shape[0]
        sigma = self.sigma_data
        t = t_hat.to(dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(bsz)

        # Step 1: noise-dependent (single) conditioning; the projection it builds on is step-invariant
        single_repr = self.conditioning(t_hat=t, single_repr=step_invariants.single_repr_inputs)

        # Step 2: normalize noisy coords
        denominator = torch.sqrt(t * t + sigma * sigma)
        normalized_coords = x_noisy / denominator[:, None, None]

        # Step 3: atom encoder
        token_acts, atom_queries_skip, atom_cond_skip = self.atom_encoder(
            c_base=step_invariants.c_base,
            attention_mask=step_invariants.attention_mask,
            position_embeddings=step_invariants.position_embeddings,
            atom_mask=step_invariants.atom_mask,
            atom_to_token=step_invariants.atom_to_token,
            n_tokens=step_invariants.n_tokens,
            atom_coords=normalized_coords,
        )

        # Step 4: add conditioned single repr
        token_acts = token_acts + self.s_to_token(self.s_step_norm(single_repr))

        # Step 5: token transformer
        token_acts = self.token_transformer(
            token_acts,
            single_repr,
            step_invariants.token_pair_bias,
            attention_mask=token_attention_mask,
        )

        # Step 6: token norm
        token_acts = self.token_norm(token_acts)

        # Step 7: atom decoder
        coord_update = self.atom_decoder(
            token_acts=token_acts,
            atom_queries=atom_queries_skip,
            atom_cond=atom_cond_skip,
            attention_mask=step_invariants.attention_mask,
            position_embeddings=step_invariants.position_embeddings,
            atom_to_token=step_invariants.atom_to_token,
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
        # The sampling hyperparameters below are read from ``config`` at call time rather than copied
        # here, so editing e.g. ``model.config.structure_head.noise_scale`` after loading takes effect.
        self.config = config
        self.diffusion_module = EsmFold2DiffusionModule(config)

    def inference_noise_schedule(self, num_steps: int | None = None, device: torch.device | None = None) -> Tensor:
        """Karras power-law noise schedule."""
        sampling = self.config.structure_head
        sigma_data = sampling.diffusion_module.sigma_data
        steps = sampling.inference_num_steps if num_steps is None else int(num_steps)
        if steps == 1:
            return torch.tensor(
                [sampling.inference_s_max * sigma_data, 0.0],
                device=device,
                dtype=torch.float32,
            )
        inv_p = 1.0 / sampling.inference_p
        ramp = torch.arange(steps, device=device, dtype=torch.float32)
        base = sampling.inference_s_max**inv_p + (ramp / (steps - 1)) * (
            sampling.inference_s_min**inv_p - sampling.inference_s_max**inv_p
        )
        schedule = sigma_data * base.pow(sampling.inference_p)
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

        The schedule is capped at ``config.structure_head.max_inference_sigma``: the high-σ tail above
        the cap is truncated and the cap re-prepended so sampling still starts from it.
        """
        sampling = self.config.structure_head
        steps = sampling.inference_num_steps if num_sampling_steps is None else int(num_sampling_steps)
        schedule = self.inference_noise_schedule(steps, device)
        max_inference_sigma = sampling.max_inference_sigma
        if max_inference_sigma is not None:
            schedule = schedule[schedule <= max_inference_sigma]
            schedule = F.pad(schedule, (1, 0), value=max_inference_sigma)
        gammas = torch.where(
            schedule > sampling.gamma_min,
            torch.full_like(schedule, sampling.gamma_0),
            torch.zeros_like(schedule),
        )
        return schedule, gammas


class EsmFold2RowAttentionPooling(nn.Module):
    """Row-wise attention pooling: attn_proj, out_proj."""

    def __init__(self, d_pair: int, d_single: int) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(d_pair, 1, bias=False)
        self.out_proj = nn.Linear(d_pair, d_single, bias=False)

    def forward(self, pair_repr: Tensor, attention_mask: Tensor) -> Tensor:
        scores = self.attn_proj(pair_repr).squeeze(-1)
        mask_bias = torch.where(
            attention_mask[:, None, :].bool(),
            torch.zeros_like(scores),
            torch.full_like(scores, torch.finfo(scores.dtype).min),
        )
        scores = scores + mask_bias
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        pooled = torch.einsum("bnm,bnmd->bnd", weights, pair_repr)
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

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.downproject(hidden_states)
        hidden_states = torch.cat(
            [
                (hidden_states.unsqueeze(2) * hidden_states.unsqueeze(1)),
                (hidden_states.unsqueeze(2) - hidden_states.unsqueeze(1)),
            ],
            dim=3,
        )
        return self.output_fc2(F.gelu(self.output_fc1(hidden_states)))


class EsmFold2LanguageModelShim(nn.Module):
    """Shim holding the trainable projection weights for LM integration.

    Contains:
    - base_z_combine: nn.Parameter [num_layers+1]
    - base_z_input_norm -> base_z_proj: EsmFold2LayerNorm(d_model) then Linear(d_model, d_z, bias=False)
    - base_z_to_pair -> base_z_output_norm: EsmFold2SingleToPair(d_z, d_z, d_z) then EsmFold2LayerNorm(d_z)
    """

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        d_z = config.pairwise_hidden_size
        d_model = config.esmc_config.d_model
        num_layers = config.esmc_config.n_layers

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

    def __init__(self, dim: int, outgoing: bool = True, chunk_size: int | None = 64) -> None:
        super().__init__()
        self.dim = dim
        self.flow = "outgoing" if outgoing else "incoming"
        self.norm_start = EsmFold2LayerNorm(dim)
        self.norm_mix = EsmFold2LayerNorm(dim)
        self.proj_bundle = nn.Linear(dim, 4 * dim, bias=False)
        self.proj_emit = nn.Linear(dim, dim, bias=False)
        self.proj_gate = nn.Linear(dim, dim, bias=False)

        # Chunk the O(N^3) contraction for memory on long sequences, from ``config.chunk_size``.
        self.chunk_size: int | None = chunk_size

    def _triangular_contract(self, left_stream: Tensor, right_stream: Tensor) -> Tensor:
        """Triangular einsum, chunked along the output i-dimension.

        A falsy ``chunk_size`` (``None`` / ``0``) means one chunk spanning the whole sequence, i.e.
        the unchunked contraction — same single code path, bit-identical result.
        """
        seq_len = left_stream.shape[1] if self.flow == "outgoing" else left_stream.shape[2]
        chunk_size = self.chunk_size or seq_len
        chunks = []
        for start in range(0, seq_len, chunk_size):
            window = slice(start, start + chunk_size)
            if self.flow == "outgoing":
                chunks.append(torch.einsum("bikd,bjkd->bijd", left_stream[:, window], right_stream))
            else:
                chunks.append(torch.einsum("bkid,bkjd->bijd", left_stream[:, :, window], right_stream))
        return torch.cat(chunks, dim=1)

    def forward(self, pair_grid: Tensor, visibility: Tensor | None = None) -> Tensor:
        if visibility is None:
            visibility = pair_grid.new_ones(pair_grid.shape[:-1])

        normalized_grid = self.norm_start(pair_grid)
        bundled = self.proj_bundle(normalized_grid)
        signal, gate_logits = bundled.split(2 * self.dim, dim=-1)
        routed = signal * torch.sigmoid(gate_logits)
        # ``visibility`` is the fp32 pair mask; cast it so masking does not silently promote ``routed``
        # and run the O(N^3) contraction in fp32. The reference contracts at the ambient precision --
        # it upcasts explicitly, but its own bf16 autocast overrides that, so a bf16 model contracts in
        # bf16 and an fp32 model in fp32. Keeping ``routed`` in the activation dtype reproduces both.
        routed = routed * visibility.unsqueeze(-1).to(routed.dtype)

        left_stream, right_stream = routed.chunk(2, dim=-1)
        contracted = self._triangular_contract(left_stream, right_stream)
        mixed = self.proj_emit(self.norm_mix(contracted))
        output_gate = torch.sigmoid(self.proj_gate(normalized_grid))
        return mixed * output_gate


class EsmFold2Transition(nn.Module):
    """LayerNorm + EsmFold2SwiGLU feed-forward residual block, chunked along the token axis."""

    def __init__(self, d_model: int, intermediate_size: int, chunk_size: int | None = 64) -> None:
        super().__init__()
        self.norm = EsmFold2LayerNorm(d_model)
        self.ffn = EsmFold2SwiGLU(d_model, intermediate_size)
        # Chunk along the token axis on long sequences, from ``config.chunk_size``. A falsy value
        # (``None`` / ``0``) means one chunk spanning the whole axis.
        self.chunk_size: int | None = chunk_size

    def forward(self, hidden_states: Tensor) -> Tensor:
        seq_len = hidden_states.shape[1]
        chunk_size = self.chunk_size or seq_len
        chunks: list[Tensor] = []
        for start in range(0, seq_len, chunk_size):
            chunk = hidden_states[:, start : start + chunk_size]
            chunks.append(chunk + self.ffn(self.norm(chunk)))
        return torch.cat(chunks, dim=1)


class EsmFold2PairUpdateBlock(GradientCheckpointingLayer):
    """tri_mul_out, tri_mul_in, pair_transition."""

    def __init__(self, d_pair: int, intermediate_size: int, chunk_size: int | None = 64) -> None:
        super().__init__()
        self.tri_mul_out = EsmFold2TriangleMultiplicativeUpdate(dim=d_pair, outgoing=True, chunk_size=chunk_size)
        self.tri_mul_in = EsmFold2TriangleMultiplicativeUpdate(dim=d_pair, outgoing=False, chunk_size=chunk_size)
        self.pair_transition = EsmFold2Transition(d_pair, intermediate_size, chunk_size=chunk_size)

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

    def forward(self, pair: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            pair = layer(pair, pair_attention_mask=pair_attention_mask)
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
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.divide_outer_before_proj = divide_outer_before_proj
        self.norm = EsmFold2LayerNorm(d_msa)
        self.W = nn.Linear(d_msa, 2 * d_hidden, bias=False)
        self.Wout = nn.Linear(d_hidden * d_hidden, d_pair, bias=True)
        # From ``config.msa_encoder.outer_product_chunk_size``, which defaults to off: unlike the
        # other chunked ops, chunking this einsum is not always bit-exact in bf16 (the kernel
        # picked depends on the operand shape). A falsy value means one full-length chunk.
        self.chunk_size: int | None = chunk_size

    def forward(self, msa_repr: Tensor, msa_attention_mask: Tensor) -> Tensor:
        msa_normed = self.norm(msa_repr)
        x = self.W(msa_normed) * msa_attention_mask.unsqueeze(-1).to(msa_normed.dtype)
        left, right = x.chunk(2, dim=-1)
        mask_f = msa_attention_mask.to(left.dtype)
        n_valid = (mask_f @ mask_f.transpose(-1, -2)).unsqueeze(-1).clamp(min=1.0)
        # Chunk along the left (i) axis so the peak einsum intermediate is
        # [B, chunk, L, c, d] instead of [B, L, L, c, d].
        seq_len = left.shape[1]
        chunk_size = self.chunk_size or seq_len
        out_chunks: list[Tensor] = []
        for start in range(0, seq_len, chunk_size):
            window = slice(start, start + chunk_size)
            outer = torch.einsum("bimc,bjmd->bijcd", left[:, window], right).flatten(-2)
            if self.divide_outer_before_proj:
                out_chunks.append(self.Wout(outer / n_valid[:, window]))
            else:
                out_chunks.append(self.Wout(outer) / n_valid[:, window])
        return torch.cat(out_chunks, dim=1)


class EsmFold2MSAPairWeightedAveraging(nn.Module):
    """Pair-biased MSA row update (AF3 Supplement Algorithm 10)."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        d_msa = config.msa_encoder.hidden_size
        d_pair = config.pairwise_hidden_size
        self.n_heads = config.msa_encoder.num_attention_heads
        self.head_width = config.msa_encoder.head_width
        self.norm_single = EsmFold2LayerNorm(d_msa)
        self.bias_norm = EsmFold2LayerNorm(d_pair)
        self.bias_proj = nn.Linear(d_pair, self.n_heads, bias=False)
        self.Wv = nn.Linear(d_msa, self.n_heads * self.head_width, bias=False)
        self.Wgate = nn.Linear(d_msa, self.n_heads * self.head_width, bias=False)
        self.Wout = nn.Linear(self.n_heads * self.head_width, d_msa, bias=False)

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
        bias.masked_fill_(~pair_attention_mask.unsqueeze(-1).bool(), torch.finfo(bias.dtype).min)
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


@dataclass
class EsmFold2TrunkOutput(ModelOutput):
    """
    Output of [`EsmFold2Model`]: the folding trunk's pair representation, the distogram read off it,
    and the conditioning tensors that the structure and confidence heads consume. Everything here is
    deterministic given the inputs apart from the trunk's random initial pair state.

    Args:
        distogram_logits (`torch.FloatTensor` of shape `(batch_size, num_tokens, num_tokens, distogram_bins)`):
            Predicted distance-distribution logits over residue pairs.
        pair_states (`torch.FloatTensor` of shape `(batch_size, num_tokens, num_tokens, pairwise_hidden_size)`):
            The trunk's final pair representation, in fp32.
        single_inputs (`torch.FloatTensor` of shape `(batch_size, num_tokens, single_inputs_size)`):
            Concatenated single-input features built by the inputs embedder.
        relative_position_encoding (`torch.FloatTensor` of shape `(batch_size, num_tokens, num_tokens, pairwise_hidden_size)`):
            Relative-position pair encoding.
        token_bonds_encoding (`torch.FloatTensor` of shape `(batch_size, num_tokens, num_tokens, pairwise_hidden_size)`):
            Embedded inter-token covalent-bond feature.
        atom_inputs ([`EsmFold2AtomInputs`]):
            The featurized reference-conformer atom inputs, reused by the diffusion atom stack.
    """

    distogram_logits: Tensor | None = None
    pair_states: Tensor | None = None
    single_inputs: Tensor | None = None
    relative_position_encoding: Tensor | None = None
    token_bonds_encoding: Tensor | None = None
    atom_inputs: EsmFold2AtomInputs | None = None


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

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        # Additive guard for masked-mean denominators (empty chains / all-padding rows), kept as an
        # explicit ``+ eps`` (rather than clamping the denominator to 1) to reproduce the reference numerics.
        self.eps = config.confidence_head.eps
        d_single = config.hidden_size
        d_pair = config.pairwise_hidden_size
        d_inputs = config.single_inputs_size
        distogram_bins = config.confidence_head.distogram_bins

        boundaries = torch.linspace(
            config.confidence_head.min_dist, config.confidence_head.max_dist, distogram_bins - 1
        )
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(distogram_bins, d_pair)

        self.input_embedder = EsmFold2ConfidenceInputEmbedder(d_pair=d_pair, d_inputs=d_inputs)

        self.row_attention_pooling = EsmFold2RowAttentionPooling(d_pair=d_pair, d_single=d_single)

        self.folding_trunk = EsmFold2FoldingTrunk(
            n_layers=config.confidence_head.num_hidden_layers,
            d_pair=d_pair,
            intermediate_size=config.pair_transition_intermediate_size,
            chunk_size=config.chunk_size,
        )

        # Heads.
        self.plddt_ln = EsmFold2LayerNorm(d_single)
        max_atoms_per_token = config.max_atoms_per_token
        self.plddt_weight = nn.Parameter(
            torch.zeros(max_atoms_per_token, d_single, config.confidence_head.num_plddt_bins)
        )

        self.pae_ln = EsmFold2LayerNorm(d_pair)
        self.pae_head = nn.Linear(d_pair, config.confidence_head.num_pae_bins, bias=False)

        self.pde_ln = EsmFold2LayerNorm(d_pair)
        self.pde_head = nn.Linear(d_pair, config.confidence_head.num_pde_bins, bias=False)

        self.resolved_ln = EsmFold2LayerNorm(d_single)
        # 2 = resolved logits ([unresolved, resolved]).
        self.resolved_weight = nn.Parameter(torch.zeros(max_atoms_per_token, d_single, 2))

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
            # ``atom_mask_f`` and ``plddt_per_atom`` are both fp32, as ``scatter_add_`` requires.
            plddt_sum.scatter_add_(1, atom_to_token_m, plddt_per_atom * atom_mask_f)
            atom_count.scatter_add_(1, atom_to_token_m, atom_mask_f)
            plddt = plddt_sum / atom_count.clamp(min=1e-6)

            complex_plddt = (plddt_per_atom * atom_mask_f).sum(dim=-1) / (atom_mask_f.sum(dim=-1) + self.eps)

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
            complex_iplddt = (plddt_per_atom * atom_iplddt_w).sum(dim=-1) / (atom_iplddt_w.sum(dim=-1) + self.eps)

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
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (pair_mask_2d.sum(dim=-1) + self.eps)
        ptm = ptm_per_row.max(dim=-1).values

        inter_chain_mask = (expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)).float() * pair_mask_2d
        iptm_per_row = (tm_expected * inter_chain_mask).sum(dim=-1) / (inter_chain_mask.sum(dim=-1) + self.eps)
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
            avg_tm = (tm_expected * col_mask).sum(dim=-1) / (col_mask.sum(dim=-1) + self.eps)
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
        d_msa = config.msa_encoder.hidden_size
        d_pair = config.pairwise_hidden_size
        d_hidden = config.msa_encoder.outer_hidden_size
        self.is_final_block = is_final_block
        self.outer_product_mean = EsmFold2OuterProductMean(
            d_msa,
            d_hidden,
            d_pair,
            divide_outer_before_proj=config.msa_encoder.divide_outer_before_proj,
            chunk_size=config.msa_encoder.outer_product_chunk_size,
        )
        if not is_final_block:
            self.msa_pair_weighted_averaging = EsmFold2MSAPairWeightedAveraging(config)
            self.msa_transition = EsmFold2Transition(
                d_msa, config.msa_encoder.transition_intermediate_size, chunk_size=config.chunk_size
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
        d_msa = config.msa_encoder.hidden_size
        d_inputs = config.single_inputs_size
        n_layers = config.msa_encoder.num_hidden_layers
        # num_res_types one-hot + has_deletion + deletion_value.
        self.embed = nn.Linear(config.num_res_types + 2, d_msa, bias=False)
        self.project_inputs = nn.Linear(d_inputs, d_msa, bias=False)
        self.layers = nn.ModuleList(
            [EsmFold2MSAEncoderBlock(config, is_final_block=(i == n_layers - 1)) for i in range(n_layers)]
        )

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


class EsmFold2Parcae(nn.Module):
    """The trunk's linear-recurrence state update (internally "parcae") and its coda pair stack.

    Each trunk loop injects the refreshed pair representation into a discretized linear state-space
    recurrence -- ``state_decay`` (Ā) and ``input_matrix`` (B̄) come from ``log_a``/``log_delta``/
    ``b_cont`` via :meth:`discretized_dynamics` -- and the final state is read out through ``readout``
    and refined by ``coda``. The parameters are allocated here and initialized in
    ``EsmFold2PreTrainedModel._init_weights`` (``log_a`` -> 0, ``log_delta`` -> a fixed decay constant,
    ``b_cont``/``readout`` -> identity).
    """

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        d_pair = config.pairwise_hidden_size
        self.input_norm = EsmFold2LayerNorm(d_pair)
        self.log_a = nn.Parameter(torch.zeros(d_pair))
        self.log_delta = nn.Parameter(torch.empty(d_pair, dtype=torch.float32))
        self.b_cont = nn.Parameter(torch.empty(d_pair, d_pair))
        self.readout = nn.Linear(d_pair, d_pair, bias=False)
        self.coda = EsmFold2FoldingTrunk(
            n_layers=config.parcae_num_coda_layers,
            d_pair=d_pair,
            intermediate_size=config.pair_transition_intermediate_size,
            chunk_size=config.chunk_size,
        )

    def discretized_dynamics(self) -> tuple[Tensor, Tensor]:
        """``(state_decay, input_matrix)`` -- the per-channel state transition (Ā) and the discretized
        input projection (B̄) of the recurrence."""
        delta = F.softplus(self.log_delta)
        state_decay = torch.exp(-delta * torch.exp(self.log_a))
        input_matrix = delta[:, None] * self.b_cont
        return state_decay, input_matrix


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
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$"]
    # Keep every normalization weight (and the Fourier noise-embedding features) in fp32 even under
    # dtype=bf16, matching the transformers norm convention. Tokens:
    #   - "norm"/"_ln": every LayerNorm weight/bias across the trunk and the bundled ESMC backbone
    #     (the backbone's norms would otherwise round to bf16; here they stay fp32), plus the
    #     adaLN-Zero conditioning scale (``norm_scale``). No linear/proj parameter in either
    #     sub-model contains these tokens, so only norms are matched.
    #   - "fourier": the random-Gaussian noise-embedding frequencies/phases whose precision drives
    #     the diffusion conditioning.
    _keep_in_fp32_modules_strict = ["fourier", "norm", "_ln"]
    _supports_sdpa = True

    def _init_weights(self, module):
        # The non-default weight inits (adaLN-Zero gates, the parcae recurrence, zeroed output
        # projections). They live here rather than in each submodule's __init__ so post_init()
        # applies them; on a from_pretrained load the init.* helpers below no-op, leaving the
        # checkpoint weights untouched.
        super()._init_weights(module)
        if isinstance(module, EsmFold2Parcae):
            init.eye_(module.readout.weight)
            init.eye_(module.b_cont)
            init.zeros_(module.log_a)
            parcae_delta_init = -math.log(math.sqrt(1.0 / 5.0))
            init.constant_(module.log_delta, _inverse_softplus(parcae_delta_init))
        elif isinstance(module, EsmFold2ConfidenceHead):
            init.zeros_(module.plddt_weight)
            init.zeros_(module.resolved_weight)
        elif isinstance(module, EsmFold2AdaptiveLayerNorm):
            init.ones_(module.norm_scale)
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
class EsmFold2Model(EsmFold2PreTrainedModel, EsmFold2GenerationMixin):
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
        self.language_model = EsmFold2LanguageModelShim(config)
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
            n_layers=config.lm_encoder.num_hidden_layers,
            d_pair=d_pair,
            intermediate_size=config.pair_transition_intermediate_size,
            chunk_size=config.chunk_size,
        )

        self.parcae = EsmFold2Parcae(config)

        # Heads --------------------------------------------------------------
        self.structure_head = EsmFold2DiffusionStructureHead(config)
        self.distogram_head = nn.Linear(d_pair, config.structure_head.distogram_bins, bias=True)
        self.confidence_head = EsmFold2ConfidenceHead(config)

        self.msa_encoder = EsmFold2MSAEncoder(config)

        self.post_init()

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

    def _run_trunk_loops(
        self,
        z: Tensor,
        z_init: Tensor,
        lm_z: Tensor | None,
        msa_kwargs: dict | None,
        pair_mask: Tensor,
        state_decay: Tensor,
        input_matrix: Tensor,
        total_steps: int,
    ) -> Tensor:
        # A helper rather than an inline loop so the per-iteration L^2 x c_z locals are freed on
        # return instead of staying alive through the distogram/sampling scope (~2 GB at length 1k).
        # ``training=True`` forces the per-loop LM dropout to resample under ``eval()``.
        lm_dropout_p = self.config.lm_encoder.lm_dropout
        per_loop_lm_dropout = lm_z is not None and self.config.lm_encoder.per_loop_lm_dropout and lm_dropout_p > 0.0

        for _ in range(total_steps):
            if per_loop_lm_dropout:
                lm_z_i: Tensor | None = F.dropout(lm_z, p=lm_dropout_p, training=True)
            else:
                lm_z_i = lm_z

            refined_lm_z: Tensor | None = None
            if lm_z_i is not None:
                refined_lm_z = self.lm_encoder(lm_z_i, pair_attention_mask=pair_mask)

            z_inject_pair = z_init
            if msa_kwargs is not None:
                msa_pair = self.msa_encoder(x_pair=z_inject_pair, **msa_kwargs)
                z_inject_pair = msa_pair if self.config.msa_encoder.overwrite else (z_inject_pair + msa_pair)

            if refined_lm_z is not None:
                z_inject_pair = z_inject_pair + refined_lm_z

            injected_pair = self.parcae.input_norm(z_inject_pair)
            z = state_decay * z + F.linear(injected_pair, input_matrix)
            z = self.folding_trunk(z, pair_attention_mask=pair_mask)

        return z

    @auto_docstring(
        custom_intro="""
        Run the folding trunk: featurize the inputs, embed them into a pair representation, refine it over
        `num_loops` recycling iterations, and read off the distogram. This is the deterministic half of a
        structure prediction; the diffusion sampler that turns the returned pair representation into 3D
        coordinates lives in `EsmFold2GenerationMixin` — call [`~EsmFold2Model.fold`] or
        [`~EsmFold2Model.infer_protein`] for an end-to-end prediction.
        """
    )
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
        deletion_mean: Tensor | None = None,
        msa: Tensor | None = None,
        has_deletion: Tensor | None = None,
        deletion_value: Tensor | None = None,
        msa_attention_mask: Tensor | None = None,
        input_ids: Tensor | None = None,
        lm_hidden_states: Tensor | None = None,
        num_loops: int | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> EsmFold2TrunkOutput:
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
        """
        tok_mask = token_attention_mask
        n_loops: int = num_loops if num_loops is not None else self.config.num_loops
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
                atom_attention_mask=atom_attention_mask,
                atom_to_token=atom_to_token,
            )
        )

        atom_inputs = EsmFold2AtomInputs(
            ref_pos=ref_pos,
            ref_charge=ref_charge,
            atom_attention_mask=atom_attention_mask,
            ref_element=ref_element_oh,
            ref_atom_name_chars=ref_atom_name_chars_oh,
            ref_space_uid=ref_space_uid,
            atom_to_token=atom_to_token,
        )

        # The inputs embedder runs its atom stack exactly once and never expands across diffusion
        # samples, so it just builds what it needs and calls straight through.
        c_base, position_embeddings = self.inputs_atom_encoder.embed_atoms(atom_inputs)
        atom_encoding = self.inputs_atom_encoder(
            c_base=c_base,
            attention_mask=self.inputs_atom_encoder.build_attention_mask(atom_attention_mask, position_embeddings),
            position_embeddings=position_embeddings,
            atom_mask=atom_attention_mask,
            atom_to_token=atom_to_token,
            n_tokens=tok_mask.shape[1],
        )[0]
        # The continuous input features are fp32; fold them into the atom encoding's
        # (compute) dtype so the single representation is one dtype.
        dtype = atom_encoding.dtype
        single_inputs = torch.cat(
            [
                atom_encoding,
                res_type_oh.to(dtype),
                profile.to(dtype),
                deletion_mean.unsqueeze(-1).to(dtype),
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

        state_decay, input_matrix = self.parcae.discretized_dynamics()
        state_decay = state_decay.view(1, 1, 1, -1).to(device=z.device, dtype=z.dtype)
        input_matrix = input_matrix.to(device=z.device, dtype=z.dtype)

        msa_kwargs = self._build_msa_kwargs(
            msa=msa,
            msa_attention_mask=msa_attention_mask,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            tok_mask=tok_mask,
            single_inputs=single_inputs,
        )

        z = self._run_trunk_loops(
            z=z,
            z_init=z_init,
            lm_z=lm_z,
            msa_kwargs=msa_kwargs,
            pair_mask=pair_mask,
            state_decay=state_decay,
            input_matrix=input_matrix,
            total_steps=total_steps,
        )
        del z_init, lm_z, msa_kwargs, state_decay, input_matrix

        z = self.parcae.readout(z)
        z = self.parcae.coda(z, pair_attention_mask=pair_mask)

        z = z.float()
        distogram_logits = self.distogram_head((z + z.transpose(-2, -3)).to(self.distogram_head.weight.dtype))

        return EsmFold2TrunkOutput(
            distogram_logits=distogram_logits,
            pair_states=z,
            single_inputs=single_inputs,
            relative_position_encoding=relative_position_encoding,
            token_bonds_encoding=token_bonds_encoding,
            atom_inputs=atom_inputs,
        )


__all__ = ["EsmFold2Model", "EsmFold2PreTrainedModel"]
