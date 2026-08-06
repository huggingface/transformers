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
from ...activations import ACT2FN
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_bidirectional_mask, sliding_window_bidirectional_overlay
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import ModelOutput
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ..auto import AutoModel
from ..llama.modeling_llama import eager_attention_forward, rotate_half
from ..nanochat.modeling_nanochat import NanoChatRMSNorm
from ..phi3.modeling_phi3 import Phi3MLP
from .configuration_esmfold2 import EsmFold2AtomEncoderConfig, EsmFold2Config, EsmFold2DiffusionModuleConfig
from .generation_esmfold2 import EsmFold2FoldingMixin


@dataclass
class EsmFold2AtomInputs:
    """Featurized reference-conformer atom inputs, bundled so the atom stack takes one argument.

    Built by ``EsmFold2Model._prepare_features``, which one-hot encodes the categorical fields and
    zeroes the padding. Returned on [`EsmFold2TrunkOutput`] so the diffusion sampler and the
    confidence head reuse the trunk's featurization rather than redoing it.

    A plain dataclass rather than a [`~utils.ModelOutput`]: every field here is required and is read
    unconditionally downstream, and ``ModelOutput`` permits at most one required field (the rest must
    default to ``None``), which would turn a missing tensor from a constructor error into a ``None``
    surfacing deep in the atom encoder.

    Args:
        ref_pos: `(batch_size, num_atoms, 3)` reference-conformer Cartesian coordinates.
        ref_charge: `(batch_size, num_atoms)` formal charge of each atom.
        atom_attention_mask: `(batch_size, num_atoms)` valid atoms (`1`) versus padding (`0`).
        ref_element: `(batch_size, num_atoms, max_atomic_number)` one-hot atomic number.
        ref_atom_name_chars: `(batch_size, num_atoms, max_chars, char_vocab_size)` one-hot atom name.
        ref_space_uid: `(batch_size, num_atoms)` per-atom group ID (the atom's token index).
        atom_to_token: `(batch_size, num_atoms)` index of the token each atom belongs to.
    """

    ref_pos: Tensor
    ref_charge: Tensor
    atom_attention_mask: Tensor
    ref_element: Tensor
    ref_atom_name_chars: Tensor
    ref_space_uid: Tensor
    atom_to_token: Tensor


@dataclass
class EsmFold2DenoiserConditioning:
    """Denoiser conditioning that depends on neither the noise level nor the noisy coordinates.

    Built once per fold by ``EsmFold2DiffusionModule.prepare_conditioning``, with both attention masks
    folded in, so nothing downstream takes a mask. See there for which fields are expanded across
    diffusion samples and which are left broadcastable.
    """

    # Atom stack. ``attention_mask`` is the prepared 4D sliding-window mask; ``atom_mask`` the 2D
    # valid-atom mask that the atom->token scatter needs.
    atom_embeds: Tensor
    attention_mask: Tensor
    position_embeddings: tuple[Tensor, Tensor]
    atom_mask: Tensor
    atom_to_token: Tensor
    # Token stack. One additive bias per block, `[batch, num_attention_heads, num_queries, num_keys]`.
    projected_single_inputs: Tensor
    token_attention_bias: list[Tensor]


class EsmFold2LayerNorm(nn.LayerNorm):
    """LayerNorm that computes in fp32 (its weight/bias are fp32-pinned) and returns its input dtype."""

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.layer_norm(hidden_states.float(), self.normalized_shape, self.weight, self.bias, self.eps).to(
            hidden_states.dtype
        )


class EsmFold2RMSNorm(NanoChatRMSNorm):
    """Weightless RMSNorm that computes in fp32 and returns its input dtype."""


class EsmFold2Transition(nn.Module):
    """LayerNorm + SwiGLU feed-forward residual block, chunked along the token axis.

    ``chunk_size`` has no default and every call site states it, because passing ``None`` (unchunked)
    is a deliberate numerical choice at two of them: chunking is exact, but its reassociation moves
    the last bf16 bits, which the diffusion sampler amplifies over its steps.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, chunk_size: int | None) -> None:
        super().__init__()
        self.norm = EsmFold2LayerNorm(hidden_size)
        self.mlp = EsmFold2SwiGLU(hidden_size, intermediate_size)
        # A falsy chunk size means one chunk spanning the whole axis, i.e. unchunked.
        self.chunk_size = chunk_size

    def forward(self, hidden_states: Tensor) -> Tensor:
        seq_len = hidden_states.shape[1]
        chunk_size = self.chunk_size or seq_len
        chunks: list[Tensor] = []
        for start in range(0, seq_len, chunk_size):
            chunk = hidden_states[:, start : start + chunk_size]
            chunks.append(chunk + self.mlp(self.norm(chunk)))
        return torch.cat(chunks, dim=1)


class EsmFold2AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalization (adaLN-Zero)."""

    def __init__(self, config: EsmFold2DiffusionModuleConfig, eps: float = 1e-5) -> None:
        super().__init__()
        # Both the token and the single stream are the diffusion token width at both call sites.
        self.hidden_size = config.hidden_size
        self.eps = eps
        self.cond_norm = EsmFold2LayerNorm(self.hidden_size, eps=eps, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.shift_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor, single_states: Tensor) -> Tensor:
        # Weightless, and deliberately not an ``EsmFold2LayerNorm``: that returns the input dtype,
        # whereas leaving this fp32 lets the affine below promote and round exactly once.
        normed_hidden_states = F.layer_norm(hidden_states.float(), (self.hidden_size,), None, None, self.eps)
        normed_conditioning = self.cond_norm(single_states)
        # ``normed_hidden_states`` is fp32, so the affine promotes; downcast for the next op.
        gate = torch.sigmoid(self.gate_proj(normed_conditioning))
        shift = self.shift_proj(normed_conditioning)
        return (gate * normed_hidden_states + shift).to(hidden_states.dtype)


class EsmFold2FourierEmbedding(nn.Module):
    """Fourier embedding ``cos(2*pi*(t*frequencies + phases))``, with the random frequencies and
    phases sampled once and stored in the checkpoint rather than learned."""

    frequencies: Tensor
    phases: Tensor

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.randn(embedding_dim))
        self.register_buffer("phases", torch.randn(embedding_dim))

    def forward(self, noise_level: Tensor) -> Tensor:
        # ``noise_level`` and the buffers are both fp32, so the angles are built in fp32.
        return torch.cos(2.0 * torch.pi * (noise_level[:, None] * self.frequencies[None, :] + self.phases[None, :]))


class EsmFold2SwiGLU(Phi3MLP):
    """SwiGLU feed-forward with a fused gate+up projection (``gate_up_proj``) and output ``down_proj``.

    Takes explicit widths rather than a config: the six call sites draw them from four different
    sub-configs.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        nn.Module.__init__(self)
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation_fn = ACT2FN["silu"]


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Identical to Llama's, but deliberately *not* inherited from it: Llama's carries
    ``@use_kernel_func_from_hub("rotary_pos_emb")``, and the fused kernel is CUDA-only while the atom
    stack's rope runs on CPU in the tests. Swapping in a kernel here would also break the bit-exact
    bf16 reference comparison.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The dimension along which to unsqueeze `cos` and `sin` so they broadcast against `q`/`k`.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class EsmFold2AtomAttention(nn.Module):
    """Sliding-window self-attention with 3D RoPE and a gated output.

    The sliding window is expressed as an additive attention mask; dims come from ``atom_config``,
    the atom sub-config `EsmFold2AtomEncoder` resolved for this call site.
    """

    def __init__(self, config: EsmFold2Config, atom_config: EsmFold2AtomEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = atom_config.head_dim
        self.scaling = self.head_dim**-0.5
        # No grouped-query attention; identity repeat keeps the interface happy.
        self.num_key_value_groups = 1
        self.is_causal = False  # bidirectional encoder, even when the mask is None

        self.q_proj = nn.Linear(atom_config.hidden_size, atom_config.hidden_size, bias=False)
        self.k_proj = nn.Linear(atom_config.hidden_size, atom_config.hidden_size, bias=False)
        self.v_proj = nn.Linear(atom_config.hidden_size, atom_config.hidden_size, bias=False)
        self.o_proj = nn.Linear(atom_config.hidden_size, atom_config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(atom_config.hidden_size, atom_config.hidden_size, bias=False)
        self.q_norm = EsmFold2RMSNorm()
        self.k_norm = EsmFold2RMSNorm()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )
        # Needed because padding can sometimes create tokens with no attendable keys, creating NaN outputs
        attn_output = torch.nan_to_num(attn_output)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))
        return self.o_proj(attn_output), attn_weights


class EsmFold2AtomLayer(nn.Module):
    """adaLN-Zero + SWA attention + SwiGLU FFN, modulated by ``adaln_linear(silu(atom_conditioning))``."""

    def __init__(self, config: EsmFold2Config, atom_config: EsmFold2AtomEncoderConfig) -> None:
        super().__init__()
        # adaln-Zero gate; zero-init lives in EsmFold2PreTrainedModel._init_weights.
        self.adaln_linear = nn.Linear(atom_config.hidden_size, 6 * atom_config.hidden_size, bias=False)

        self.self_attn = EsmFold2AtomAttention(config, atom_config)
        self.mlp = EsmFold2SwiGLU(atom_config.hidden_size, atom_config.intermediate_size)
        self.input_layernorm = EsmFold2RMSNorm()
        self.post_attention_layernorm = EsmFold2RMSNorm()

    def forward(
        self,
        hidden_states: Tensor,
        atom_conditioning: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        modulation = self.adaln_linear(F.silu(atom_conditioning))
        if modulation.dim() == 2:
            modulation = modulation.unsqueeze(1)
        attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate = modulation.chunk(6, dim=-1)

        attn_hidden_states = self.input_layernorm(hidden_states) * (1 + attn_scale) + attn_shift
        attn_output, _ = self.self_attn(attn_hidden_states, attention_mask, position_embeddings)
        hidden_states = hidden_states + attn_gate * attn_output

        mlp_hidden_states = self.post_attention_layernorm(hidden_states) * (1 + mlp_scale) + mlp_shift
        mlp_output = self.mlp(mlp_hidden_states)
        hidden_states = hidden_states + mlp_gate * mlp_output
        return hidden_states


class EsmFold2RotaryEmbedding(nn.Module):
    """Rotary embedding over physical position (``ref_pos`` = x/y/z) plus a per-atom space UID,
    rather than token indices, with separate spatial and UID base frequencies.

    Returns cos/sin at the full head dim, so callers apply plain rotate-half RoPE.
    """

    def __init__(self, atom_config: EsmFold2AtomEncoderConfig) -> None:
        super().__init__()
        self.head_dim = atom_config.head_dim
        self.num_spatial_pairs_per_axis = atom_config.num_spatial_rope_pairs_per_axis
        self.num_uid_pairs = atom_config.num_uid_rope_pairs
        self.spatial_rope_base_frequency = atom_config.spatial_rope_base_frequency
        self.uid_rope_base_frequency = atom_config.uid_rope_base_frequency

    def forward(self, ref_pos: Tensor, ref_space_uid: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        device = ref_pos.device
        batch_size, num_atoms = ref_pos.shape[:2]
        half_dim = self.head_dim // 2
        num_spatial_frequencies = 3 * self.num_spatial_pairs_per_axis

        spatial_inv_freq = 1.0 / (
            self.spatial_rope_base_frequency
            ** (
                torch.arange(0, self.num_spatial_pairs_per_axis, dtype=torch.float32, device=device)
                / self.num_spatial_pairs_per_axis
            )
        )
        uid_inv_freq = 1.0 / (
            self.uid_rope_base_frequency
            ** (torch.arange(0, self.num_uid_pairs, dtype=torch.float32, device=device) / self.num_uid_pairs)
        )

        spatial_freqs = (ref_pos.unsqueeze(-1) * spatial_inv_freq).reshape(
            batch_size, num_atoms, num_spatial_frequencies
        )
        uid_freqs = ref_space_uid.float().unsqueeze(-1) * uid_inv_freq

        freqs = torch.cat([spatial_freqs, uid_freqs], dim=-1)
        num_active_frequencies = num_spatial_frequencies + self.num_uid_pairs
        if num_active_frequencies < half_dim:
            freqs = torch.cat(
                [freqs, freqs.new_zeros(batch_size, num_atoms, half_dim - num_active_frequencies)], dim=-1
            )

        # Duplicate to the full head dim; angles are built in fp32 and returned at the caller's dtype.
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def scatter_atom_to_token(
    atom_features: Tensor,
    atom_to_token_idx: Tensor,
    num_tokens: int,
    atom_mask: Tensor | None = None,
) -> Tensor:
    """Aggregate per-atom features to per-token features (mean).

    Args:
        atom_features: `(batch_size, num_atoms, hidden_size)`
        atom_to_token_idx: `(batch_size, num_atoms)` int64
        num_tokens: the token-axis length to scatter into
        atom_mask: `(batch_size, num_atoms)` bool

    Returns:
        `(batch_size, num_tokens, hidden_size)`
    """
    batch_size, _, hidden_size = atom_features.shape
    idx = atom_to_token_idx
    if atom_mask is not None:
        idx = idx.masked_fill(~atom_mask, num_tokens)
    token_features = atom_features.new_zeros(batch_size, num_tokens + int(atom_mask is not None), hidden_size)
    token_features.scatter_reduce_(
        dim=1,
        index=idx.unsqueeze(-1).expand_as(atom_features),
        src=atom_features,
        reduce="mean",
        include_self=False,
    )
    return token_features[:, :num_tokens]


class EsmFold2AtomEncoder(nn.Module):
    """SWA atom encoder. ``structure_prediction=True`` (the diffusion module) adds ``coords_linear``.

    Resolves which of the two atom sub-configs a call site uses and hands it down, so nothing below
    repeats the choice.
    """

    def __init__(self, config: EsmFold2Config, structure_prediction: bool = True) -> None:
        super().__init__()
        atom_config = (
            config.structure_head.diffusion_module.atom_encoder if structure_prediction else config.atom_encoder
        )
        self.structure_prediction = structure_prediction

        # The atom-name-char slice of `config.atom_feature_dim`.
        self.char_feature_dim = config.char_vocab_size * config.max_chars
        self.atom_linear = nn.Linear(config.atom_feature_dim, atom_config.hidden_size, bias=False)
        self.atom_norm = EsmFold2LayerNorm(atom_config.hidden_size)

        if structure_prediction:
            self.coords_linear = nn.Linear(6, atom_config.hidden_size, bias=False)

        self.config = config
        self.rotary_emb = EsmFold2RotaryEmbedding(atom_config)
        self.layers = nn.ModuleList(
            [EsmFold2AtomLayer(config, atom_config) for _ in range(atom_config.num_hidden_layers)]
        )

        self.atom_to_token_linear = nn.Linear(atom_config.hidden_size, atom_config.output_dim, bias=False)

    def embed_atoms(self, atom_inputs: EsmFold2AtomInputs) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Per-atom base embedding and 3D-RoPE position embeddings.

        Noise-independent, so callers compute it once per fold outside the sampling loop.
        """
        ref_pos = atom_inputs.ref_pos
        batch_size, num_atoms = ref_pos.shape[:2]
        atom_features = torch.cat(
            [
                ref_pos,
                atom_inputs.ref_charge.unsqueeze(-1),
                atom_inputs.atom_attention_mask.unsqueeze(-1),
                atom_inputs.ref_element,
                atom_inputs.ref_atom_name_chars.reshape(batch_size, num_atoms, self.char_feature_dim),
            ],
            dim=-1,
        )
        # ``atom_features`` is fp32 (one-hots and masks), so the downcast into the projection is real.
        atom_embeds = self.atom_norm(self.atom_linear(atom_features.to(self.atom_linear.weight.dtype)))
        return atom_embeds, self.rotary_emb(ref_pos, atom_inputs.ref_space_uid, dtype=atom_embeds.dtype)

    def build_attention_mask(self, atom_mask: Tensor, atom_embeds: Tensor) -> Tensor:
        """Symmetric sliding-window attention mask over atom index.

        Noise-independent, so callers build it once per fold outside the sampling loop.

        ``atom_embeds`` (the ``embed_atoms`` output) only supplies the mask metadata
        (dtype/device/shape). ``atom_mask`` is the boolean valid-atom mask, passed as the standard 2D
        ``attention_mask``. Note ``config.sliding_window`` is the *total* window width here, while the
        overlay takes the (inclusive) radius, hence the halving.
        """
        return create_bidirectional_mask(
            config=self.config,
            inputs_embeds=atom_embeds,
            attention_mask=atom_mask,
            and_mask_function=sliding_window_bidirectional_overlay(self.config.sliding_window // 2),
        )

    def forward(
        self,
        atom_embeds: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        atom_mask: Tensor,
        atom_to_token: Tensor,
        num_tokens: int,
        atom_coords: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns ``(token_hidden_states, hidden_states, atom_conditioning)``. Every argument is already at the
        caller's batch size (for the diffusion stack, ``batch_size * num_diffusion_samples``).
        """
        atom_conditioning = atom_embeds
        hidden_states = atom_conditioning

        if self.structure_prediction:
            # The second coord slot is unused in this release, so coords_linear sees [atom_coords, 0].
            coord_input = torch.cat([atom_coords, torch.zeros_like(atom_coords)], dim=-1)
            coord_embeds = self.coords_linear(coord_input.to(self.coords_linear.weight.dtype))
            hidden_states = hidden_states + coord_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states, atom_conditioning, attention_mask, position_embeddings)

        token_features = F.relu(self.atom_to_token_linear(hidden_states))
        token_hidden_states = scatter_atom_to_token(
            token_features,
            atom_to_token,
            num_tokens,
            atom_mask=atom_mask,
        )

        return token_hidden_states, hidden_states, atom_conditioning


def _gather_along_dim1(source: Tensor, index: Tensor) -> Tensor:
    """Gather ``source`` (``[B, N, d]``) along dim 1 with a ``[B, M]`` index, returning ``[B, M, d]``."""
    idx = index.unsqueeze(-1).expand(-1, -1, source.size(-1))
    return torch.gather(source, 1, idx)


def _expand_samples(num_diffusion_samples: int, *tensors: Tensor) -> tuple[Tensor, ...]:
    """Repeat each tensor along its batch axis onto the sampler's replica batch.

    ``num_diffusion_samples`` is a replica axis, not a batch axis: the diffusion module and the
    confidence head both run at ``batch_size * num_diffusion_samples``, so every per-fold tensor they
    consume has to be interleaved to match.
    """
    return tuple(tensor.repeat_interleave(num_diffusion_samples, 0) for tensor in tensors)


class EsmFold2AtomDecoder(nn.Module):
    """SWA atom decoder. Only used inside the diffusion module, so its atom dims are always the
    structure-prediction (diffusion) ones.
    """

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        diffusion_config = config.structure_head.diffusion_module
        atom_config = diffusion_config.atom_encoder
        self.token_to_atom_linear = nn.Linear(diffusion_config.hidden_size, atom_config.hidden_size, bias=False)

        self.layers = nn.ModuleList(
            [EsmFold2AtomLayer(config, atom_config) for _ in range(atom_config.num_hidden_layers)]
        )

        self.norm = EsmFold2LayerNorm(atom_config.hidden_size)
        self.output_linear = nn.Linear(atom_config.hidden_size, 3, bias=False)  # (x, y, z) coordinates

    def forward(
        self,
        token_hidden_states: Tensor,
        hidden_states: Tensor,
        atom_conditioning: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        atom_to_token: Tensor,
    ) -> Tensor:
        """Returns coord_update."""
        token_to_atom_features = self.token_to_atom_linear(token_hidden_states)
        token_to_atom_features = _gather_along_dim1(token_to_atom_features, atom_to_token)
        hidden_states = hidden_states + token_to_atom_features

        for layer in self.layers:
            hidden_states = layer(hidden_states, atom_conditioning, attention_mask, position_embeddings)

        atom_coords = self.output_linear(self.norm(hidden_states))
        return atom_coords


class EsmFold2AttentionPairBias(nn.Module):
    """Gated multi-head attention with pair bias conditioning."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.config = config
        diffusion_config = config.structure_head.diffusion_module
        self.head_dim = diffusion_config.head_dim
        self.scaling = self.head_dim**-0.5
        # No grouped-query attention; identity repeat keeps the attention interface happy.
        self.num_key_value_groups = 1
        self.is_causal = False

        self.adaln = EsmFold2AdaptiveLayerNorm(diffusion_config)
        self.out_gate = nn.Linear(diffusion_config.hidden_size, diffusion_config.hidden_size, bias=True)

        self.q_proj = nn.Linear(diffusion_config.hidden_size, diffusion_config.hidden_size, bias=True)
        self.k_proj = nn.Linear(diffusion_config.hidden_size, diffusion_config.hidden_size, bias=False)
        self.v_proj = nn.Linear(diffusion_config.hidden_size, diffusion_config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(diffusion_config.hidden_size, diffusion_config.hidden_size, bias=False)
        self.o_proj = nn.Linear(diffusion_config.hidden_size, diffusion_config.hidden_size, bias=False)

        self.pair_norm = EsmFold2LayerNorm(config.pairwise_hidden_size)
        self.pair_bias_proj = nn.Linear(config.pairwise_hidden_size, diffusion_config.num_attention_heads, bias=False)

    def compute_pair_bias(self, pair_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """This block's additive attention bias, ``[batch_size, num_attention_heads, num_queries, num_keys]``:
        the per-head projection of the normed pair representation with token padding folded in.

        Step-invariant, so it is built once per fold outside the sampling loop and ``forward`` never
        sees a mask.
        """
        pair_bias = self.pair_bias_proj(self.pair_norm(pair_states))
        attention_bias = pair_bias.permute(0, 3, 1, 2)  # [B, Q, K, H] -> [B, H, Q, K]
        if attention_mask is not None:
            # Built at the bias dtype: python scalars would promote the whole bias to fp32.
            attention_bias = attention_bias + torch.where(
                attention_mask[:, None, None, :],
                attention_bias.new_zeros(()),
                attention_bias.new_full((), torch.finfo(attention_bias.dtype).min),
            )
        return attention_bias

    def forward(
        self,
        token_hidden_states: Tensor,
        single_states: Tensor,
        attention_bias: Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[Tensor, Tensor | None]:
        hidden_states = self.adaln(token_hidden_states, single_states)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        gate = torch.sigmoid(self.gate_proj(hidden_states)).view(hidden_shape)

        # The step-invariant per-head bias doubles as the additive attention mask. Returns [B, Q, H, D].
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_bias,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = gate * attn_output
        attn_output = self.o_proj(attn_output.reshape(*input_shape, -1))
        return torch.sigmoid(self.out_gate(single_states)) * attn_output, attn_weights


class EsmFold2ConditionedTransition(nn.Module):
    """Conditioned EsmFold2SwiGLU transition with adaptive layer norm."""

    def __init__(self, config: EsmFold2DiffusionModuleConfig) -> None:
        super().__init__()
        self.adaln = EsmFold2AdaptiveLayerNorm(config)
        self.output_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.mlp = EsmFold2SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, token_hidden_states: Tensor, single_states: Tensor) -> Tensor:
        hidden_states = self.adaln(token_hidden_states, single_states)
        hidden_states = self.mlp(hidden_states)
        return torch.sigmoid(self.output_gate(single_states)) * hidden_states


class EsmFold2DiffusionTransformer(nn.Module):
    """Diffusion denoising transformer with attention pair bias."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        # The pair-bias attention also needs trunk-level widths, so it keeps the full config; the
        # transition is entirely described by the diffusion sub-config and takes only that.
        diffusion_config = config.structure_head.diffusion_module
        self.attention_layers = nn.ModuleList(
            [EsmFold2AttentionPairBias(config) for _ in range(diffusion_config.num_hidden_layers)]
        )
        self.transition_layers = nn.ModuleList(
            [EsmFold2ConditionedTransition(diffusion_config) for _ in range(diffusion_config.num_hidden_layers)]
        )

    def compute_pair_biases(self, pair_states: Tensor, attention_mask: Tensor | None = None) -> list[Tensor]:
        """Per-block additive attention biases, built once per fold outside the sampling loop; see
        ``EsmFold2AttentionPairBias.compute_pair_bias``."""
        return [attn.compute_pair_bias(pair_states, attention_mask) for attn in self.attention_layers]

    def forward(self, token_hidden_states: Tensor, single_states: Tensor, attention_biases: list[Tensor]) -> Tensor:
        hidden_states = token_hidden_states
        for attn, transition, attention_bias in zip(self.attention_layers, self.transition_layers, attention_biases):
            attn_output, _ = attn(hidden_states, single_states, attention_bias)
            hidden_states = hidden_states + attn_output
            hidden_states = hidden_states + transition(hidden_states, single_states)
        return hidden_states


class EsmFold2DiffusionConditioning(nn.Module):
    """Conditions pair and single representations on noise timestep."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        diffusion_config = config.structure_head.diffusion_module
        self.sigma_data = diffusion_config.sigma_data

        self.pair_input_norm = EsmFold2LayerNorm(2 * config.pairwise_hidden_size)
        self.pair_proj = nn.Linear(2 * config.pairwise_hidden_size, config.pairwise_hidden_size, bias=False)
        self.pair_transitions = nn.ModuleList(
            [
                EsmFold2Transition(
                    config.pairwise_hidden_size,
                    diffusion_config.pair_intermediate_size,
                    chunk_size=None,
                )
                for _ in range(2)
            ]
        )

        self.single_input_norm = EsmFold2LayerNorm(config.single_inputs_size)
        self.single_proj = nn.Linear(config.single_inputs_size, diffusion_config.hidden_size, bias=False)
        self.fourier = EsmFold2FourierEmbedding(diffusion_config.fourier_dim)
        self.noise_norm = EsmFold2LayerNorm(diffusion_config.fourier_dim)
        self.noise_proj = nn.Linear(diffusion_config.fourier_dim, diffusion_config.hidden_size, bias=False)
        self.single_transitions = nn.ModuleList(
            [
                EsmFold2Transition(
                    diffusion_config.hidden_size,
                    diffusion_config.intermediate_size,
                    chunk_size=None,
                )
                for _ in range(2)
            ]
        )

    def compute_pair_repr(self, pair_trunk: Tensor, relative_position_encoding: Tensor) -> Tensor:
        """The pair half of the conditioning. Noise-independent, so it is built once per fold outside
        the sampling loop; ``forward`` owns the noise-dependent half."""
        # ``pair_trunk`` is fp32, so the concat and the norm stay fp32; z_proj downcasts.
        pair_states = torch.cat([pair_trunk, relative_position_encoding], dim=-1)
        pair_states = self.pair_proj(self.pair_input_norm(pair_states).to(self.pair_proj.weight.dtype))
        for block in self.pair_transitions:
            pair_states = block(pair_states)
        return pair_states

    def compute_single_repr(self, single_inputs: Tensor) -> Tensor:
        """Project the single-inputs tensor into the diffusion token width.

        Noise-independent, so it is built once per fold outside the sampling loop; only the Fourier
        noise embedding ``forward`` adds to it varies per step.
        """
        return self.single_proj(self.single_input_norm(single_inputs))

    def forward(self, noise_level: Tensor, single_states: Tensor) -> Tensor:
        """The noise-dependent half of the single conditioning: add the Fourier noise embedding to the
        precomputed ``single_states`` and run the transitions."""
        log_noise_level = 0.25 * torch.log((noise_level / self.sigma_data).clamp(min=1e-20))
        noise_embeds = self.fourier(log_noise_level)
        noise_embeds = self.noise_proj(self.noise_norm(noise_embeds).to(self.noise_proj.weight.dtype))
        single_states = single_states + noise_embeds.unsqueeze(1)

        for block in self.single_transitions:
            single_states = block(single_states)

        return single_states


class EsmFold2DiffusionModule(nn.Module):
    """Diffusion denoising module for structure prediction."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        # Keeps the full config: every submodule below reads trunk-level fields too.
        diffusion_config = config.structure_head.diffusion_module
        self.sigma_data = diffusion_config.sigma_data

        self.conditioning = EsmFold2DiffusionConditioning(config)
        self.atom_encoder = EsmFold2AtomEncoder(config, structure_prediction=True)
        self.atom_decoder = EsmFold2AtomDecoder(config)
        self.single_to_token = nn.Linear(diffusion_config.hidden_size, diffusion_config.hidden_size, bias=False)
        self.token_transformer = EsmFold2DiffusionTransformer(config)
        self.single_step_norm = EsmFold2LayerNorm(diffusion_config.hidden_size)
        self.token_norm = EsmFold2LayerNorm(diffusion_config.hidden_size)

    def prepare_conditioning(
        self,
        atom_inputs: EsmFold2AtomInputs,
        pair_trunk: Tensor,
        relative_position_encoding: Tensor,
        single_inputs: Tensor,
        token_attention_mask: Tensor | None = None,
        num_diffusion_samples: int = 1,
    ) -> EsmFold2DenoiserConditioning:
        """Precompute the conditioning that every denoising step reuses.

        Order matters: the featurization and the projections run at the unexpanded batch size and only
        their (much narrower) results are expanded.

        ``num_diffusion_samples`` is a replica axis rather than a batch axis -- every tensor built here
        is identical across samples, while the sampler runs the denoiser at
        ``batch_size * num_diffusion_samples``. The two attention masks are only ever broadcast
        against, so they stay at the unexpanded batch size whenever that broadcasts over the sample
        batch (``batch_size == 1``, i.e. every single-sequence fold) and are materialized only when it
        cannot. They are also the only large tensors here: at length 1000 the per-block token biases
        are ~366 MiB unexpanded, against ~11.4 GiB expanded across the 32 samples the released
        checkpoints ask for (the class default is 8). Everything else is a few MB, and expanding it
        keeps the scatter/gather helpers on a single shape.
        """
        samples = num_diffusion_samples
        # A batch dim of 1 broadcasts over the sample batch; anything else has to be materialized.
        masks_broadcastable = single_inputs.shape[0] == 1

        # --- precomputed at batch_size ---
        atom_embeds, position_embeddings = self.atom_encoder.embed_atoms(atom_inputs)
        attention_mask = self.atom_encoder.build_attention_mask(atom_inputs.atom_attention_mask, atom_embeds)
        pair_states = self.conditioning.compute_pair_repr(pair_trunk, relative_position_encoding)
        token_attention_bias = self.token_transformer.compute_pair_biases(pair_states, token_attention_mask)
        if not masks_broadcastable:
            attention_mask, *token_attention_bias = _expand_samples(samples, attention_mask, *token_attention_bias)

        # --- everything else: batch_size -> batch_size * num_diffusion_samples ---
        cos, sin = position_embeddings
        atom_embeds, cos, sin, atom_mask, atom_to_token, expanded_single_inputs = _expand_samples(
            samples,
            atom_embeds,
            cos,
            sin,
            atom_inputs.atom_attention_mask,
            atom_inputs.atom_to_token,
            single_inputs,
        )
        return EsmFold2DenoiserConditioning(
            atom_embeds=atom_embeds,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
            atom_mask=atom_mask,
            atom_to_token=atom_to_token,
            # Projected after expanding, unlike the pair bias: a Linear reassociates per batch size.
            projected_single_inputs=self.conditioning.compute_single_repr(expanded_single_inputs),
            token_attention_bias=token_attention_bias,
        )

    def forward(
        self,
        noisy_coords: Tensor,
        noise_level: Tensor,
        conditioning: EsmFold2DenoiserConditioning,
    ) -> Tensor:
        # ``noise_level`` is the sampler's per-sample noise level: fp32, flat, at the expanded batch length.
        sigma_data = self.sigma_data

        # Step 1: noise-dependent (single) conditioning
        single_states = self.conditioning(noise_level=noise_level, single_states=conditioning.projected_single_inputs)

        # Step 2: normalize noisy coords
        denominator = torch.sqrt(noise_level * noise_level + sigma_data * sigma_data)
        normalized_coords = noisy_coords / denominator[:, None, None]

        # Step 3: atom encoder
        token_hidden_states, atom_hidden_states, atom_conditioning = self.atom_encoder(
            atom_embeds=conditioning.atom_embeds,
            attention_mask=conditioning.attention_mask,
            position_embeddings=conditioning.position_embeddings,
            atom_mask=conditioning.atom_mask,
            atom_to_token=conditioning.atom_to_token,
            num_tokens=single_states.shape[1],
            atom_coords=normalized_coords,
        )

        # Step 4: add conditioned single repr
        token_hidden_states = token_hidden_states + self.single_to_token(self.single_step_norm(single_states))

        # Step 5: token transformer
        token_hidden_states = self.token_transformer(
            token_hidden_states, single_states, conditioning.token_attention_bias
        )

        # Step 6: token norm
        token_hidden_states = self.token_norm(token_hidden_states)

        # Step 7: atom decoder
        coord_update = self.atom_decoder(
            token_hidden_states=token_hidden_states,
            hidden_states=atom_hidden_states,
            atom_conditioning=atom_conditioning,
            attention_mask=conditioning.attention_mask,
            position_embeddings=conditioning.position_embeddings,
            atom_to_token=conditioning.atom_to_token,
        )

        # Step 8: denoised output; the fp32 noise-level scalars keep the coordinates fp32.
        sigma_data_sq = sigma_data * sigma_data
        noise_level_sq = noise_level * noise_level
        denoised_coords = (sigma_data_sq / (sigma_data_sq + noise_level_sq))[:, None, None] * noisy_coords
        denoised_coords = (
            denoised_coords
            + ((sigma_data * noise_level) / torch.sqrt(sigma_data_sq + noise_level_sq))[:, None, None] * coord_update
        )

        return denoised_coords


class EsmFold2RowAttentionPooling(nn.Module):
    """Row-wise attention pooling: attn_proj, out_proj."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(config.pairwise_hidden_size, 1, bias=False)
        self.out_proj = nn.Linear(config.pairwise_hidden_size, config.hidden_size, bias=False)

    def forward(self, pair_states: Tensor, attention_mask: Tensor) -> Tensor:
        scores = self.attn_proj(pair_states).squeeze(-1)
        # Mask the pooled-over (key) axis directly instead of adding a separate bias tensor: same
        # softmax, without three extra `[batch, num_tokens, num_tokens]` temporaries.
        scores = scores.masked_fill(~attention_mask[:, None, :], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        pooled = torch.einsum("bnm,bnmd->bnd", weights, pair_states)
        return self.out_proj(pooled)


def _relative_position_one_hot(diff: Tensor, num_bins: int, keep_mask: Tensor) -> Tensor:
    """One-hot encode a relative index difference into ``2 * num_bins + 2`` classes: the clipped offset,
    plus a final "out-of-context" bin wherever ``keep_mask`` is False (e.g. a cross-chain pair).
    """
    binned = torch.clip(diff + num_bins, 0, 2 * num_bins)
    binned = torch.where(keep_mask, binned, 2 * num_bins + 1)
    return F.one_hot(binned, 2 * num_bins + 2)


class EsmFold2RelativePositionEncoding(nn.Module):
    """Pair encoding of relative residue index, token index, chain (sym_id) offset and same-entity.

    ``embed.weight`` is ``[pairwise_hidden_size, num_features]`` where ``num_features =
    2*(2*residue_bins+2) + 1 + (2*chain_bins+2)``. For the defaults residue_bins=32, chain_bins=2:
    2*66 + 1 + 6 = 139.
    """

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.num_relative_residx_bins = config.num_relative_residx_bins
        self.num_relative_chain_bins = config.num_relative_chain_bins

        num_residue_features = 2 * self.num_relative_residx_bins + 2
        num_token_features = 2 * self.num_relative_residx_bins + 2
        num_chain_features = 2 * self.num_relative_chain_bins + 2
        num_same_entity_features = 1
        num_features = num_residue_features + num_token_features + num_chain_features + num_same_entity_features
        self.embed = nn.Linear(num_features, config.pairwise_hidden_size, bias=False)

    def forward(
        self,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        token_index: Tensor,
    ) -> Tensor:
        same_chain = asym_id.unsqueeze(2) == asym_id.unsqueeze(1)
        same_residue = residue_index.unsqueeze(2) == residue_index.unsqueeze(1)
        same_entity = entity_id.unsqueeze(2) == entity_id.unsqueeze(1)

        # Residue, token and chain offsets; the last keeps cross-chain pairs, so its mask is inverted.
        residue_bins, chain_bins = self.num_relative_residx_bins, self.num_relative_chain_bins
        relative_residue_one_hot = _relative_position_one_hot(
            residue_index.unsqueeze(2) - residue_index.unsqueeze(1), residue_bins, same_chain
        )
        relative_token_one_hot = _relative_position_one_hot(
            token_index.unsqueeze(2) - token_index.unsqueeze(1), residue_bins, same_chain & same_residue
        )
        relative_chain_one_hot = _relative_position_one_hot(
            sym_id.unsqueeze(2) - sym_id.unsqueeze(1), chain_bins, ~same_chain
        )

        # Cast the 0/1 one-hots straight to the projection dtype: exact, and skips a large fp32 tensor.
        dtype = self.embed.weight.dtype
        features = torch.cat(
            [
                relative_residue_one_hot.to(dtype),
                relative_token_one_hot.to(dtype),
                same_entity.to(dtype).unsqueeze(-1),
                relative_chain_one_hot.to(dtype),
            ],
            dim=-1,
        )

        return self.embed(features)


class EsmFold2SingleToPair(nn.Module):
    """downproject -> outer product/difference -> two-layer MLP, all at ``pairwise_hidden_size``."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.downproject = nn.Linear(config.pairwise_hidden_size, config.pairwise_hidden_size)
        self.output_fc1 = nn.Linear(2 * config.pairwise_hidden_size, config.pairwise_hidden_size)
        self.output_fc2 = nn.Linear(config.pairwise_hidden_size, config.pairwise_hidden_size)

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


class EsmFold2LanguageModelEncoder(nn.Module):
    """Projects the ESMC hidden states into a pair representation, learning a per-layer mixture
    (``base_z_combine``) over them."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()

        self.single_to_pair = EsmFold2SingleToPair(config)
        self.pair_output_norm = EsmFold2LayerNorm(config.pairwise_hidden_size)
        self.pair_input_norm = EsmFold2LayerNorm(config.esmc_config.hidden_size)
        self.pair_proj = nn.Linear(config.esmc_config.hidden_size, config.pairwise_hidden_size, bias=False)
        self.layer_weights = nn.Parameter(torch.zeros(config.esmc_config.num_hidden_layers + 1))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project ESMC hidden states ``[batch_size, num_tokens, num_layers + 1, lm_hidden_size]`` to a
        ``[batch_size, num_tokens, num_tokens, pairwise_hidden_size]``
        pair representation."""
        hidden_states = hidden_states.to(self.pair_proj.weight.dtype)
        normed = self.pair_input_norm(hidden_states)
        lm_pair_states = self.pair_proj(normed)  # [batch_size, num_tokens, num_layers + 1, pairwise_hidden_size]
        weights = self.layer_weights.softmax(0)
        lm_pair_states = (weights @ lm_pair_states).squeeze(-2)  # [batch_size, num_tokens, pairwise_hidden_size]
        pair_states = self.single_to_pair(lm_pair_states)
        lm_pair_states = self.pair_output_norm(
            pair_states
        )  # [batch_size, num_tokens, num_tokens, pairwise_hidden_size]
        return lm_pair_states


@use_kernel_forward_from_hub("EsmFold2TriangleMultiplication")
class EsmFold2TriangleMultiplicativeUpdate(nn.Module):
    """Triangle multiplicative update with gated signal routing and explicit orientation.

    The O(N^3) contraction is the trunk's dominant cost, so ``use_kernels=True`` swaps this forward
    for a fused Triton Hub kernel (CUDA, inference only).
    """

    def __init__(self, config: EsmFold2Config, outgoing: bool = True) -> None:
        super().__init__()
        self.dim = config.pairwise_hidden_size
        self.flow = "outgoing" if outgoing else "incoming"
        self.norm_start = EsmFold2LayerNorm(self.dim)
        self.norm_mix = EsmFold2LayerNorm(self.dim)
        self.proj_bundle = nn.Linear(self.dim, 4 * self.dim, bias=False)
        self.proj_emit = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_gate = nn.Linear(self.dim, self.dim, bias=False)

        # Chunk the O(N^3) contraction for memory on long sequences.
        self.chunk_size: int | None = config.chunk_size

    def _triangular_contract(self, left_stream: Tensor, right_stream: Tensor) -> Tensor:
        """Triangular einsum, chunked along the output i-dimension (a falsy ``chunk_size`` means one
        full-length chunk, i.e. unchunked)."""
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
        # Cast the fp32 pair mask, so masking does not promote the O(N^3) contraction to fp32.
        routed = routed * visibility.unsqueeze(-1).to(routed.dtype)

        left_stream, right_stream = routed.chunk(2, dim=-1)
        contracted = self._triangular_contract(left_stream, right_stream)
        mixed = self.proj_emit(self.norm_mix(contracted))
        output_gate = torch.sigmoid(self.proj_gate(normalized_grid))
        return mixed * output_gate


class EsmFold2PairUpdateLayer(GradientCheckpointingLayer):
    """tri_mul_out, tri_mul_in, pair_transition."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.tri_mul_out = EsmFold2TriangleMultiplicativeUpdate(config, outgoing=True)
        self.tri_mul_in = EsmFold2TriangleMultiplicativeUpdate(config, outgoing=False)
        self.pair_transition = EsmFold2Transition(
            config.pairwise_hidden_size, config.pair_transition_intermediate_size, config.chunk_size
        )

    def forward(self, pair_states: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        # Inference-only: trained row-shared dropout omitted.
        pair_states = pair_states + self.tri_mul_out(pair_states, visibility=pair_attention_mask)
        pair_states = pair_states + self.tri_mul_in(pair_states, visibility=pair_attention_mask)
        pair_states = self.pair_transition(pair_states)
        return pair_states


class EsmFold2PairUpdateStack(nn.Module):
    """A stack of ``num_layers`` `EsmFold2PairUpdateLayer` layers refining the pair representation."""

    def __init__(self, config: EsmFold2Config, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EsmFold2PairUpdateLayer(config) for _ in range(num_layers)])

    def forward(self, pair_states: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            pair_states = layer(pair_states, pair_attention_mask=pair_attention_mask)
        return pair_states


class EsmFold2OuterProductMean(nn.Module):
    """Outer-product mean: maps an MSA representation into a pair update.

    ``divide_outer_before_proj`` selects ``Wout(outer / num_valid_pairs)`` over ``Wout(outer) / num_valid_pairs``,
    as different released checkpoints were trained with different orderings.
    """

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.outer_hidden_size = config.msa_encoder.outer_hidden_size
        self.divide_outer_before_proj = config.msa_encoder.divide_outer_before_proj
        self.norm = EsmFold2LayerNorm(config.msa_encoder.hidden_size)
        self.input_proj = nn.Linear(
            config.msa_encoder.hidden_size, 2 * config.msa_encoder.outer_hidden_size, bias=False
        )
        self.output_proj = nn.Linear(
            config.msa_encoder.outer_hidden_size * config.msa_encoder.outer_hidden_size,
            config.pairwise_hidden_size,
            bias=True,
        )
        # Its own chunk size, off by default: chunking this einsum is not always bit-exact in bf16.
        self.chunk_size: int | None = config.msa_encoder.outer_product_chunk_size

    def forward(self, msa_states: Tensor, msa_attention_mask: Tensor) -> Tensor:
        msa_normed = self.norm(msa_states)
        projected_msa = self.input_proj(msa_normed) * msa_attention_mask.unsqueeze(-1).to(msa_normed.dtype)
        left, right = projected_msa.chunk(2, dim=-1)
        mask_float = msa_attention_mask.to(left.dtype)
        num_valid_pairs = (mask_float @ mask_float.transpose(-1, -2)).unsqueeze(-1).clamp(min=1.0)
        # Chunk along the left (i) axis, shrinking the peak intermediate to [B, chunk, L, c, d].
        seq_len = left.shape[1]
        chunk_size = self.chunk_size or seq_len
        out_chunks: list[Tensor] = []
        for start in range(0, seq_len, chunk_size):
            window = slice(start, start + chunk_size)
            outer = torch.einsum("bimc,bjmd->bijcd", left[:, window], right).flatten(-2)
            if self.divide_outer_before_proj:
                out_chunks.append(self.output_proj(outer / num_valid_pairs[:, window]))
            else:
                out_chunks.append(self.output_proj(outer) / num_valid_pairs[:, window])
        return torch.cat(out_chunks, dim=1)


class EsmFold2MSAPairWeightedAveraging(nn.Module):
    """Pair-biased MSA row update (AF3 Supplement Algorithm 10)."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.num_attention_heads = config.msa_encoder.num_attention_heads
        self.head_dim = config.msa_encoder.head_dim
        inner_dim = self.num_attention_heads * self.head_dim
        self.norm_single = EsmFold2LayerNorm(config.msa_encoder.hidden_size)
        self.bias_norm = EsmFold2LayerNorm(config.pairwise_hidden_size)
        self.bias_proj = nn.Linear(config.pairwise_hidden_size, self.num_attention_heads, bias=False)
        self.v_proj = nn.Linear(config.msa_encoder.hidden_size, inner_dim, bias=False)
        self.gate_proj = nn.Linear(config.msa_encoder.hidden_size, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, config.msa_encoder.hidden_size, bias=False)

    def forward(self, msa_states: Tensor, pair_states: Tensor, pair_attention_mask: Tensor) -> Tensor:
        """
        Args:
            msa_states:          [batch_size, num_tokens, msa_depth, msa_hidden_size]
            pair_states:         [batch_size, num_tokens, num_tokens, pairwise_hidden_size]
            pair_attention_mask: [batch_size, num_tokens, num_tokens]
        Returns:
            [batch_size, num_tokens, msa_depth, msa_hidden_size]
        """
        batch_size, seq_len, msa_depth, _ = msa_states.shape
        hidden_shape = (batch_size, seq_len, msa_depth, self.num_attention_heads, self.head_dim)

        msa_normed = self.norm_single(msa_states)

        # Not an `ALL_ATTENTION_FUNCTIONS` call site: there are no query/key projections to make one
        # out of. The per-head weights come straight from the pair representation, and the values are
        # indexed by MSA row as well as by token, so the attended axis is neither of the usual two.
        attention_bias = self.bias_proj(self.bias_norm(pair_states))  # [B, L, L, num_attention_heads]
        attention_bias.masked_fill_(~pair_attention_mask.unsqueeze(-1), torch.finfo(attention_bias.dtype).min)
        # Softmax over the attended token axis j, not over the trailing axis.
        attn_weights = torch.softmax(attention_bias, dim=-2, dtype=torch.float32).to(attention_bias.dtype)

        value_states = self.v_proj(msa_normed).reshape(hidden_shape)
        gate = torch.sigmoid(self.gate_proj(msa_normed)).reshape(hidden_shape)

        attn_output = torch.einsum("bijh,bjmhd->bimhd", attn_weights, value_states)
        attn_output = gate * attn_output
        return self.o_proj(attn_output.reshape(batch_size, seq_len, msa_depth, -1))


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
    Output of [`EsmFold2Model.forward`]: the folding trunk's pair representation, the distogram read
    off it, and the conditioning tensors that the structure and confidence heads consume. Everything
    here is deterministic given the inputs apart from the trunk's random initial pair state.

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
    """Local atom index within each token, from the ``[B, A]`` atom->token map.

    A token's atoms are contiguous, so this is a running count that resets at each token boundary.
    """
    same_as_prev = F.pad(atom_to_token[:, 1:] == atom_to_token[:, :-1], (1, 0), value=False)
    ones = torch.ones_like(atom_to_token)
    cumsum = torch.cumsum(ones, dim=-1)
    group_start = cumsum.masked_fill(same_as_prev, 0)
    group_start = torch.cummax(group_start, dim=-1).values
    return cumsum - group_start


def _categorical_mean(logits: Tensor, start: float, end: float) -> Tensor:
    """Expected value of a categorical distribution over ``logits.shape[-1]`` evenly-spaced bins
    spanning ``[start, end]``."""
    num_bins = logits.shape[-1]
    edges = torch.linspace(start, end, num_bins + 1, device=logits.device, dtype=torch.float32)
    bin_centers = (edges[:-1] + edges[1:]) / 2  # [num_bins]
    return (logits.float().softmax(-1) @ bin_centers.unsqueeze(1)).squeeze(-1)


class EsmFold2ConfidenceInputEmbedder(nn.Module):
    """Builds the confidence head's base pair representation from the trunk pair representation and
    the single-inputs tensor (input norms + single->pair projections, including the outer product)."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.single_inputs_norm = EsmFold2LayerNorm(config.single_inputs_size)
        self.pair_norm = EsmFold2LayerNorm(config.pairwise_hidden_size)
        self.single_to_pair = nn.Linear(config.single_inputs_size, config.pairwise_hidden_size, bias=False)
        self.single_to_pair_transpose = nn.Linear(config.single_inputs_size, config.pairwise_hidden_size, bias=False)
        self.single_to_pair_prod_in1 = nn.Linear(config.single_inputs_size, config.pairwise_hidden_size, bias=False)
        self.single_to_pair_prod_in2 = nn.Linear(config.single_inputs_size, config.pairwise_hidden_size, bias=False)
        self.single_to_pair_prod_out = nn.Linear(config.pairwise_hidden_size, config.pairwise_hidden_size, bias=False)

    def forward(
        self,
        single_inputs: Tensor,
        pair_states: Tensor,
        relative_position_encoding: Tensor | None,
        token_bonds_encoding: Tensor | None,
    ) -> Tensor:
        single_inputs_normed = self.single_inputs_norm(single_inputs)

        pair_states = self.pair_norm(pair_states)
        if relative_position_encoding is not None:
            pair_states = pair_states + relative_position_encoding
        if token_bonds_encoding is not None:
            pair_states = pair_states + token_bonds_encoding
        pair_states = pair_states + self.single_to_pair(single_inputs_normed).unsqueeze(2)
        pair_states = pair_states + self.single_to_pair_transpose(single_inputs_normed).unsqueeze(1)
        pair_states = pair_states + self.single_to_pair_prod_out(
            self.single_to_pair_prod_in1(single_inputs_normed)[:, :, None, :]
            * self.single_to_pair_prod_in2(single_inputs_normed)[:, None, :, :]
        )
        return pair_states


class EsmFold2ConfidenceHead(nn.Module):
    """Predicts pLDDT, PAE, PDE, resolved-atom probability and distogram bins."""

    boundaries: Tensor

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.eps = config.confidence_head.eps

        boundaries = torch.linspace(
            config.confidence_head.min_dist, config.confidence_head.max_dist, config.confidence_head.distogram_bins - 1
        )
        self.register_buffer("boundaries", boundaries)
        self.dist_bin_pairwise_embed = nn.Embedding(config.confidence_head.distogram_bins, config.pairwise_hidden_size)

        self.input_embedder = EsmFold2ConfidenceInputEmbedder(config)

        self.row_attention_pooling = EsmFold2RowAttentionPooling(config)

        self.folding_trunk = EsmFold2PairUpdateStack(config, config.confidence_head.num_hidden_layers)

        # Heads.
        self.plddt_layernorm = EsmFold2LayerNorm(config.hidden_size)
        self.plddt_weight = nn.Parameter(
            torch.zeros(config.max_atoms_per_token, config.hidden_size, config.confidence_head.num_plddt_bins)
        )

        self.pae_layernorm = EsmFold2LayerNorm(config.pairwise_hidden_size)
        self.pae_head = nn.Linear(config.pairwise_hidden_size, config.confidence_head.num_pae_bins, bias=False)

        self.pde_layernorm = EsmFold2LayerNorm(config.pairwise_hidden_size)
        self.pde_head = nn.Linear(config.pairwise_hidden_size, config.confidence_head.num_pde_bins, bias=False)

        self.resolved_layernorm = EsmFold2LayerNorm(config.hidden_size)
        # 2 = resolved logits ([unresolved, resolved]).
        self.resolved_weight = nn.Parameter(torch.zeros(config.max_atoms_per_token, config.hidden_size, 2))

    def _build_pair_and_single(
        self,
        single_inputs: Tensor,
        pair_states: Tensor,
        predicted_coords: Tensor,
        distogram_atom_idx: Tensor,
        token_attention_mask: Tensor,
        atom_to_token: Tensor,
        atom_attention_mask: Tensor,
        num_diffusion_samples: int,
        relative_position_encoding: Tensor | None,
        token_bonds_encoding: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """Build the per-sample pair + single representations shared by every confidence head.

        Returns ``(single_states, pair_states, token_mask, rep_distances, rep_idx_expanded,
        atom_to_token_expanded, atom_mask_expanded, expanded_batch_size)``, where the ``*_expanded``
        tensors are repeated across the diffusion-sample batch axis.
        """
        pair_states = self.input_embedder(single_inputs, pair_states, relative_position_encoding, token_bonds_encoding)

        pair_states, atom_to_token_expanded, atom_mask_expanded, rep_idx_expanded, token_mask = _expand_samples(
            num_diffusion_samples,
            pair_states,
            atom_to_token,
            atom_attention_mask,
            distogram_atom_idx,
            token_attention_mask,
        )
        flat_predicted_coords = (
            predicted_coords.reshape(-1, *predicted_coords.shape[-2:])
            if predicted_coords.ndim == 4
            else predicted_coords
        )
        expanded_batch_size = pair_states.shape[0]

        rep_coords = _gather_along_dim1(flat_predicted_coords, rep_idx_expanded)
        rep_distances = torch.cdist(rep_coords, rep_coords, compute_mode="donot_use_mm_for_euclid_dist")
        distogram_bins = (rep_distances.unsqueeze(-1) > self.boundaries).sum(dim=-1)
        pair_states = pair_states + self.dist_bin_pairwise_embed(distogram_bins)

        pair_mask = token_mask[:, :, None].float() * token_mask[:, None, :].float()

        # ``pair_states`` is fp32; ``add_`` widens the compute-dtype delta without a second fp32 copy.
        pair_delta = self.folding_trunk(pair_states.to(self.pae_head.weight.dtype), pair_attention_mask=pair_mask)
        pair_states.add_(pair_delta)
        del pair_delta
        pair_states = pair_states.to(self.pae_head.weight.dtype)
        single_states = self.row_attention_pooling(pair_states, token_mask)

        return (
            single_states,
            pair_states,
            token_mask,
            rep_distances,
            rep_idx_expanded,
            atom_to_token_expanded,
            atom_mask_expanded,
            expanded_batch_size,
        )

    def _compute_atom_confidences(
        self,
        single_states: Tensor,
        atom_to_token_expanded: Tensor,
        atom_mask_expanded: Tensor,
        rep_idx_expanded: Tensor,
        rep_distances: Tensor,
        expanded_type: Tensor,
        expanded_asym: Tensor,
        expanded_batch_size: int,
    ) -> dict[str, Tensor]:
        """Per-atom confidence outputs off the single representation (pLDDT family + resolved)."""
        atom_mask_float = atom_mask_expanded.float()
        single_at_atoms = _gather_along_dim1(single_states, atom_to_token_expanded)
        normed_single_at_atoms = self.plddt_layernorm(single_at_atoms)

        intra_idx = _compute_intra_token_idx(atom_to_token_expanded)
        intra_idx = intra_idx.clamp(max=self.plddt_weight.shape[0] - 1)
        plddt_weights = self.plddt_weight[intra_idx]
        plddt_logits = torch.einsum("...c,...cb->...b", normed_single_at_atoms, plddt_weights)

        # The pLDDT-family metrics are reported summaries, not loss targets: only the logits train.
        with torch.no_grad():
            plddt_per_atom = _categorical_mean(plddt_logits, start=0.0, end=1.0)

            num_tokens = single_states.shape[1]
            plddt_sum = torch.zeros(
                expanded_batch_size, num_tokens, device=single_states.device, dtype=plddt_per_atom.dtype
            )
            atom_count = torch.zeros(
                expanded_batch_size, num_tokens, device=single_states.device, dtype=plddt_per_atom.dtype
            )
            # Both operands are fp32, as ``scatter_add_`` requires.
            plddt_sum.scatter_add_(1, atom_to_token_expanded, plddt_per_atom * atom_mask_float)
            atom_count.scatter_add_(1, atom_to_token_expanded, atom_mask_float)
            plddt = plddt_sum / atom_count.clamp(min=1e-6)

            complex_plddt = (plddt_per_atom * atom_mask_float).sum(dim=-1) / (atom_mask_float.sum(dim=-1) + self.eps)

            is_ligand = (expanded_type == 4).float()  # 4 = non-polymer (ligand) molecule type
            inter_chain = (expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)).float()
            near_contact = (rep_distances < 8).float()
            interface_per_token = (near_contact * inter_chain * (1.0 - is_ligand).unsqueeze(-1)).amax(dim=-1)
            iplddt_weight = torch.where(
                is_ligand.bool(),
                torch.full_like(interface_per_token, 2.0),
                interface_per_token,
            )
            iplddt_weight_atoms = _gather_along_dim1(iplddt_weight.unsqueeze(-1), atom_to_token_expanded).squeeze(-1)
            atom_iplddt_w = atom_mask_float * iplddt_weight_atoms
            complex_iplddt = (plddt_per_atom * atom_iplddt_w).sum(dim=-1) / (atom_iplddt_w.sum(dim=-1) + self.eps)

            plddt_ca = plddt_per_atom.gather(1, rep_idx_expanded)

        # Resolved (per-atom binary): same per-atom single features, its own weight.
        resolved_single_at_atoms = self.resolved_layernorm(single_at_atoms)
        resolved_weights = self.resolved_weight[intra_idx]
        resolved_logits = torch.einsum("...c,...cb->...b", resolved_single_at_atoms, resolved_weights)

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
        self, pae_logits: Tensor, token_mask: Tensor, expanded_asym: Tensor, expanded_batch_size: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """pTM / ipTM / per-chain-pair ipTM derived from the PAE logits (reported metrics only)."""
        num_bins = pae_logits.shape[-1]
        bin_width = 32.0 / num_bins
        bin_centers = torch.arange(0.5 * bin_width, 32.0, bin_width, device=pae_logits.device)
        mask_float = token_mask.float()
        num_residues = mask_float.sum(dim=-1, keepdim=True)
        d0 = 1.24 * (num_residues.clamp(min=19) - 15) ** (1 / 3) - 1.8
        tm_per_bin = 1 / (1 + (bin_centers / d0) ** 2)
        pae_probs = F.softmax(pae_logits, dim=-1, dtype=torch.float32)
        tm_expected = (pae_probs * tm_per_bin[:, None, None, :]).sum(dim=-1)

        pair_mask_2d = mask_float.unsqueeze(-1) * mask_float.unsqueeze(-2)
        ptm_per_row = (tm_expected * pair_mask_2d).sum(dim=-1) / (pair_mask_2d.sum(dim=-1) + self.eps)
        ptm = ptm_per_row.max(dim=-1).values

        inter_chain_mask = (expanded_asym.unsqueeze(-1) != expanded_asym.unsqueeze(-2)).float() * pair_mask_2d
        iptm_per_row = (tm_expected * inter_chain_mask).sum(dim=-1) / (inter_chain_mask.sum(dim=-1) + self.eps)
        iptm = iptm_per_row.max(dim=-1).values

        max_chain_id = int(expanded_asym.max().item()) if expanded_batch_size > 0 else 0
        num_chains = max_chain_id + 1
        pair_chains_iptm = torch.zeros(
            expanded_batch_size, num_chains, num_chains, device=tm_expected.device, dtype=tm_expected.dtype
        )
        # Max-of-row-mean per chain pair, as in the global iptm above, so iptm is the max off-diagonal.
        for chain_i in range(num_chains):
            chain_c1 = (expanded_asym == chain_i).float() * mask_float
            if chain_c1.sum() == 0:
                continue
            col_mask = chain_c1.unsqueeze(-2)
            avg_tm = (tm_expected * col_mask).sum(dim=-1) / (col_mask.sum(dim=-1) + self.eps)
            for chain_j in range(num_chains):
                chain_c2 = (expanded_asym == chain_j).float() * mask_float
                row_vals = avg_tm.masked_fill(chain_c2 == 0, float("-inf"))
                pair_chains_iptm[:, chain_i, chain_j] = row_vals.max(dim=-1).values.clamp(min=0.0)

        return ptm, iptm, pair_chains_iptm

    def forward(
        self,
        single_inputs: Tensor,
        pair_states: Tensor,
        predicted_coords: Tensor,
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
        (
            single_states,
            pair_states,
            token_mask,
            rep_distances,
            rep_idx_expanded,
            atom_to_token_expanded,
            atom_mask_expanded,
            expanded_batch_size,
        ) = self._build_pair_and_single(
            single_inputs=single_inputs,
            pair_states=pair_states,
            predicted_coords=predicted_coords,
            distogram_atom_idx=distogram_atom_idx,
            token_attention_mask=token_attention_mask,
            atom_to_token=atom_to_token,
            atom_attention_mask=atom_attention_mask,
            num_diffusion_samples=num_diffusion_samples,
            relative_position_encoding=relative_position_encoding,
            token_bonds_encoding=token_bonds_encoding,
        )

        expanded_type, expanded_asym = _expand_samples(num_diffusion_samples, mol_type, asym_id)
        atom_confidences = self._compute_atom_confidences(
            single_states=single_states,
            atom_to_token_expanded=atom_to_token_expanded,
            atom_mask_expanded=atom_mask_expanded,
            rep_idx_expanded=rep_idx_expanded,
            rep_distances=rep_distances,
            expanded_type=expanded_type,
            expanded_asym=expanded_asym,
            expanded_batch_size=expanded_batch_size,
        )

        pae_logits = self.pae_head(self.pae_layernorm(pair_states))
        pde_logits = self.pde_head(self.pde_layernorm(pair_states))
        # Expected-value pae/pde are reported metrics; only the logits are trained.
        with torch.no_grad():
            pae = _categorical_mean(pae_logits, start=0.0, end=32.0)
            pde = _categorical_mean(pde_logits, start=0.0, end=32.0)

        ptm, iptm, pair_chains_iptm = self._compute_ptm_iptm(
            pae_logits, token_mask, expanded_asym, expanded_batch_size
        )

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


class EsmFold2MSAEncoderLayer(nn.Module):
    """One MSA encoder block: OPM into pair, MSA pair-weighted averaging, triangle update.

    The final block updates only the pair stream, so it does not build the two MSA-stream submodules:
    the released checkpoints carry no weights for them (the MSA representation it would produce is
    never read), and `nn.Identity` is not a substitute -- these sit on residuals, so an identity would
    double the stream rather than leave it alone.
    """

    def __init__(self, config: EsmFold2Config, is_final_block: bool = False) -> None:
        super().__init__()
        self.is_final_block = is_final_block
        self.outer_product_mean = EsmFold2OuterProductMean(config)
        if not is_final_block:
            self.msa_pair_weighted_averaging = EsmFold2MSAPairWeightedAveraging(config)
            self.msa_transition = EsmFold2Transition(
                config.msa_encoder.hidden_size, config.msa_encoder.intermediate_size, config.chunk_size
            )
        self.tri_mul_out = EsmFold2TriangleMultiplicativeUpdate(config, outgoing=True)
        self.tri_mul_in = EsmFold2TriangleMultiplicativeUpdate(config, outgoing=False)
        self.pair_transition = EsmFold2Transition(
            config.pairwise_hidden_size, config.pair_transition_intermediate_size, config.chunk_size
        )

    def forward(
        self,
        msa_states: Tensor,
        pair_states: Tensor,
        msa_attention_mask: Tensor,
        pair_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pair_states = pair_states + self.outer_product_mean(msa_states, msa_attention_mask)
        if not self.is_final_block:
            msa_states = msa_states + self.msa_pair_weighted_averaging(msa_states, pair_states, pair_attention_mask)
            msa_states = self.msa_transition(msa_states)
        pair_states = pair_states + self.tri_mul_out(pair_states, visibility=pair_attention_mask)
        pair_states = pair_states + self.tri_mul_in(pair_states, visibility=pair_attention_mask)
        pair_states = self.pair_transition(pair_states)
        return msa_states, pair_states


class EsmFold2MSAEncoder(nn.Module):
    """Stack of `EsmFold2MSAEncoderLayer` layers that conditions the pair on an MSA."""

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        # num_res_types one-hot + has_deletion + deletion_value.
        self.embed = nn.Linear(config.num_res_types + 2, config.msa_encoder.hidden_size, bias=False)
        self.project_inputs = nn.Linear(config.single_inputs_size, config.msa_encoder.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [
                EsmFold2MSAEncoderLayer(config, is_final_block=(i == config.msa_encoder.num_hidden_layers - 1))
                for i in range(config.msa_encoder.num_hidden_layers)
            ]
        )

    def forward(
        self,
        pair_states: Tensor,
        single_inputs: Tensor,
        msa_one_hot: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
        msa_attention_mask: Tensor,
    ) -> Tensor:
        # All inputs are pre-transposed to [B, L, M, ...] before calling.
        msa_features = torch.cat([msa_one_hot, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1)
        msa_states = self.embed(msa_features.to(self.embed.weight.dtype)) + self.project_inputs(
            single_inputs
        ).unsqueeze(2)
        token_mask = msa_attention_mask[:, :, 0].bool()
        pair_attention_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
        for layer in self.layers:
            msa_states, pair_states = layer(msa_states, pair_states, msa_attention_mask, pair_attention_mask)
        return pair_states


class EsmFold2Parcae(nn.Module):
    """The trunk's linear-recurrence state update (internally "parcae").

    A diagonal state-space recurrence whose time axis is the *recycling loop*, not the sequence: each
    trunk loop injects the refreshed pair representation into the state, which decays geometrically and
    so converges however many loops run (the fork sampled the loop count during training). The final
    state is projected by ``out_proj`` and refined by ``output_stack``.
    """

    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__()
        self.input_norm = EsmFold2LayerNorm(config.pairwise_hidden_size)
        self.log_state_decay = nn.Parameter(torch.zeros(config.pairwise_hidden_size))
        self.log_delta = nn.Parameter(torch.empty(config.pairwise_hidden_size, dtype=torch.float32))
        self.input_matrix_continuous = nn.Parameter(
            torch.empty(config.pairwise_hidden_size, config.pairwise_hidden_size)
        )
        self.out_proj = nn.Linear(config.pairwise_hidden_size, config.pairwise_hidden_size, bias=False)
        self.output_stack = EsmFold2PairUpdateStack(config, config.parcae_num_coda_layers)

    def discretize(self) -> tuple[Tensor, Tensor]:
        """``(state_decay, input_matrix)`` -- the per-channel state transition (Ā) and the discretized
        input projection (B̄) of the recurrence.

        Loop-invariant, so the trunk computes it once outside the loop and passes it to ``forward``.
        """
        delta = F.softplus(self.log_delta)
        state_decay = torch.exp(-delta * torch.exp(self.log_state_decay))
        input_matrix = delta[:, None] * self.input_matrix_continuous
        return state_decay, input_matrix

    def forward(
        self, state: Tensor, injected_pair_states: Tensor, state_decay: Tensor, input_matrix: Tensor
    ) -> Tensor:
        """One recurrence step: decay the state and inject the refreshed pair representation."""
        return state_decay * state + F.linear(self.input_norm(injected_pair_states), input_matrix)

    def decode_state(self, state: Tensor, pair_attention_mask: Tensor | None = None) -> Tensor:
        """Project the final state to a pair representation and refine it with the output stack."""
        return self.output_stack(self.out_proj(state), pair_attention_mask=pair_attention_mask)


@auto_docstring
class EsmFold2PreTrainedModel(PreTrainedModel):
    config_class = EsmFold2Config
    base_model_prefix = "esmfold2"
    main_input_name = "token_index"
    _no_split_modules = [
        "EsmcLayer",
        "EsmFold2PairUpdateLayer",
        "EsmFold2AtomEncoder",
        "EsmFold2AtomDecoder",
        "EsmFold2DiffusionTransformer",
    ]
    supports_gradient_checkpointing = True
    # Every norm weight/bias, plus the Fourier noise-embedding buffers, stay fp32 under a bf16 load.
    _keep_in_fp32_modules_strict = ["fourier", "norm"]
    _supports_sdpa = True

    def _init_weights(self, module):
        # The non-default inits: adaLN-Zero gates, the parcae recurrence, zeroed output projections.
        super()._init_weights(module)
        if isinstance(module, EsmFold2Parcae):
            init.eye_(module.out_proj.weight)
            init.eye_(module.input_matrix_continuous)
            init.zeros_(module.log_state_decay)
            # Chosen so the initial per-step state decay is exactly 1/sqrt(5).
            parcae_delta_init = -math.log(math.sqrt(1.0 / 5.0))
            init.constant_(module.log_delta, _inverse_softplus(parcae_delta_init))
        elif isinstance(module, EsmFold2ConfidenceHead):
            init.zeros_(module.plddt_weight)
            init.zeros_(module.resolved_weight)
        elif isinstance(module, EsmFold2AtomLayer):
            init.zeros_(module.adaln_linear.weight)
        elif isinstance(module, EsmFold2AttentionPairBias):
            if getattr(module, "out_gate", None) is not None:
                init.zeros_(module.out_gate.weight)
                init.constant_(module.out_gate.bias, -2.0)
        elif isinstance(module, EsmFold2ConditionedTransition):
            if getattr(module, "output_gate", None) is not None:
                init.zeros_(module.output_gate.weight)
                init.constant_(module.output_gate.bias, -2.0)
        elif isinstance(module, EsmFold2DiffusionModule):
            init.zeros_(module.single_to_token.weight)
        elif isinstance(module, EsmFold2LanguageModelEncoder):
            init.zeros_(module.layer_weights)


@auto_docstring(
    custom_intro="""
    ESMFold2 all-atom protein structure predictor with a bundled ESMC protein-language-model backbone. This is the
    standard released ESMFold2 architecture, whose trunk is a linear-recurrent stack (internally referred to as
    "parcae").
    """
)
class EsmFold2Model(EsmFold2PreTrainedModel, EsmFold2FoldingMixin):
    def __init__(self, config: EsmFold2Config) -> None:
        super().__init__(config)

        self.inputs_atom_encoder = EsmFold2AtomEncoder(config, structure_prediction=False)
        self.pair_init_1 = nn.Linear(config.single_inputs_size, config.pairwise_hidden_size, bias=False)
        self.pair_init_2 = nn.Linear(config.single_inputs_size, config.pairwise_hidden_size, bias=False)
        self.rel_pos = EsmFold2RelativePositionEncoding(config)
        self.token_bonds = nn.Linear(1, config.pairwise_hidden_size, bias=False)
        self.language_model = EsmFold2LanguageModelEncoder(config)
        # Populated by from_pretrained from the checkpoint's ``esmc.*`` weights; run frozen (no_grad).
        self.esmc = AutoModel.from_config(config.esmc_config)

        self.folding_trunk = EsmFold2PairUpdateStack(config, config.folding_trunk_num_hidden_layers)
        self.lm_encoder = EsmFold2PairUpdateStack(config, config.lm_encoder.num_hidden_layers)

        self.parcae = EsmFold2Parcae(config)

        # Heads --------------------------------------------------------------
        # The denoiser itself; the sampling loop that drives it lives in ``EsmFold2FoldingMixin``.
        self.structure_head = EsmFold2DiffusionModule(config)
        self.distogram_head = nn.Linear(
            config.pairwise_hidden_size, config.structure_head.num_distogram_bins, bias=True
        )
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
        token_mask: Tensor,
    ) -> Tensor:
        """Run ESMC with BOS/EOS wrapping; returns hidden states
        ``[batch_size, num_tokens, num_esmc_layers + 1, lm_hidden_size]``.

        Atom-tokenized modified residues (HYP, MSE, ACE, NH2, ...) span several structure tokens but
        share one ``(asym_id, residue_index)`` key, so they are collapsed to a single LM token -- the
        LM was trained per-residue -- and the hidden states scattered back to the per-token layout.
        """
        batch_size, num_tokens = input_ids.shape
        device = input_ids.device
        protein_mask = (mol_type == 0) & token_mask

        lm_input_list = []
        lm_lengths = []
        # Per-batch maps from (original protein-token index) to (LM input position).
        expand_maps: list[Tensor] = []
        # Largest number of chains in any row, which decides how the backbone is masked below.
        max_chains = 0
        for b in range(batch_size):
            mask_row = protein_mask[b]
            ids_row = input_ids[b][mask_row]
            asym_row = asym_id[b][mask_row]
            res_row = residue_index[b][mask_row]

            # Keep the first token per (asym_id, residue_index) key, in input order.
            keys = torch.stack((asym_row, res_row), dim=1)
            unique_keys, inverse = torch.unique(keys, dim=0, return_inverse=True)
            num_unique_residues = unique_keys.size(0)
            token_positions = torch.arange(keys.size(0), device=device, dtype=torch.long)
            first_pos = torch.full((num_unique_residues,), keys.size(0), device=device, dtype=torch.long)
            first_pos.scatter_reduce_(0, inverse, token_positions, reduce="amin", include_self=True)
            ordered = torch.argsort(first_pos)
            first_pos_ordered = first_pos[ordered]
            ids_collapsed = ids_row[first_pos_ordered]
            asym_collapsed = asym_row[first_pos_ordered]
            remap = torch.empty_like(ordered)
            remap[ordered] = torch.arange(num_unique_residues, device=device, dtype=torch.long)
            inverse_ordered = remap[inverse]

            chain_ids = asym_collapsed.unique(sorted=True)
            max_chains = max(max_chains, len(chain_ids))
            # [BOS] chain1 [EOS BOS] chain2 ... [EOS]
            parts: list[Tensor] = [torch.tensor([0], device=device, dtype=ids_row.dtype)]
            # Per-chain LM positions accumulate; track them for the expand map.
            per_token_lm_pos = torch.empty(num_unique_residues, device=device, dtype=torch.long)
            cursor = 1  # position 0 is the leading BOS
            for i, chain_id in enumerate(chain_ids):
                in_chain = (asym_collapsed == chain_id).nonzero(as_tuple=True)[0]
                parts.append(ids_collapsed[in_chain])
                per_token_lm_pos[in_chain] = torch.arange(
                    cursor, cursor + in_chain.shape[0], device=device, dtype=torch.long
                )
                cursor += in_chain.shape[0]
                if i < len(chain_ids) - 1:
                    parts.append(torch.tensor([2, 0], device=device, dtype=ids_row.dtype))
                    cursor += 2  # EOS + BOS
            parts.append(torch.tensor([2], device=device, dtype=ids_row.dtype))
            lm_seq = torch.cat(parts)
            lm_input_list.append(lm_seq)
            lm_lengths.append(lm_seq.shape[0])

            # Original protein-token position → LM input position.
            prot_pos_row = mask_row.nonzero(as_tuple=True)[0]
            expand_map = torch.full((num_tokens,), -1, device=device, dtype=torch.long)
            expand_map[prot_pos_row] = per_token_lm_pos[inverse_ordered]
            expand_maps.append(expand_map)

        # Pad to the longest LM input.
        max_len = max(lm_lengths)
        lm_input_ids = torch.full((batch_size, max_len), 1, device=device, dtype=input_ids.dtype)  # PAD=1
        for b in range(batch_size):
            lm_input_ids[b, : lm_lengths[b]] = lm_input_list[b]

        # sequence_id for chain-aware attention; PAD tokens get -1 (no attention).
        sequence_id = (lm_input_ids == 0).cumsum(dim=1) - 1  # BOS=0
        sequence_id = sequence_id.masked_fill(lm_input_ids == 1, -1)  # PAD=1

        # Chain-aware masking is only needed when a row holds more than one chain. With a single
        # chain the two spellings agree wherever it matters: ``sequence_id`` equality gives
        # valid<->valid plus pad<->pad, a padding mask gives valid<->valid plus padded *query* rows
        # whose outputs are never gathered below. Preferring the padding mask keeps flash attention
        # available for the backbone, which dominates the cost of a fold, and matches the reference
        # (it refuses only multi-chain inputs under flash attention). The choice is made here from a
        # host-side chain count, so the backbone's forward keeps no data-dependent branch.
        mask_kwargs = {"sequence_id": sequence_id} if max_chains > 1 else {"attention_mask": sequence_id >= 0}

        use_amp = next(self.esmc.parameters()).dtype == torch.bfloat16
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
            esmc_out = self.esmc(input_ids=lm_input_ids, output_hidden_states=True, **mask_kwargs)

        # Stack the per-layer tuple into the single tensor the projection expects.
        lm_hidden_states = torch.stack(
            esmc_out.hidden_states, dim=0
        )  # [num_esmc_layers + 1, batch_size, max_len, lm_hidden_size]
        num_layers_plus_one, _, _, lm_hidden_size = lm_hidden_states.shape
        result = torch.zeros(
            batch_size, num_tokens, num_layers_plus_one, lm_hidden_size, device=device, dtype=lm_hidden_states.dtype
        )
        for b in range(batch_size):
            row_mask = protein_mask[b]
            lm_positions = expand_maps[b][row_mask]  # [num_protein_tokens] LM positions
            # lm_hidden_states[:, b, lm_positions, :] -> [num_esmc_layers + 1, num_protein_tokens, lm_hidden_size]
            gathered = lm_hidden_states[:, b, lm_positions, :].permute(1, 0, 2)
            result[b, row_mask.nonzero(as_tuple=True)[0]] = gathered

        return result

    def _init_pair_state(self, reference: Tensor) -> Tensor:
        std = math.sqrt(2.0 / (5.0 * reference.shape[-1]))
        state = torch.empty_like(reference, dtype=torch.float32)
        nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3 * std, b=3 * std)
        return state.to(dtype=reference.dtype)

    def _prepare_features(
        self,
        res_type: Tensor,
        token_mask: Tensor,
        msa: Tensor | None,
        msa_attention_mask: Tensor | None,
        deletion_mean: Tensor | None,
        ref_element: Tensor,
        ref_atom_name_chars: Tensor,
        atom_attention_mask: Tensor,
        atom_to_token: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """One-hot / mask the raw structural inputs into embedder-ready features.

        Returns ``(res_type_one_hot, profile, deletion_mean, ref_element_one_hot,
        ref_atom_name_chars_one_hot, atom_to_token)`` with ``atom_to_token`` zeroed at padding.
        """
        if res_type.dim() == 2:
            res_type_one_hot = F.one_hot(res_type, num_classes=self.config.num_res_types).float()
            res_type_one_hot = res_type_one_hot * token_mask.unsqueeze(-1).float()
        else:
            res_type_one_hot = res_type.float()

        if msa is not None:
            msa_profile_one_hot = F.one_hot(msa, num_classes=self.config.num_res_types).float()
            if msa_attention_mask is not None:
                mask_float = msa_attention_mask.float().unsqueeze(-1)
                msa_profile_one_hot = msa_profile_one_hot * mask_float
                valid_seq_count = msa_attention_mask.float().sum(dim=1).clamp(min=1)
                profile = msa_profile_one_hot.sum(dim=1) / valid_seq_count.unsqueeze(-1)
            else:
                profile = msa_profile_one_hot.mean(dim=1)
        else:
            profile = res_type_one_hot

        if deletion_mean is None:
            deletion_mean = torch.zeros(res_type.shape[0], res_type.shape[1], device=res_type.device)

        ref_element_one_hot = F.one_hot(ref_element, num_classes=self.config.max_atomic_number).float()
        ref_atom_name_chars_one_hot = F.one_hot(ref_atom_name_chars, num_classes=self.config.char_vocab_size).float()
        # Bias-free downstream Linears require zeroed padding.
        atom_mask_float = atom_attention_mask.float()
        ref_element_one_hot = ref_element_one_hot * atom_mask_float.unsqueeze(-1)
        ref_atom_name_chars_one_hot = ref_atom_name_chars_one_hot * atom_mask_float.unsqueeze(-1).unsqueeze(-1)
        atom_to_token = atom_to_token * atom_attention_mask.long()

        return (
            res_type_one_hot,
            profile,
            deletion_mean,
            ref_element_one_hot,
            ref_atom_name_chars_one_hot,
            atom_to_token,
        )

    def _build_msa_kwargs(
        self,
        msa: Tensor | None,
        msa_attention_mask: Tensor | None,
        has_deletion: Tensor | None,
        deletion_value: Tensor | None,
        token_mask: Tensor,
        single_inputs: Tensor,
    ) -> dict | None:
        """Assemble the transposed/padded one-hot MSA tensors the MSA encoder consumes."""
        if msa is None:
            return None
        batch_size, msa_depth, num_tokens = msa.shape
        msa_one_hot = F.one_hot(msa.permute(0, 2, 1), num_classes=self.config.num_res_types).float()
        msa_mask = (
            msa_attention_mask.permute(0, 2, 1).float()
            if msa_attention_mask is not None
            else token_mask[:, :, None].expand(-1, -1, msa_depth).float()
        )
        # Bias-free EsmFold2MSAEncoder.embed requires zeroed padding.
        msa_one_hot = msa_one_hot * msa_mask.unsqueeze(-1)
        has_deletion_t = (
            has_deletion.permute(0, 2, 1).float()
            if has_deletion is not None
            else torch.zeros(batch_size, num_tokens, msa_depth, device=msa.device)
        )
        deletion_value_t = (
            deletion_value.permute(0, 2, 1)
            if deletion_value is not None
            else torch.zeros(batch_size, num_tokens, msa_depth, device=msa.device)
        )
        return {
            "single_inputs": single_inputs,
            "msa_one_hot": msa_one_hot,
            "has_deletion": has_deletion_t,
            "deletion_value": deletion_value_t,
            "msa_attention_mask": msa_mask,
        }

    def _run_trunk_loops(
        self,
        pair_states: Tensor,
        initial_pair_states: Tensor,
        lm_pair_states: Tensor | None,
        msa_kwargs: dict | None,
        pair_mask: Tensor,
        state_decay: Tensor,
        input_matrix: Tensor,
        total_steps: int,
    ) -> Tensor:
        # A helper rather than an inline loop, so the per-iteration num_tokens^2 x pairwise_hidden_size locals are freed on return.
        lm_dropout_p = self.config.lm_encoder.lm_dropout
        per_loop_lm_dropout = (
            lm_pair_states is not None and self.config.lm_encoder.per_loop_lm_dropout and lm_dropout_p > 0.0
        )

        for _ in range(total_steps):
            if per_loop_lm_dropout:
                # ``training=True`` forces the per-loop LM dropout to resample under ``eval()``.
                dropped_lm_pair_states: Tensor | None = F.dropout(lm_pair_states, p=lm_dropout_p, training=True)
            else:
                dropped_lm_pair_states = lm_pair_states

            refined_lm_pair_states: Tensor | None = None
            if dropped_lm_pair_states is not None:
                refined_lm_pair_states = self.lm_encoder(dropped_lm_pair_states, pair_attention_mask=pair_mask)

            injected_pair_states = initial_pair_states
            if msa_kwargs is not None:
                msa_pair = self.msa_encoder(pair_states=injected_pair_states, **msa_kwargs)
                injected_pair_states = (
                    msa_pair if self.config.msa_encoder.overwrite else (injected_pair_states + msa_pair)
                )

            if refined_lm_pair_states is not None:
                injected_pair_states = injected_pair_states + refined_lm_pair_states

            pair_states = self.parcae(pair_states, injected_pair_states, state_decay, input_matrix)
            pair_states = self.folding_trunk(pair_states, pair_attention_mask=pair_mask)

        return pair_states

    @auto_docstring(
        custom_intro="""
        Run the folding trunk: featurize the inputs, embed them into a pair representation, refine it over
        `num_loops` recycling iterations, and read off the distogram. This is the deterministic half of a
        structure prediction; the diffusion sampler that turns the returned pair representation into 3D
        coordinates lives in `EsmFold2FoldingMixin` — call [`~EsmFold2Model.fold`] or
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
        lm_hidden_states (`torch.Tensor` of shape `(batch_size, num_tokens, num_esmc_layers + 1, esmc_hidden_size)`, *optional*):
            Precomputed ESMC backbone hidden states, one per backbone layer plus the embeddings. When
            provided, the backbone is not run and `input_ids` is unused.
        num_loops (`int`, *optional*):
            Number of trunk refinement loops. Defaults to `config.num_loops`.
        """
        token_mask = token_attention_mask
        num_loops = num_loops if num_loops is not None else self.config.num_loops
        total_steps = max(1, num_loops + 1)

        res_type_one_hot, profile, deletion_mean, ref_element_one_hot, ref_atom_name_chars_one_hot, atom_to_token = (
            self._prepare_features(
                res_type=res_type,
                token_mask=token_mask,
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
            ref_element=ref_element_one_hot,
            ref_atom_name_chars=ref_atom_name_chars_one_hot,
            ref_space_uid=ref_space_uid,
            atom_to_token=atom_to_token,
        )

        # The inputs embedder runs its atom stack once, at the unexpanded batch size.
        atom_embeds, position_embeddings = self.inputs_atom_encoder.embed_atoms(atom_inputs)
        atom_encoding = self.inputs_atom_encoder(
            atom_embeds=atom_embeds,
            attention_mask=self.inputs_atom_encoder.build_attention_mask(atom_attention_mask, atom_embeds),
            position_embeddings=position_embeddings,
            atom_mask=atom_attention_mask,
            atom_to_token=atom_to_token,
            num_tokens=token_mask.shape[1],
        )[0]
        # Fold the fp32 input features into the atom encoding's dtype, so single_inputs is one dtype.
        dtype = atom_encoding.dtype
        single_inputs = torch.cat(
            [
                atom_encoding,
                res_type_one_hot.to(dtype),
                profile.to(dtype),
                deletion_mean.unsqueeze(-1).to(dtype),
            ],
            dim=-1,
        )

        initial_pair_states = self.pair_init_1(single_inputs).unsqueeze(2) + self.pair_init_2(single_inputs).unsqueeze(
            1
        )

        relative_position_encoding = self.rel_pos(
            residue_index=residue_index,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            token_index=token_index,
        )
        token_bonds_encoding = self.token_bonds(token_bonds.to(self.token_bonds.weight.dtype))
        initial_pair_states = initial_pair_states + relative_position_encoding + token_bonds_encoding

        if lm_hidden_states is None and input_ids is not None:
            lm_hidden_states = self._compute_lm_hidden_states(input_ids, asym_id, residue_index, mol_type, token_mask)
        lm_pair_states: Tensor | None = None
        if lm_hidden_states is not None:
            lm_pair_states = self.language_model(lm_hidden_states)
        del lm_hidden_states

        pair_mask = token_mask[:, :, None].float() * token_mask[:, None, :].float()

        pair_states = self._init_pair_state(initial_pair_states)

        state_decay, input_matrix = self.parcae.discretize()
        state_decay = state_decay.view(1, 1, 1, -1).to(device=pair_states.device, dtype=pair_states.dtype)
        input_matrix = input_matrix.to(device=pair_states.device, dtype=pair_states.dtype)

        msa_kwargs = self._build_msa_kwargs(
            msa=msa,
            msa_attention_mask=msa_attention_mask,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            token_mask=token_mask,
            single_inputs=single_inputs,
        )

        pair_states = self._run_trunk_loops(
            pair_states=pair_states,
            initial_pair_states=initial_pair_states,
            lm_pair_states=lm_pair_states,
            msa_kwargs=msa_kwargs,
            pair_mask=pair_mask,
            state_decay=state_decay,
            input_matrix=input_matrix,
            total_steps=total_steps,
        )
        del initial_pair_states, lm_pair_states, msa_kwargs, state_decay, input_matrix

        pair_states = self.parcae.decode_state(pair_states, pair_attention_mask=pair_mask)

        pair_states = pair_states.float()
        distogram_logits = self.distogram_head(
            (pair_states + pair_states.transpose(-2, -3)).to(self.distogram_head.weight.dtype)
        )

        return EsmFold2TrunkOutput(
            distogram_logits=distogram_logits,
            pair_states=pair_states,
            single_inputs=single_inputs,
            relative_position_encoding=relative_position_encoding,
            token_bonds_encoding=token_bonds_encoding,
            atom_inputs=atom_inputs,
        )


__all__ = ["EsmFold2Model", "EsmFold2PreTrainedModel"]
