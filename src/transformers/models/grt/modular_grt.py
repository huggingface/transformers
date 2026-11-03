# Copyright 2026 MatterWave AI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch GRT model."""

import math
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..llama.modeling_llama import eager_attention_forward


@auto_docstring(checkpoint="matterwaveai/grt-30m")
@strict
class GRTConfig(PreTrainedConfig):
    r"""
    intermediate_size (`int`, *optional*):
        Width the feed-forward network pads the hidden state to before its rotation chains. Defaults to
        `4 * hidden_size`. Must be even and at least `hidden_size`.
    output_scale_init (`float`, *optional*, defaults to 30.0):
        Initial value of the learnable logit temperature of the cosine head. It is stored through a softplus, so the
        effective temperature is always positive.
    ffn_n_rotations (`int`, *optional*, defaults to 21):
        Number of rotation stages in each of the feed-forward expand and contract chains.
    n_qkv_rotations (`int`, *optional*, defaults to 11):
        Number of rotation stages in each of the Q/K/V/O projection chains.
    n_mixer_rotations (`int`, *optional*, defaults to 1):
        Number of rotation stages in the cross-head mixer. `0` disables the mixer.
    coupling_init_scale (`float`, *optional*, defaults to 0.01):
        Standard deviation used to initialize the coupling tensors that modulate rotation angles by content.
    angle_init_scale (`float`, *optional*, defaults to 0.1):
        Standard deviation used to initialize the rotation angles.
    phase_init_scale (`float`, *optional*, defaults to 0.1):
        Standard deviation used to initialize the embedding phase angles.
    log_magnitude_init (`float`, *optional*, defaults to 0.0):
        Initial per-token log magnitude of the embedding.
    pos_freq_base (`float`, *optional*, defaults to 10000.0):
        Base of the geometric frequency ladder used by the rotational position encoding.
    learnable_rope (`bool`, *optional*, defaults to `True`):
        Whether the position frequencies are learned parameters.
    coupling_init_mode (`str`, *optional*, defaults to `"diagonal"`):
        Coupling initialization: `"diagonal"` adds a self-coupling term on top of the random init, `"random"` does not.
    learnable_residual_gates (`bool`, *optional*, defaults to `True`):
        Whether each residual branch is scaled by a learned gate.
    residual_gate_mode (`str`, *optional*, defaults to `"bounded_depth"`):
        Gate parameterization, one of `"free"`, `"bounded"`, `"bounded_depth"` or `"none"`.
    residual_norm_mode (`str`, *optional*, defaults to `"none"`):
        `"pearlnorm"` normalizes the residual stream after every block, `"none"` does not.
    peri_ln (`bool`, *optional*, defaults to `False`):
        Whether to normalize the attention and feed-forward outputs before they enter the residual stream.
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether the cosine head reads the phase angles of the input embedding rather than its own table.

    ```python
    >>> from transformers import GRTConfig, GRTModel

    >>> # Initializing a GRT configuration
    >>> configuration = GRTConfig()

    >>> # Initializing a model from the configuration
    >>> model = GRTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "grt"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 151936
    hidden_size: int = 256
    intermediate_size: int | None = None
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    max_position_embeddings: int = 2048
    attention_dropout: float | int = 0.0
    output_scale_init: float = 30.0
    ffn_n_rotations: int = 21
    n_qkv_rotations: int = 11
    n_mixer_rotations: int = 1
    coupling_init_scale: float = 0.01
    angle_init_scale: float = 0.1
    phase_init_scale: float = 0.1
    log_magnitude_init: float = 0.0
    pos_freq_base: float = 10000.0
    learnable_rope: bool = True
    coupling_init_mode: Literal["diagonal", "random"] = "diagonal"
    learnable_residual_gates: bool = True
    residual_gate_mode: Literal["free", "bounded", "bounded_depth", "none"] = "bounded_depth"
    residual_norm_mode: Literal["none", "pearlnorm"] = "none"
    peri_ln: bool = False
    initializer_range: float = 0.02
    use_cache: bool = True
    # The Qwen3 tokenizer's `<|endoftext|>`.
    pad_token_id: int | None = 151643
    bos_token_id: int | None = 151643
    eos_token_id: int | list[int] | None = 151643
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be divisible by `num_attention_heads` "
                f"({self.num_attention_heads})."
            )
        head_dim = self.hidden_size // self.num_attention_heads
        if self.intermediate_size < self.hidden_size:
            raise ValueError(
                f"`intermediate_size` ({self.intermediate_size}) must be at least `hidden_size` ({self.hidden_size}): "
                "the feed-forward network zero-pads the hidden state up to it."
            )
        widths = (
            ("head_dim", head_dim, self.n_qkv_rotations),
            ("hidden_size", self.hidden_size, self.n_mixer_rotations),
            ("intermediate_size", self.intermediate_size, self.ffn_n_rotations),
        )
        for name, width, n_stages in widths:
            if width % 2 != 0:
                raise ValueError(f"`{name}` ({width}) must be even: every rotation acts on pairs of dimensions.")
            # Stage k of a chain pairs dimensions at stride 2**(k % n_levels) inside blocks of twice that stride,
            # so the width must split into whole blocks of the largest stride the chain reaches.
            half = width // 2
            n_levels = max(1, int(math.log2(half))) if half > 1 else 1
            block = 2 ** min(n_stages, n_levels)
            if n_stages > 0 and width % block != 0:
                raise ValueError(
                    f"`{name}` ({width}) must be a multiple of {block} for a chain of {n_stages} butterfly stages."
                )


def _raw_scale_init(target_scale: float) -> float:
    """Inverse of the softplus that every scale goes through, so that `softplus(raw) ~= target_scale`."""
    if target_scale > 20:
        # softplus is the identity to float precision this far out
        return target_scale
    return math.log(math.exp(target_scale) - 1) if target_scale > 0 else 0.0


def _build_butterfly_perms(n_stages: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Butterfly permutations and their inverses, stacked as `(n_stages, dim)`.

    Stage `k` pairs dimensions at stride `2 ** (k % n_levels)` with `n_levels = log2(dim / 2)`, so after
    `n_levels` stages every dimension has a path to every other one.
    """
    half = dim // 2
    n_levels = max(1, int(math.log2(half))) if half > 1 else 1
    perms = []
    for k in range(n_stages):
        stride = 1 << (k % n_levels)
        order = []
        for block_start in range(0, dim, 2 * stride):
            for i in range(stride):
                order.append(block_start + i)
                order.append(block_start + i + stride)
        perms.append(order)
    perms = torch.tensor(perms, dtype=torch.long).view(n_stages, dim)
    return perms, torch.argsort(perms, dim=-1)


def _clifford_features(x0: torch.Tensor, x1: torch.Tensor, mixer_bivector: bool = False) -> torch.Tensor:
    """Content features that modulate the rotation angles.

    Each pair `(x0, x1)` is read as a Cl(2,0) multivector: the scalar pair energy `S0`, both vector parts and a
    bivector with the next pair. GRT contracts the bivector differently in the mixer than in the projections,
    which `mixer_bivector` selects.
    """
    x0_next = torch.cat([x0[..., 1:], x0[..., :1]], dim=-1)
    x1_next = torch.cat([x1[..., 1:], x1[..., :1]], dim=-1)
    bivector = x0 * x0_next - x1 * x1_next if mixer_bivector else x0 * x1_next - x1 * x0_next
    return torch.cat([x0 * x0 + x1 * x1, x0, x1, bivector], dim=-1)


def _quantize_symmetric(tensor: torch.Tensor, bits: int = 8, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel/per-row weight quantization (`bits` in `{8, 4}`)."""
    if bits not in (4, 8):
        raise ValueError(f"Unsupported quantization bits={bits}; expected 4 or 8")
    qmax = 127.0 if bits == 8 else 7.0
    t_f32 = tensor.to(torch.float32)
    scale = t_f32.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(t_f32 / scale).clamp(-qmax, qmax).to(torch.int8)
    if bits == 8:
        return q, scale.to(tensor.dtype)
    if q.shape[dim] % 2 != 0:
        raise ValueError(f"INT4 packing requires even size along dim={dim}, got {q.shape[dim]}")
    q_u8 = (q + 8).to(torch.uint8)
    idx0 = torch.arange(0, q_u8.shape[dim], 2, device=q.device)
    idx1 = idx0 + 1
    low = q_u8.index_select(dim, idx0)
    high = q_u8.index_select(dim, idx1)
    packed = (low & 0x0F) | ((high & 0x0F) << 4)
    return packed, scale.to(tensor.dtype)


def _dequantize_symmetric(qweight: torch.Tensor, scale: torch.Tensor, bits: int = 8, dim: int = -1) -> torch.Tensor:
    """Dequantize a weight tensor produced by `_quantize_symmetric`."""
    if bits == 8:
        return qweight.to(scale.dtype) * scale
    if bits == 4:
        low = (qweight & 0x0F).to(torch.int8) - 8
        high = ((qweight >> 4) & 0x0F).to(torch.int8) - 8
        d = dim if dim >= 0 else qweight.ndim + dim
        unpacked = torch.stack([low, high], dim=d + 1).flatten(d, d + 1)
        return unpacked.to(scale.dtype) * scale
    raise ValueError(f"Unsupported quantization bits={bits}; expected 4 or 8")


def _coupled_rotate(
    hidden_states: torch.Tensor,
    angles: torch.Tensor,
    coupling: torch.Tensor,
    scales: torch.Tensor,
    perms: torch.Tensor | None,
    inv_perms: torch.Tensor | None,
    coupling_scale: torch.Tensor | None = None,
    coupling_bits: int | None = None,
) -> torch.Tensor:
    """Apply a chain of content-modulated rotation + scale stages.

    Every projection in GRT is such a chain. A stage splits its input into adjacent pairs `(x0, x1)`, rotates
    each pair by an angle and applies a per-dimension scale; the butterfly permutation before each stage changes
    which dimensions are paired, so the chain mixes the whole vector like a dense matrix with O(D) parameters per
    stage instead of O(D^2).

    Args:
        hidden_states: `(num_tokens, num_heads, head_dim)`.
        angles: `(n_stages, num_heads, head_dim // 2)` base rotation angles.
        coupling: `(n_stages, num_heads, 4 * head_dim // 2, head_dim // 2)`.
        scales: `(n_stages, num_heads, head_dim)`, already positive.
        perms / inv_perms: `(n_stages, head_dim)` butterfly permutations, `None` for a single stage.

    Each stage offsets its angles by a function of the vector it rotates, which is what gives the chain the
    input dependence a dense matrix would have.
    """
    n_pairs = hidden_states.shape[-1] // 2
    out = hidden_states
    for k in range(angles.shape[0]):
        if perms is not None:
            out = out[..., perms[k]]

        pairs = out.view(*out.shape[:-1], n_pairs, 2)
        x0, x1 = pairs[..., 0], pairs[..., 1]

        features = _clifford_features(x0, x1)
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-6)
        ck = (
            _dequantize_symmetric(coupling[k], coupling_scale[k], bits=coupling_bits, dim=-2).to(features.dtype)
            if coupling_scale is not None and coupling_bits is not None
            else coupling[k]
        )
        theta = angles[k] + torch.tanh(torch.einsum("nhj,hji->nhi", features, ck))

        cos, sin = torch.cos(theta), torch.sin(theta)
        out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).view(out.shape) * scales[k]

        if inv_perms is not None:
            out = out[..., inv_perms[k]]
    return out


def _plain_rotate(
    hidden_states: torch.Tensor,
    angles: torch.Tensor,
    raw_scales: torch.Tensor,
    perms: torch.Tensor | None,
    inv_perms: torch.Tensor | None,
) -> torch.Tensor:
    """Apply a chain of fixed rotation + scale stages; the feed-forward activation supplies the non-linearity."""
    scales = F.softplus(raw_scales)
    orig_shape = hidden_states.shape
    out = hidden_states.reshape(-1, orig_shape[-1])
    num_tokens, dim = out.shape
    n_pairs = angles.shape[-1]
    for k in range(angles.shape[0]):
        if perms is not None:
            out = out[:, perms[k]]

        pairs = out.view(num_tokens, n_pairs, 2)
        x0, x1 = pairs[:, :, 0], pairs[:, :, 1]

        cos, sin = torch.cos(angles[k]), torch.sin(angles[k])
        out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).view(num_tokens, dim) * scales[k]

        if inv_perms is not None:
            out = out[:, inv_perms[k]]
    return out.reshape(orig_shape)


class GRTPearlNorm(nn.Module):
    """Pair-aware RMSNorm.

    Divides by the RMS of the pair radii `r_i = sqrt(x_{2i}^2 + x_{2i+1}^2)` instead of the RMS over all
    dimensions, which rescales pair magnitudes and leaves every phase angle untouched.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pairs = hidden_states.unflatten(-1, (-1, 2))
        radius_sq = (pairs**2).sum(dim=-1)
        rms = torch.sqrt(radius_sq.mean(dim=-1, keepdim=True) + self.eps)
        return hidden_states / rms * self.scale


class GRTRadialGELU(nn.Module):
    """GELU on pair magnitudes: each pair is scaled by `gelu(r) / r`, so its direction is untouched."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pairs = hidden_states.unflatten(-1, (-1, 2))
        radius = torch.norm(pairs, dim=-1, keepdim=True)
        gate = F.gelu(radius) / (radius + self.eps)
        return (pairs * gate).flatten(-2)


class GRTEmbedding(nn.Module):
    """Phase-angle token embedding.

    A token is `hidden_size // 2` angles. The forward pass emits `(cos, sin)` per angle, which puts every token on
    the same sphere, and a per-token log magnitude lets the model move tokens off it.
    """

    def __init__(self, config: GRTConfig):
        super().__init__()
        self.phase_angles = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size // 2))
        self.log_magnitude = nn.Parameter(torch.empty(config.vocab_size))
        self.register_buffer("_table_q", None, persistent=False)
        self.register_buffer("_table_scale", None, persistent=False)
        self._quant_bits = None

    def table(self) -> torch.Tensor:
        if self._table_q is not None and self._table_scale is not None and self._quant_bits is not None:
            return _dequantize_symmetric(self._table_q, self._table_scale, bits=self._quant_bits, dim=-1)
        embeddings = torch.stack([torch.cos(self.phase_angles), torch.sin(self.phase_angles)], dim=-1).flatten(-2)
        magnitude = torch.exp(self.log_magnitude.clamp(-5.0, 5.0)).unsqueeze(-1)
        return embeddings * magnitude

    @torch.no_grad()
    def quantize_weights(self, bits: int = 8) -> None:
        full_table = self.table()
        qweight, scale = _quantize_symmetric(full_table, bits=bits, dim=-1)
        self.register_buffer("_table_q", qweight, persistent=False)
        self.register_buffer("_table_scale", scale, persistent=False)
        self._quant_bits = bits
        self.phase_angles = nn.Parameter(
            torch.empty(0, dtype=full_table.dtype, device=full_table.device), requires_grad=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._table_q is not None and self._table_scale is not None and self._quant_bits is not None:
            return _dequantize_symmetric(
                self._table_q[input_ids], self._table_scale[input_ids], bits=self._quant_bits, dim=-1
            )
        angles = self.phase_angles[input_ids]
        embeddings = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).flatten(-2)
        magnitude = torch.exp(self.log_magnitude[input_ids].clamp(-5.0, 5.0)).unsqueeze(-1)
        return embeddings * magnitude


def _default_log_freqs(config: GRTConfig) -> torch.Tensor:
    n_pairs = config.hidden_size // 2
    return -torch.arange(0, n_pairs, dtype=torch.float32) / n_pairs * math.log(config.pos_freq_base)


class GRTRotaryEmbedding(nn.Module):
    """Rotational position encoding.

    Rotates each dimension pair of the input embedding by `position * freq_i` once, before the first block, with a
    geometric ladder of (optionally learned) frequencies. Nothing is added to the hidden state.
    """

    def __init__(self, config: GRTConfig):
        super().__init__()
        log_freqs = _default_log_freqs(config)
        if config.learnable_rope:
            self.log_freqs = nn.Parameter(log_freqs)
        else:
            self.log_freqs = nn.Buffer(log_freqs, persistent=False)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # The angle grows with the position, and bf16 cannot even represent positions past 256 exactly, so the
        # rotation runs in fp32 whatever the activation dtype.
        angles = position_ids.float().unsqueeze(-1) * torch.exp(self.log_freqs.float())
        cos, sin = torch.cos(angles), torch.sin(angles)
        pairs = hidden_states.float().unflatten(-1, (-1, 2))
        x0, x1 = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)
        return rotated.to(hidden_states.dtype)


class GRTAttention(nn.Module):
    """Multi-head attention whose Q/K/V/O projections are rotation chains.

    Each projection is `n_qkv_rotations` stages of butterfly shuffle, content-modulated Givens rotation and scale,
    applied per head. The attention itself is ordinary scaled dot-product attention.
    """

    def __init__(self, config: GRTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.n_stages = config.n_qkv_rotations
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.num_key_value_groups = 1

        n_pairs = self.head_dim // 2
        for name in ("q", "k", "v", "o"):
            setattr(self, f"{name}_angles", nn.Parameter(torch.empty(self.n_stages, self.num_heads, n_pairs)))
            setattr(self, f"{name}_scales", nn.Parameter(torch.empty(self.n_stages, self.num_heads, self.head_dim)))
            setattr(
                self,
                f"{name}_coupling",
                nn.Parameter(torch.empty(self.n_stages, self.num_heads, 4 * n_pairs, n_pairs)),
            )

        # A single-stage chain has no butterfly, and the checkpoints store no permutations for it.
        if self.n_stages > 1:
            perms, inv_perms = _build_butterfly_perms(self.n_stages, self.head_dim)
            self._attn_perms = nn.Buffer(perms)
            self._attn_inv_perms = nn.Buffer(inv_perms)
        else:
            self._attn_perms = self._attn_inv_perms = None
        for name in ("q", "k", "v", "o"):
            self.register_buffer(f"_{name}_coupling_scale", None, persistent=False)
        self._quant_bits = None

    @torch.no_grad()
    def quantize_weights(self, bits: int = 8) -> None:
        for name in ("q", "k", "v", "o"):
            c = getattr(self, f"{name}_coupling")
            s = getattr(self, f"_{name}_coupling_scale")
            if self._quant_bits is not None and s is not None:
                c = _dequantize_symmetric(c, s, bits=self._quant_bits, dim=-2)
            cq, cs = _quantize_symmetric(c, bits=bits, dim=-2)
            setattr(self, f"{name}_coupling", nn.Parameter(cq, requires_grad=False))
            self.register_buffer(f"_{name}_coupling_scale", cs, persistent=False)
        self._quant_bits = bits

    def _project(
        self,
        hidden_states: torch.Tensor,
        angles: torch.Tensor,
        raw_scales: torch.Tensor,
        coupling: torch.Tensor,
        coupling_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _coupled_rotate(
            hidden_states,
            angles,
            coupling,
            F.softplus(raw_scales),
            self._attn_perms,
            self._attn_inv_perms,
            coupling_scale=coupling_scale,
            coupling_bits=self._quant_bits,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_heads, self.head_dim)

        # The projections act on each position independently, so re-projecting an earlier token reproduces the
        # cached key and value exactly.
        heads = hidden_states.reshape(-1, self.num_heads, self.head_dim)
        query_states = self._project(heads, self.q_angles, self.q_scales, self.q_coupling, self._q_coupling_scale)
        key_states = self._project(heads, self.k_angles, self.k_scales, self.k_coupling, self._k_coupling_scale)
        value_states = self._project(heads, self.v_angles, self.v_scales, self.v_coupling, self._v_coupling_scale)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
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
            **kwargs,
        )

        attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim)
        attn_output = self._project(attn_output, self.o_angles, self.o_scales, self.o_coupling, self._o_coupling_scale)
        return attn_output.reshape(*input_shape, -1), attn_weights


class GRTMixer(nn.Module):
    """Cross-head mixing at full width.

    Attention mixes within heads; this mixes across them with the same shuffle, modulated rotation and scale, using
    a block-diagonal coupling so the parameter count stays linear in `hidden_size`.
    """

    def __init__(self, config: GRTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_rotations = config.n_mixer_rotations
        self.n_pairs = config.hidden_size // 2
        self.group_size = next((g for g in (16, 8, 4, 2) if self.n_pairs % g == 0), 1)
        self.n_groups = self.n_pairs // self.group_size

        self.angles = nn.Parameter(torch.empty(self.n_rotations, self.n_pairs))
        self.scales = nn.Parameter(torch.empty(self.n_rotations, config.hidden_size))
        self.coupling = nn.Parameter(
            torch.empty(self.n_rotations, self.n_groups, 4 * self.group_size, self.group_size)
        )
        self.register_buffer("_coupling_scale", None, persistent=False)
        self._quant_bits = None

        perms, inv_perms = _build_butterfly_perms(self.n_rotations, config.hidden_size)
        self._perms_stack = nn.Buffer(perms)
        self._inv_perms_stack = nn.Buffer(inv_perms)

    @torch.no_grad()
    def quantize_weights(self, bits: int = 8) -> None:
        c = self.coupling
        if self._quant_bits is not None and self._coupling_scale is not None:
            c = _dequantize_symmetric(c, self._coupling_scale, bits=self._quant_bits, dim=-2)
        cq, cs = _quantize_symmetric(c, bits=bits, dim=-2)
        self.coupling = nn.Parameter(cq, requires_grad=False)
        self.register_buffer("_coupling_scale", cs, persistent=False)
        self._quant_bits = bits

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        for k in range(self.n_rotations):
            hidden_states = hidden_states[..., self._perms_stack[k]]

            flat = hidden_states.reshape(-1, self.hidden_size)
            num_tokens = flat.shape[0]
            pairs = flat.view(num_tokens, self.n_groups, self.group_size, 2)
            x0, x1 = pairs[..., 0], pairs[..., 1]

            features = _clifford_features(x0, x1, mixer_bivector=True)
            # eps 1e-2 here but 1e-6 in the projections: the trained weights depend on both
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-2)

            ck = (
                _dequantize_symmetric(self.coupling[k], self._coupling_scale[k], bits=self._quant_bits, dim=-2).to(
                    features.dtype
                )
                if self._coupling_scale is not None and self._quant_bits is not None
                else self.coupling[k]
            )
            modulation = torch.einsum("ngj,gji->ngi", features, ck)
            theta = self.angles[k].view(self.n_groups, self.group_size) + torch.tanh(modulation)

            cos, sin = torch.cos(theta), torch.sin(theta)
            rotated = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
            rotated = rotated.view(num_tokens, self.n_groups, 2 * self.group_size)
            rotated = rotated * F.softplus(self.scales[k]).view(self.n_groups, 2 * self.group_size)

            hidden_states = rotated.view(orig_shape)
            hidden_states = hidden_states[..., self._inv_perms_stack[k]]
        return hidden_states


class GRTFeedForward(nn.Module):
    """Feed-forward network built from rotation chains.

    The hidden state is zero-padded to `intermediate_size`, rotated, gated by `GRTRadialGELU`, rotated again and
    truncated back to `hidden_size`. The rotations mix signal into the padded dimensions, which is what makes the
    expansion possible without a weight matrix.
    """

    def __init__(self, config: GRTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.n_rotations = config.ffn_n_rotations
        n_pairs = self.intermediate_size // 2
        self.activation = GRTRadialGELU()

        self.expand_angles = nn.Parameter(torch.empty(self.n_rotations, n_pairs))
        self.expand_scales = nn.Parameter(torch.empty(self.n_rotations, self.intermediate_size))
        self.contract_angles = nn.Parameter(torch.empty(self.n_rotations, n_pairs))
        self.contract_scales = nn.Parameter(torch.empty(self.n_rotations, self.intermediate_size))

        if self.n_rotations > 1:
            perms, inv_perms = _build_butterfly_perms(self.n_rotations, self.intermediate_size)
            self._ffn_perms = nn.Buffer(perms)
            self._ffn_inv_perms = nn.Buffer(inv_perms)
        else:
            self._ffn_perms = self._ffn_inv_perms = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, (0, self.intermediate_size - self.hidden_size))
        hidden_states = _plain_rotate(
            hidden_states, self.expand_angles, self.expand_scales, self._ffn_perms, self._ffn_inv_perms
        )
        hidden_states = self.activation(hidden_states)
        hidden_states = _plain_rotate(
            hidden_states, self.contract_angles, self.contract_scales, self._ffn_perms, self._ffn_inv_perms
        )
        return hidden_states[..., : self.hidden_size]


class GRTDecoderLayer(GradientCheckpointingLayer):
    """One GRT block: attention, cross-head mixer and feed-forward, each pre-normed and added to the residual
    stream through a learnable gate.
    """

    def __init__(self, config: GRTConfig, layer_idx: int):
        super().__init__()
        self.gate_mode = config.residual_gate_mode if config.learnable_residual_gates else "none"
        if self.gate_mode == "free":
            self.gate_init = 1.0
        elif self.gate_mode == "bounded":
            self.gate_init = 0.0
        elif self.gate_mode == "bounded_depth":
            # Start the effective gate at 1/sqrt(num_hidden_layers), so the summed residual contributions grow as
            # the square root of the depth rather than linearly.
            target = 1.0 / max(1, config.num_hidden_layers) ** 0.5
            self.gate_init = math.atanh(min(0.9999, max(-0.9999, (target - 0.1) / 0.45 - 1.0)))
        else:
            self.gate_init = None

        if self.gate_init is None:
            self.res_gate_attn = self.res_gate_ffn = self.res_gate_mixer = None
        else:
            # The mixer gate exists whenever gates do, mixer or not, so checkpoints keep one layout.
            self.res_gate_attn = nn.Parameter(torch.empty(()))
            self.res_gate_ffn = nn.Parameter(torch.empty(()))
            self.res_gate_mixer = nn.Parameter(torch.empty(()))

        self.norm1 = GRTPearlNorm(config.hidden_size)
        self.attention = GRTAttention(config, layer_idx)

        self.mixer = None
        if config.n_mixer_rotations > 0:
            self.norm_mixer = GRTPearlNorm(config.hidden_size)
            self.mixer = GRTMixer(config)

        self.norm_ffn = GRTPearlNorm(config.hidden_size)
        self.ffn = GRTFeedForward(config)

        self.post_norm_attn = GRTPearlNorm(config.hidden_size) if config.peri_ln else None
        self.post_norm_ffn = GRTPearlNorm(config.hidden_size) if config.peri_ln else None
        self.res_norm_block = GRTPearlNorm(config.hidden_size) if config.residual_norm_mode == "pearlnorm" else None

    def _gate(self, raw: torch.Tensor | None) -> torch.Tensor | float:
        if raw is None:
            return 1.0
        if self.gate_mode == "bounded":
            return 1.0 + torch.tanh(raw) * 0.5
        if self.gate_mode == "bounded_depth":
            return 0.55 + torch.tanh(raw) * 0.45
        return raw

    @torch.no_grad()
    def quantize_weights(self, bits: int = 8) -> None:
        self.attention.quantize_weights(bits)
        if self.mixer is not None:
            self.mixer.quantize_weights(bits)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        attn_output, _ = self.attention(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        if self.post_norm_attn is not None:
            attn_output = self.post_norm_attn(attn_output)
        hidden_states = hidden_states + self._gate(self.res_gate_attn) * attn_output

        if self.mixer is not None:
            mixer_output = self.mixer(self.norm_mixer(hidden_states))
            hidden_states = hidden_states + self._gate(self.res_gate_mixer) * mixer_output

        ffn_output = self.ffn(self.norm_ffn(hidden_states))
        if self.post_norm_ffn is not None:
            ffn_output = self.post_norm_ffn(ffn_output)
        hidden_states = hidden_states + self._gate(self.res_gate_ffn) * ffn_output

        if self.res_norm_block is not None:
            hidden_states = self.res_norm_block(hidden_states)
        return hidden_states


class GRTCosineHead(nn.Module):
    """Language modeling head: the cosine between the hidden state and every token embedding, times a learned
    temperature.

    A token embedding is `exp(log_magnitude) * (cos, sin)` of its phase angles, with norm
    `exp(log_magnitude) * sqrt(hidden_size / 2)`, so its unit vector is `(cos, sin) / sqrt(hidden_size / 2)` and
    only the phase angles reach the head. They are tied to the input embedding's by default.
    """

    def __init__(self, config: GRTConfig):
        super().__init__()
        self.phase_angles = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size // 2))
        self.output_scale = nn.Parameter(torch.empty(()))
        self.register_buffer("_unit_table_q", None, persistent=False)
        self.register_buffer("_unit_table_scale", None, persistent=False)
        self._quant_bits = None

    @torch.no_grad()
    def quantize_weights(
        self,
        bits: int = 8,
        unit_table: torch.Tensor | None = None,
        shared_qweight: torch.Tensor | None = None,
    ) -> None:
        if unit_table is None:
            if self._unit_table_q is not None and self._unit_table_scale is not None and self._quant_bits is not None:
                unit_table = _dequantize_symmetric(
                    self._unit_table_q, self._unit_table_scale, bits=self._quant_bits, dim=-1
                )
            else:
                unit_table = torch.stack([torch.cos(self.phase_angles), torch.sin(self.phase_angles)], dim=-1).flatten(
                    -2
                )
                unit_table = unit_table * self.phase_angles.shape[-1] ** -0.5
        uq, us = _quantize_symmetric(unit_table, bits=bits, dim=-1)
        self.register_buffer("_unit_table_q", shared_qweight if shared_qweight is not None else uq, persistent=False)
        self.register_buffer("_unit_table_scale", us, persistent=False)
        self._quant_bits = bits
        self.phase_angles = nn.Parameter(
            torch.empty(0, dtype=unit_table.dtype, device=unit_table.device), requires_grad=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states / (hidden_states.norm(dim=-1, keepdim=True) + 1e-6)
        if self._unit_table_q is not None and self._unit_table_scale is not None and self._quant_bits is not None:
            unit_table = _dequantize_symmetric(
                self._unit_table_q, self._unit_table_scale, bits=self._quant_bits, dim=-1
            ).to(hidden_states.dtype)
        else:
            unit_table = torch.stack([torch.cos(self.phase_angles), torch.sin(self.phase_angles)], dim=-1).flatten(-2)
            unit_table = unit_table * self.phase_angles.shape[-1] ** -0.5
        return (hidden_states @ unit_table.T) * F.softplus(self.output_scale)


@auto_docstring
class GRTPreTrainedModel(PreTrainedModel):
    config: GRTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GRTDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _input_embed_layer = "embedding"
    _supports_sdpa = True
    # Position angles are `position * exp(log_freqs)`, so the frequencies keep fp32 precision in a bf16/fp16 model.
    _keep_in_fp32_modules_strict = ["pos_encoding"]
    _can_record_outputs = {
        "hidden_states": GRTDecoderLayer,
        "attentions": GRTAttention,
    }

    @torch.no_grad()
    def quantize_weights(self, bits: int = 8):
        """Quantize the materialized embedding/LM-head tables and stage coupling weights to `bits`-bit (`8` or `4`)
        while keeping rotation angles, scales, and norms in floating point.
        """
        base = getattr(self, "model", self)
        unit_table = None
        shared_q = None
        if hasattr(base, "embedding") and isinstance(base.embedding, GRTEmbedding):
            if base.embedding.phase_angles.numel() > 0:
                pa = base.embedding.phase_angles
                unit_table = torch.stack([torch.cos(pa), torch.sin(pa)], dim=-1).flatten(-2) * (pa.shape[-1] ** -0.5)
            else:
                table = base.embedding.table()
                unit_table = table / (table.norm(dim=-1, keepdim=True) + 1e-8)
            base.embedding.quantize_weights(bits)
            if getattr(self.config, "tie_word_embeddings", True):
                shared_q = base.embedding._table_q
        if hasattr(base, "blocks"):
            for block in base.blocks:
                block.quantize_weights(bits)
        if hasattr(self, "lm_head") and isinstance(self.lm_head, GRTCosineHead):
            self.lm_head.quantize_weights(bits, unit_table=unit_table, shared_qweight=shared_q)
        return self

    @torch.no_grad()
    def _init_weights(self, module):
        """Angular initialization.

        GRT has angles, scales and couplings rather than weight matrices, so every tensor is initialized in the
        geometry it lives in. The `init` helpers skip tensors already loaded from a checkpoint.
        """
        super()._init_weights(module)
        config = self.config
        if isinstance(module, (GRTEmbedding, GRTCosineHead)):
            init.normal_(module.phase_angles, mean=0.0, std=config.phase_init_scale)
            if isinstance(module, GRTEmbedding):
                init.constant_(module.log_magnitude, config.log_magnitude_init)
            else:
                init.constant_(module.output_scale, _raw_scale_init(config.output_scale_init))
        elif isinstance(module, GRTPearlNorm):
            init.ones_(module.scale)
        elif isinstance(module, GRTRotaryEmbedding):
            init.copy_(module.log_freqs, _default_log_freqs(config))
        elif isinstance(module, GRTAttention):
            for name in ("q", "k", "v", "o"):
                init.normal_(getattr(module, f"{name}_angles"), mean=0.0, std=config.angle_init_scale)
                _init_scale_tensor(getattr(module, f"{name}_scales"), 1.0)
                _init_coupling_tensor(
                    getattr(module, f"{name}_coupling"), config.coupling_init_scale, config.coupling_init_mode
                )
            _init_butterfly_buffers(module._attn_perms, module._attn_inv_perms)
        elif isinstance(module, GRTMixer):
            init.normal_(module.angles, mean=0.0, std=config.angle_init_scale)
            _init_scale_tensor(module.scales, 1.0)
            _init_coupling_tensor(module.coupling, config.coupling_init_scale, config.coupling_init_mode)
            _init_butterfly_buffers(module._perms_stack, module._inv_perms_stack)
        elif isinstance(module, GRTFeedForward):
            init.normal_(module.expand_angles, mean=0.0, std=config.angle_init_scale)
            init.normal_(module.contract_angles, mean=0.0, std=config.angle_init_scale)
            _init_scale_tensor(module.expand_scales, 1.0)
            _init_scale_tensor(module.contract_scales, 1.0)
            _init_butterfly_buffers(module._ffn_perms, module._ffn_inv_perms)
        elif isinstance(module, GRTDecoderLayer):
            for gate in (module.res_gate_attn, module.res_gate_ffn, module.res_gate_mixer):
                if gate is not None:
                    init.constant_(gate, module.gate_init)


def _init_butterfly_buffers(perms: torch.Tensor | None, inv_perms: torch.Tensor | None) -> None:
    """Rebuild the butterfly permutations, which a model materialized from the meta device holds empty."""
    if perms is None:
        return
    new_perms, new_inv_perms = _build_butterfly_perms(perms.shape[0], perms.shape[1])
    init.copy_(perms, new_perms.to(perms.device))
    init.copy_(inv_perms, new_inv_perms.to(inv_perms.device))


def _init_scale_tensor(tensor: torch.Tensor, target_scale: float, spread: float = 0.01) -> None:
    """Initialize a raw scale around `target_scale` with a small per-element spread.

    Identical scales would receive near-identical gradients and stay stuck together.
    """
    if getattr(tensor, "_is_hf_initialized", False):
        return
    raw_center = _raw_scale_init(target_scale)
    # Divide by the softplus slope at the center so the spread is ~`spread` after the softplus.
    raw_spread = spread / max(1.0 / (1.0 + math.exp(-raw_center)), 0.1)
    values = torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device)
    init.copy_(tensor, values * raw_spread + raw_center)


def _init_coupling_tensor(tensor: torch.Tensor, scale: float, mode: str) -> None:
    """Initialize a coupling tensor around zero.

    `diagonal` adds a self-coupling term, so each pair starts out modulated mostly by its own features.
    """
    if getattr(tensor, "_is_hf_initialized", False):
        return
    values = torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device) * scale
    if mode == "diagonal":
        dim_out, dim_in = tensor.shape[-1], tensor.shape[-2]
        eye = torch.eye(dim_out, device=tensor.device, dtype=tensor.dtype) * (scale * 10)
        if dim_in == dim_out:
            values = values + eye
        elif dim_in > dim_out and dim_in % dim_out == 0:
            values[..., :dim_out, :] = values[..., :dim_out, :] + eye
    init.copy_(tensor, values)


@auto_docstring
class GRTModel(GRTPreTrainedModel):
    def __init__(self, config: GRTConfig):
        super().__init__(config)
        self.embedding = GRTEmbedding(config)
        self.pos_encoding = GRTRotaryEmbedding(config)
        self.blocks = nn.ModuleList(
            [GRTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = GRTPearlNorm(config.hidden_size)

        self.post_init()

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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Position is applied once, as a rotation of the input embedding; the blocks never see position ids.
        hidden_states = self.pos_encoding(inputs_embeds, position_ids)

        for block in self.blocks[: self.config.num_hidden_layers]:
            hidden_states = block(
                hidden_states,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.final_norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class GRTForCausalLM(GRTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.phase_angles": "model.embedding.phase_angles"}

    def __init__(self, config: GRTConfig):
        super().__init__(config)
        self.model = GRTModel(config)
        self.lm_head = GRTCosineHead(config)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, GRTForCausalLM

        >>> model = GRTForCausalLM.from_pretrained("matterwaveai/grt-30m")
        >>> tokenizer = AutoTokenizer.from_pretrained("matterwaveai/grt-30m")

        >>> prompt = "Once upon a time"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> generate_ids = model.generate(inputs.input_ids, max_new_tokens=8, do_sample=False)
        >>> text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["GRTConfig", "GRTForCausalLM", "GRTModel", "GRTPreTrainedModel"]
