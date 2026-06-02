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
"""PyTorch ESMC model."""

import math
import re
from dataclasses import dataclass
from typing import Callable, Optional, cast

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from ...masking_utils import create_bidirectional_mask  # type: ignore[import]
from ...modeling_outputs import (  # type: ignore[import]
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel  # type: ignore[import]
from ...utils import (  # type: ignore[import]
    auto_docstring,
    can_return_tuple,
    logging,
)
from .configuration_esmc import ESMCConfig
from .modeling_esmc_sae import _ESMCSAELayer

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ESMCConfig"

# Transformer Engine: fused LayerNorm+Linear / LayerNorm+MLP kernels with
# fp32 reduction inside the LayerNorm. Recommended on GPU for accurate bf16
# inference; without it the pure-PyTorch fallback drifts ~O(10) in fp32 and
# ~O(100) in bf16 on the unnormalized residual stream (perplexity stays
# within rounding noise).
try:
    import transformer_engine.pytorch as te  # type: ignore[import-untyped]

    _te_available = True
except ImportError:
    te = None  # type: ignore[assignment]
    _te_available = False

if not _te_available:
    logger.warning(
        "ESMC: transformer_engine is not installed; falling back to "
        "pure-PyTorch LayerNorm+Linear / LayerNorm+MLP. Outputs will differ "
        "numerically — measured on the unnormalized residual stream (before "
        "the final LayerNorm), ~O(10) max-diff in fp32 and ~O(100) in bf16; "
        "after the final LayerNorm these shrink to a few ULP and perplexity "
        "stays within rounding noise. Install with "
        "`pip install transformer-engine[pytorch]` to enable fused fp32-"
        "reduction LayerNorm."
    )

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ESMCOutput(ModelOutput):
    """
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, d_model)`):
            Sequence of hidden states at the output of the last layer, after layer normalisation.
        hidden_states (`torch.FloatTensor`, *optional*):
            Stacked hidden states for all encoder layers.
            Shape ``(n_layers, batch_size, sequence_length, d_model)``.
            Returned when ``output_hidden_states=True``.
        sae_outputs (`dict[str, torch.Tensor]`, *optional*):
            SAE feature magnitudes keyed by SAE model name (sparse tensors).
            Only populated when SAE models have been registered via
            ``add_sae_models`` and ``compute_sae=True``.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Per-layer attention weights of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``.
            Returned when ``output_attentions=True``.  Not available on the
            ``flash_attention_2`` path.
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None
    sae_outputs: dict[str, torch.Tensor] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ESMCMaskedLMOutput(MaskedLMOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Masked language modelling loss. Returned when ``labels`` are provided.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Prediction scores of the language modelling head.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, d_model)`):
            Final hidden states after layer normalisation.
        hidden_states (`torch.FloatTensor`, *optional*):
            Stacked hidden states. Shape ``(n_layers, batch_size, sequence_length, d_model)``.
        sae_outputs (`dict[str, torch.Tensor]`, *optional*):
            SAE feature magnitudes keyed by SAE model name (sparse tensors).
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Per-layer attention weights of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``.
            Returned when ``output_attentions=True``.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None
    sae_outputs: dict[str, torch.Tensor] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ESMCTokenClassifierOutput(TokenClassifierOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Token classification loss. Returned when ``labels`` are provided.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_labels)`):
            Classification scores (before SoftMax).
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, d_model)`):
            Final hidden states after layer normalisation.
        hidden_states (`torch.FloatTensor`, *optional*):
            Stacked hidden states. Shape ``(n_layers, batch_size, sequence_length, d_model)``.
        sae_outputs (`dict[str, torch.Tensor]`, *optional*):
            SAE feature magnitudes keyed by SAE model name (sparse tensors).
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Per-layer attention weights of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``.
            Returned when ``output_attentions=True``.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None
    sae_outputs: dict[str, torch.Tensor] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ESMCSequenceClassifierOutput(SequenceClassifierOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Sequence classification loss. Returned when ``labels`` are provided.
        logits (`torch.FloatTensor` of shape `(batch_size, num_labels)`):
            Classification scores (before SoftMax).
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, d_model)`):
            Final hidden states after layer normalisation.
        hidden_states (`torch.FloatTensor`, *optional*):
            Stacked hidden states. Shape ``(n_layers, batch_size, sequence_length, d_model)``.
        sae_outputs (`dict[str, torch.Tensor]`, *optional*):
            SAE feature magnitudes keyed by SAE model name (sparse tensors).
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Per-layer attention weights of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``.
            Returned when ``output_attentions=True``.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None
    sae_outputs: dict[str, torch.Tensor] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


# ---------------------------------------------------------------------------
# Rotary position embedding helpers
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2, -1)


def _apply_rotary_emb_torch(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    """Apply rotary position embeddings (pure PyTorch, no Triton dependency).

    Args:
        x: ``(batch, seqlen, n_heads, head_dim)``
        cos: ``(seqlen, rotary_dim / 2)``
        sin: ``(seqlen, rotary_dim / 2)``
    """
    ro_dim = cos.shape[-1] * 2
    seqlen = x.size(1)
    cos = cos[:seqlen].unsqueeze(1).repeat(1, 1, 2)
    sin = sin[:seqlen].unsqueeze(1).repeat(1, 1, 2)
    return torch.cat(
        [
            x[..., :ro_dim] * cos + _rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE) as described in `RoFormer`_.

    .. _RoFormer: https://arxiv.org/abs/2104.09864

    Args:
        dim: Size of a single attention head.
        base: Frequency base for the sinusoidal positions.
        interleaved: If ``True`` rotate adjacent pairs (GPT-J style) instead of
            splitting the head dimension in half (GPT-NeoX style).
        scaling_factor: Linear scaling factor applied to position indices.
        pos_idx_in_fp32: Compute position indices in float32 to avoid bf16
            rounding errors at large sequence lengths.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: float | None = None,
        scaling_factor: float = 1.0,
        pos_idx_in_fp32: bool = True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.pos_idx_in_fp32 = pos_idx_in_fp32

        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._cos_k_cached: torch.Tensor | None = None
        self._sin_k_cached: torch.Tensor | None = None

        self.reset_parameters(device=device)

    def reset_parameters(self, device=None):
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        arange = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
        scale = (
            (arange + 0.4 * self.dim) / (1.4 * self.dim)
            if self.scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

    def _compute_inv_freq(self, device=None) -> torch.Tensor:
        return 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen: int, device=None, dtype=None):
        if self.inv_freq.is_meta:
            self.reset_parameters(device=device)
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = (
                    torch.arange(seqlen, device=device, dtype=torch.float32)
                    / self.scaling_factor
                )
                inv_freq = (
                    self.inv_freq.to(torch.float32)
                    if self.inv_freq.dtype != torch.float32
                    else self.inv_freq
                )
            else:
                t = (
                    torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)  # type: ignore[call-overload]
                    / self.scaling_factor
                )
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)  # type: ignore[arg-type]

            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                _scale: torch.Tensor = self.scale  # type: ignore[assignment]
                power = (
                    torch.arange(seqlen, dtype=_scale.dtype, device=_scale.device)
                    - seqlen // 2
                ) / self.scale_base  # type: ignore[operator]
                scale = _scale.to(device=power.device) ** power.unsqueeze(-1)
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def _apply(self, fn, recurse=True):
        if self.inv_freq.is_meta:
            self.reset_parameters(device="cpu")
        result = super()._apply(fn, recurse=recurse)
        # Recompute inv_freq on the new device: CPU vs CUDA ``pow`` differ by
        # ~1 fp32 ULP, which compounds across attention layers. Keep this
        # buffer fp32 even when the module is cast to bf16/fp16; otherwise the
        # rounded RoPE frequencies drift from the internal ESMC path.
        new_inv_freq = self._compute_inv_freq(device=self.inv_freq.device)
        self.register_buffer("inv_freq", new_inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        return result

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seqlen_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            q: ``(batch, seqlen, n_heads, head_dim)``
            k: ``(batch, seqlen, n_heads, head_dim)``
            seqlen_offset: Offset used in incremental decoding.

        Returns:
            Tuple of rotated ``(q, k)`` tensors with the same shape as the inputs.
        """
        self._update_cos_sin_cache(
            q.shape[1] + seqlen_offset, device=q.device, dtype=q.dtype
        )
        assert self._cos_cached is not None and self._sin_cached is not None

        if self.scale is not None:
            raise NotImplementedError("XPos scaling is not supported in this path.")

        cos = self._cos_cached[seqlen_offset:]
        sin = self._sin_cached[seqlen_offset:]

        q_rot = _apply_rotary_emb_torch(q, cos, sin, self.interleaved)
        k_rot = _apply_rotary_emb_torch(k, cos, sin, self.interleaved)
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Feed-forward network helpers
# ---------------------------------------------------------------------------


def _swiglu_hidden_dim(expansion_ratio: float, d_model: int) -> int:
    """Round hidden dim to the nearest multiple of 256 after applying expansion_ratio."""
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class _SwiGLU(nn.Module):
    """SwiGLU activation: ``silu(x1) * x2`` where ``x`` is split along the last dim."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class _PyTorchLayerNormLinear(nn.Module):
    """LayerNorm followed by a Linear projection, sharing the parameter
    names ``layer_norm_weight``, ``layer_norm_bias`` and ``weight`` so the
    state-dict layout matches the accelerated TE module loaded on GPU.
    """

    def __init__(self, d_in: int, d_out: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_in = d_in
        self.eps = eps
        self.layer_norm_weight = nn.Parameter(torch.ones(d_in))
        self.layer_norm_bias = nn.Parameter(torch.zeros(d_in))
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(
            x, (self.d_in,), self.layer_norm_weight, self.layer_norm_bias, self.eps
        )
        return F.linear(x, self.weight)


class _PyTorchLayerNormMLP(nn.Module):
    """LayerNorm + SwiGLU MLP, sharing the parameter names
    ``layer_norm_weight``, ``layer_norm_bias``, ``fc1_weight``,
    ``fc2_weight`` so the state-dict layout matches the accelerated TE
    module loaded on GPU.
    """

    def __init__(
        self, hidden_size: int, ffn_hidden_size: int, eps: float = 1e-5
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.eps = eps
        self.layer_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.layer_norm_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc1_weight = nn.Parameter(torch.empty(2 * ffn_hidden_size, hidden_size))
        self.fc2_weight = nn.Parameter(torch.empty(hidden_size, ffn_hidden_size))
        nn.init.normal_(self.fc1_weight, std=0.02)
        nn.init.normal_(self.fc2_weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(
            x,
            (self.hidden_size,),
            self.layer_norm_weight,
            self.layer_norm_bias,
            self.eps,
        )
        x = F.linear(x, self.fc1_weight)
        x1, x2 = x.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        return F.linear(x, self.fc2_weight)


def _swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool) -> nn.Module:
    """LayerNorm + SwiGLU MLP. Uses Transformer Engine's fused LN+MLP when
    available; otherwise returns the pure-PyTorch fallback with matching
    state-dict layout."""
    assert not bias, "ESMC was trained with bias=False; bias=True not supported"
    hidden = _swiglu_hidden_dim(expansion_ratio, d_model)
    if _te_available:
        return te.LayerNormMLP(  # type: ignore[union-attr]
            hidden_size=d_model,
            ffn_hidden_size=hidden,
            bias=bias,
            activation="swiglu",
            init_method=None,
            output_layer_init_method=None,
        )
    return _PyTorchLayerNormMLP(hidden_size=d_model, ffn_hidden_size=hidden)


def _make_attn_layernorm_qkv(d_model: int, bias: bool) -> nn.Module:
    """LayerNorm + fused QKV projection. Uses Transformer Engine when
    available; pure-PyTorch fallback otherwise."""
    assert not bias, "ESMC was trained with bias=False; bias=True not supported"
    if _te_available:
        return te.LayerNormLinear(  # type: ignore[union-attr]
            d_model, d_model * 3, bias=bias, init_method=None
        )
    return _PyTorchLayerNormLinear(d_model, d_model * 3)


def _make_attn_out_proj(d_model: int, bias: bool) -> nn.Module:
    """Attention output projection. Uses Transformer Engine when available;
    pure-PyTorch ``nn.Linear`` otherwise."""
    if _te_available:
        return te.Linear(  # type: ignore[union-attr]
            d_model, d_model, bias=bias, init_method=None
        )
    return nn.Linear(d_model, d_model, bias=bias)


def _gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool) -> nn.Sequential:
    hidden = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden, bias=bias),
        nn.GELU(),
        nn.Linear(hidden, d_model, bias=bias),
    )


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference attention ``softmax(Q @ Kᵀ * scaling + mask) @ V``.

    Operates on ``(batch, n_heads, seq_len, head_dim)`` query/key/value and
    returns ``(attn_output, attn_weights)``; ``attn_output`` is transposed back
    to ``(batch, seq_len, n_heads, head_dim)`` to match the other
    ``ALL_ATTENTION_FUNCTIONS`` backends.
    """
    if scaling is None:
        scaling = query.size(-1) ** -0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with QK LayerNorm and RoPE.

    Args:
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        bias: Whether to use bias in linear layers.
        qk_layernorm: Whether to apply LayerNorm to queries and keys before
            computing attention scores.
    """

    def __init__(
        self,
        config: ESMCConfig,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scaling = self.d_head**-0.5
        self.attention_dropout = 0.0

        assert not bias, "ESMC was trained with bias=False; bias=True not supported"
        self.layernorm_qkv = _make_attn_layernorm_qkv(d_model, bias)
        self.out_proj = _make_attn_out_proj(d_model, bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(context, attn_weights)``.

        Attention is computed by the backend selected through
        ``config._attn_implementation`` (``eager`` / ``sdpa`` /
        ``flash_attention_2`` / ...). RoPE and QK-LayerNorm are applied here; the
        backend only computes ``softmax(QKᵀ)V``. ``attn_weights`` is ``None``
        unless the backend exposes the probabilities — ``output_attentions=True``
        forces the ``eager`` interface so they are observable.
        """
        b, s, _ = hidden_states.shape
        qkv = self.layernorm_qkv(hidden_states)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = self.q_ln(q).to(q.dtype)
        k = self.k_ln(k).to(q.dtype)
        q, k = self._apply_rotary(q, k)

        # (B, S, D) -> (B, H, S, Dh)
        q = q.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        attention_interface: Callable = eager_attention_forward
        if not output_attentions:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )

        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(b, s, -1)
        return self.out_proj(attn_output), attn_weights


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class UnifiedTransformerBlock(nn.Module):
    """Single transformer block: pre-norm attention + pre-norm FFN with residual scaling.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        use_flash_attn: Use Flash Attention 2 kernel if available.
        bias: Whether linear layers include bias terms.
        expansion_ratio: Hidden-dim expansion ratio for the FFN.
        residue_scaling_factor: Scales residual connections to stabilise deep
            networks (``1 / sqrt(n_layers / 36)`` is the ESM3 scheme).
        qk_layernorm: Whether to apply QK LayerNorm in attention.
        ffn_type: Feed-forward activation: ``"swiglu"`` or ``"gelu"``.
    """

    def __init__(
        self,
        config: ESMCConfig,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1.0,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()

        self.attn = MultiHeadAttention(
            config, d_model, n_heads, bias=bias, qk_layernorm=qk_layernorm
        )

        if ffn_type == "swiglu":
            self.ffn = _swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = _gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(
                f"Unknown ffn_type: {ffn_type!r}. Choose 'swiglu' or 'gelu'."
            )

        self.scaling_factor = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: ``(batch, seq_len, d_model)``
            attention_mask: Additive attention bias broadcastable to
                ``(batch, num_heads, seq_len, seq_len)``, or ``None`` for full
                (unmasked) attention.
            output_attentions: When ``True``, returns the per-head attention
                weights for this block alongside the residual output.

        Returns:
            ``(output, attn_weights_or_None)``.  Shape of ``output`` is
            ``(batch, seq_len, d_model)``; ``attn_weights`` shape is
            ``(batch, num_heads, seq_len, seq_len)`` or ``None``.
        """
        attn_out, attn_weights = self.attn(
            x, attention_mask, output_attentions=output_attentions, **kwargs
        )
        x = x + attn_out / self.scaling_factor
        x = x + self.ffn(x) / self.scaling_factor
        return x, attn_weights


class TransformerStack(nn.Module):
    """Stack of :class:`UnifiedTransformerBlock` layers with a final LayerNorm.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        scale_residue: When ``True`` apply ESM3 residue scaling
            ``sqrt(n_layers / 36)`` to each block.
        bias: Bias flag forwarded to every sub-module.
        qk_layernorm: QK LayerNorm flag forwarded to every block.
        ffn_type: FFN activation type (``"swiglu"`` or ``"gelu"``).
        expansion_ratio: FFN expansion ratio.
        use_flash_attn: Use Flash Attention 2 kernel when available.
    """

    def __init__(
        self,
        config: ESMCConfig,
        d_model: int,
        n_heads: int,
        n_layers: int,
        scale_residue: bool = True,
        bias: bool = False,
        qk_layernorm: bool = True,
        ffn_type: str = "swiglu",
        expansion_ratio: float = 8 / 3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    config,
                    d_model,
                    n_heads,
                    residue_scaling_factor=math.sqrt(n_layers / 36)
                    if scale_residue
                    else 1.0,
                    expansion_ratio=expansion_ratio,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                    ffn_type=ffn_type,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layers_to_collect: list[int] | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...] | None,
    ]:
        """Run the full transformer stack.

        Args:
            x: ``(batch, seq_len, d_model)``
            attention_mask: Additive attention bias forwarded to each block, or
                ``None`` for full attention.
            layers_to_collect: Layer indices (0-based pre-block inputs plus
                ``n_layers`` for the post-norm output) whose hidden states
                should be returned.
            output_attentions: When ``True``, collects the per-block attention
                weights and returns them as the fourth tuple element.

        Returns:
            ``(post_norm, pre_norm, hidden_states, attentions)`` where
            ``hidden_states`` is a (possibly empty) tuple of tensors and
            ``attentions`` is a tuple of per-block ``(B, H, L, L)`` tensors
            or ``None`` when ``output_attentions`` is ``False``.
        """
        if layers_to_collect is None:
            layers_to_collect = []

        collected: list[torch.Tensor] = []
        all_attentions: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx in layers_to_collect:
                collected.append(x)
            x, attn_weights = block(
                x, attention_mask, output_attentions=output_attentions, **kwargs
            )
            if output_attentions and attn_weights is not None:
                all_attentions.append(attn_weights)

        norm_x = self.norm(x)
        if len(self.blocks) in layers_to_collect:
            collected.append(norm_x)

        attentions = tuple(all_attentions) if output_attentions else None
        return norm_x, x, tuple(collected), attentions


# ---------------------------------------------------------------------------
# Pre-trained model base class
# ---------------------------------------------------------------------------


@auto_docstring
class ESMCPreTrainedModel(PreTrainedModel):
    """Base class for ESMC models.

    Handles weight initialisation and declares module-level capabilities.
    """

    config_class = ESMCConfig
    base_model_prefix = "esmc"
    supports_gradient_checkpointing = False
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True
    _no_split_modules = ["UnifiedTransformerBlock"]
    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$"]

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, RotaryEmbedding):
            module.reset_parameters(device=self.device)


# ---------------------------------------------------------------------------
# Base encoder model
# ---------------------------------------------------------------------------


@auto_docstring
class ESMCModel(ESMCPreTrainedModel):
    """The bare ESMC encoder outputting raw hidden states.

    ESMC is a protein language model trained by EvolutionaryScale using a
    masked-token objective over amino acid sequences.  The architecture is a
    standard Transformer encoder with RoPE positional embeddings, QK LayerNorm,
    and SwiGLU feed-forward networks.

    Args:
        config: An :class:`ESMCConfig` instance.
    """

    def __init__(self, config: ESMCConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.transformer = TransformerStack(
            config,
            config.d_model,
            config.n_heads,
            config.n_layers,
        )
        self._sae_models: nn.ModuleDict = nn.ModuleDict()
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed = value

    def add_sae_models(self, sae_models: list[_ESMCSAELayer]) -> None:
        """Register one or more SAEs obtained from an :class:`ESMCSAEModel`.

        Each is keyed by ``f"layer{N}"`` (the backbone-layer index ``N`` the
        SAE is trained against, set by
        :meth:`ESMCSAEModel.initialize_layers`). Attaching two SAEs for the
        same backbone layer raises — only one SAE per layer can be active.

        Example::

            sae = ESMCSAEModel.from_pretrained(
                "biohub/esmc-600m-2024-12-sae-k64-codebook16384"
            )
            sae.initialize_layers([27, 33])
            model.add_sae_models([sae.layers["27"], sae.layers["33"]])
        """
        for layer in sae_models:
            assert isinstance(layer, _ESMCSAELayer), (
                f"Expected an SAE layer (model.layers['<idx>']), got "
                f"{type(layer).__name__}."
            )
            key = f"layer{int(layer.layer)}"
            if key in self._sae_models:
                raise ValueError(
                    f"An SAE is already registered at {key!r}. Only one SAE "
                    "per backbone layer can be active — pick a different "
                    "layer on one of them, or attach in a fresh model."
                )
            self._sae_models[key] = layer

    _SAE_KEY_RE = re.compile(r"layer(\d+)")

    def _get_sae_layer_num_requested(self, model_name: str) -> int:
        """Recover the backbone-layer index from a key written by
        :meth:`add_sae_models` (``"layer{N}"`` → ``N``)."""
        match = self._SAE_KEY_RE.fullmatch(model_name)
        assert (
            match is not None
        ), f"Unexpected SAE key {model_name!r}; expected 'layer{{N}}'."
        return int(match.group(1))

    def _validate_sae_inputs(self, input_ids: torch.Tensor) -> None:
        assert torch.all(input_ids != self.config.mask_token_id), (
            "SAE inputs must not contain mask tokens. "
            "SAEs were trained on unmasked sequences."
        )

    def _get_sae_outputs(
        self,
        hidden_states: torch.Tensor,
        layers_to_collect: list[int],
        token_mask: torch.Tensor,
        normalize_sae: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run all registered SAEs and return their feature magnitudes.

        Args:
            hidden_states: Stacked tensor of shape
                ``(len(layers_to_collect), batch, seq_len, d_model)``.
            layers_to_collect: The ESMC layer indices that were collected,
                in the same order as the first dim of ``hidden_states``.
            token_mask: Boolean mask ``(batch, seq_len)`` — ``True`` for
                real (non-padding) tokens.
            normalize_sae: When ``True``, scale features by ``idf / max``
                using the per-feature stats trained alongside each SAE.
        """
        layer_to_idx = {layer: idx for idx, layer in enumerate(layers_to_collect)}
        sae_outputs: dict[str, torch.Tensor] = {}

        for model_name, sae_module in self._sae_models.items():
            # `nn.ModuleDict` only stores `nn.Module`s at the type level;
            # ``add_sae_models`` enforces that each entry is an ``_ESMCSAELayer``.
            assert isinstance(sae_module, _ESMCSAELayer)
            layer: _ESMCSAELayer = sae_module
            requested_layer = self._get_sae_layer_num_requested(model_name)
            layer_idx = layer_to_idx[requested_layer]
            layer_states = hidden_states[layer_idx].clone().to(self.device)

            sae_out = layer.get_sae_output(layer_states, token_mask)
            features = sae_out.feature_magnitudes.detach()

            if normalize_sae:
                # ``register_buffer`` is typed as ``Tensor | Module`` on
                # ``nn.Module``; narrow here since these are Tensors.
                idf = cast(torch.Tensor, layer.idf)
                max_val = cast(torch.Tensor, layer.max)
                features = (features / max_val) * idf

            sae_outputs[model_name] = features.to_sparse()

        return sae_outputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_id: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        compute_sae: bool = True,
        normalize_sae: bool = False,
    ) -> tuple[torch.Tensor, ...] | ESMCOutput:
        r"""
        sequence_id (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Integer chain-ID tensor for chain-aware attention masking. Tokens with the same
            non-negative integer value can attend to each other; tokens with different values
            cannot (cross-chain masking). Padding positions should be set to ``-1``.
            When provided, ``attention_mask`` is ignored. The ``flash_attention_2`` backend
            only supports single-chain inputs (all non-padding values must be ``0``); pass
            multi-chain ``sequence_id`` with ``attn_implementation='sdpa'`` (or ``'eager'``).
        output_attentions (`bool`, *optional*):
            Whether to return the per-block attention weights of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``.
            Forces a manual-SDPA path inside :class:`MultiHeadAttention` so the
            attention probabilities are observable; raises on the
            ``flash_attention_2`` path.
        compute_sae (`bool`, *optional*, defaults to ``True``):
            Whether to run any SAE models registered via :meth:`add_sae_models`.
            Has no effect when no SAEs are registered.
        normalize_sae (`bool`, *optional*, defaults to ``False``):
            When ``True``, scale SAE feature magnitudes by ``idf / max`` (only
            applied when the SAE's normalization buffers contain non-trivial values).

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ESMCModel

        >>> model = ESMCModel.from_pretrained("Biohub/ESMC-600M-2024-12")
        >>> tokenizer = AutoTokenizer.from_pretrained("Biohub/ESMC-600M-2024-12")
        >>> inputs = tokenizer(["MLKNVQVQLV"], return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> outputs.last_hidden_state.shape
        torch.Size([1, 12, 960])
        ```
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        output_sae = compute_sae and len(self._sae_models) > 0

        # Determine which intermediate layers to collect.  When SAEs are
        # registered we must collect at least the layers they target, even if
        # the caller did not ask for all hidden states.
        if output_hidden_states:
            layers_to_collect: list[int] = list(range(self.config.n_layers + 1))
        elif output_sae:
            layers_to_collect = sorted(
                {self._get_sae_layer_num_requested(name) for name in self._sae_models}
            )
        else:
            layers_to_collect = []

        user_supplied_sequence_id = sequence_id is not None
        if user_supplied_sequence_id:
            bool_mask = sequence_id >= 0
        else:
            if attention_mask is None:
                attention_mask = input_ids != self.config.pad_token_id
            assert attention_mask is not None
            bool_mask = attention_mask.bool()

        x = self.embed(input_ids)

        if user_supplied_sequence_id:
            if self.config._attn_implementation == "flash_attention_2" and (
                sequence_id > 0
            ).any():
                raise ValueError(
                    "Multi-chain ``sequence_id`` (any value > 0) is not "
                    "supported with attn_implementation='flash_attention_2'. "
                    "Re-load the model with attn_implementation='sdpa' (or "
                    "'eager') for chain-aware attention masking."
                )
            # Block-diagonal chain mask: a token attends only to tokens sharing
            # its ``sequence_id``. Additive bias broadcast over heads, shape
            # ``(batch, 1, seq_len, seq_len)``; handled by the eager / sdpa paths.
            same_chain = sequence_id.unsqueeze(-1) == sequence_id.unsqueeze(-2)
            attn_bias = torch.zeros(
                same_chain.shape, dtype=x.dtype, device=x.device
            ).masked_fill_(~same_chain, torch.finfo(x.dtype).min)
            attn_bias = attn_bias.unsqueeze(1)
        else:
            attn_bias = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=x,
                attention_mask=attention_mask,
            )

        last_hidden_state, _, collected, attentions = self.transformer(
            x,
            attention_mask=attn_bias,
            layers_to_collect=layers_to_collect,
            output_attentions=output_attentions,
        )

        # Stack once; reused for both SAE and hidden-state output.
        collected_tensor: torch.Tensor | None = (
            torch.stack(collected, dim=0) if collected else None  # type: ignore[arg-type]
        )

        sae_outputs: dict[str, torch.Tensor] | None = None
        if output_sae and collected_tensor is not None:
            assert input_ids is not None
            self._validate_sae_inputs(input_ids)
            sae_outputs = self._get_sae_outputs(
                collected_tensor, layers_to_collect, bool_mask, normalize_sae
            )

        hidden_states_tensor = collected_tensor if output_hidden_states else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    hidden_states_tensor,
                    sae_outputs,
                    attentions,
                ]
                if v is not None
            )

        return ESMCOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states_tensor,
            sae_outputs=sae_outputs,
            attentions=attentions,
        )


# ---------------------------------------------------------------------------
# LM head
# ---------------------------------------------------------------------------


def _esmc_lm_head(
    d_model: int, output_dim: int, hidden_dim: int | None = None
) -> nn.Sequential:
    """Linear → GELU → LayerNorm → Linear projection head for masked LM."""
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


# ---------------------------------------------------------------------------
# Masked language model
# ---------------------------------------------------------------------------


@auto_docstring
class ESMCForMaskedLM(ESMCPreTrainedModel):
    """ESMC with a masked language modelling head.

    This is the primary pre-training objective of ESMC.  The LM head consists
    of a single hidden layer with GELU activation followed by LayerNorm and a
    linear projection to ``vocab_size``.
    """

    def __init__(self, config: ESMCConfig):
        super().__init__(config)
        self.esmc = ESMCModel(config)
        self.lm_head = _esmc_lm_head(config.d_model, config.vocab_size)
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head[-1]  # type: ignore[return-value]

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head[-1] = new_embeddings

    def add_sae_models(self, sae_models: list[_ESMCSAELayer]) -> None:
        """Proxy to :meth:`ESMCModel.add_sae_models`."""
        self.esmc.add_sae_models(sae_models)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_id: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        compute_sae: bool = True,
        normalize_sae: bool = False,
    ) -> tuple[torch.Tensor, ...] | ESMCMaskedLMOutput:
        r"""
        sequence_id (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Integer chain-ID tensor forwarded to the encoder for chain-aware
            attention masking. See :meth:`ESMCModel.forward` for the encoding.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for masked language modelling loss.  Positions with label ``-100``
            are ignored.  Other positions must be in ``[0, config.vocab_size)``.
        output_attentions (`bool`, *optional*):
            Whether to return per-block attention weights. Forwarded to the
            backbone; raises on the ``flash_attention_2`` path.
        compute_sae (`bool`, *optional*, defaults to ``True``):
            Whether to run registered SAE models. Has no effect when none are registered.
        normalize_sae (`bool`, *optional*, defaults to ``False``):
            When ``True``, scale SAE features by ``idf / max`` normalization buffers.

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ESMCForMaskedLM
        >>> import torch

        >>> model = ESMCForMaskedLM.from_pretrained("Biohub/ESMC-600M-2024-12")
        >>> tokenizer = AutoTokenizer.from_pretrained("Biohub/ESMC-600M-2024-12")
        >>> inputs = tokenizer(["MLKNVQ<mask>LV"], return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> outputs.logits.shape
        torch.Size([1, 11, 64])
        ```
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.esmc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            compute_sae=compute_sae,
            normalize_sae=normalize_sae,
        )

        logits = self.lm_head(encoder_outputs.last_hidden_state)

        loss: torch.Tensor | None = None
        if labels is not None:
            loss = CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    logits,
                    encoder_outputs.last_hidden_state,
                    encoder_outputs.hidden_states,
                    encoder_outputs.sae_outputs,
                    encoder_outputs.attentions,
                ]
                if v is not None
            )

        return ESMCMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            sae_outputs=encoder_outputs.sae_outputs,
            attentions=encoder_outputs.attentions,
        )


# ---------------------------------------------------------------------------
# Classification heads
# ---------------------------------------------------------------------------


class _ESMCClassificationHead(nn.Module):
    """Dense classification head applied to the ``<cls>`` token representation."""

    def __init__(self, config: ESMCConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states[:, 0, :]  # <cls> token
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Sequence classification
# ---------------------------------------------------------------------------


@auto_docstring
class ESMCForSequenceClassification(ESMCPreTrainedModel):
    """ESMC with a sequence-level classification head.

    A linear layer is applied to the ``<cls>`` token representation.
    Supports regression (``num_labels == 1``), single-label classification,
    and multi-label classification.
    """

    def __init__(self, config: ESMCConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.esmc = ESMCModel(config)
        self.classifier = _ESMCClassificationHead(config)
        self.post_init()

    def add_sae_models(self, sae_models: list[_ESMCSAELayer]) -> None:
        """Proxy to :meth:`ESMCModel.add_sae_models`."""
        self.esmc.add_sae_models(sae_models)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        compute_sae: bool = True,
        normalize_sae: bool = False,
    ) -> tuple[torch.Tensor, ...] | ESMCSequenceClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for sequence classification loss.  Indices must be in
            ``[0, config.num_labels - 1]``.  For regression pass a float
            tensor of shape ``(batch_size,)``.
        output_attentions (`bool`, *optional*):
            Whether to return per-block attention weights. Forwarded to the
            backbone; raises on the ``flash_attention_2`` path.
        compute_sae (`bool`, *optional*, defaults to ``True``):
            Whether to run registered SAE models. Has no effect when none are registered.
        normalize_sae (`bool`, *optional*, defaults to ``False``):
            When ``True``, scale SAE features by ``idf / max`` normalization buffers.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.esmc(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            compute_sae=compute_sae,
            normalize_sae=normalize_sae,
        )
        logits = self.classifier(encoder_outputs.last_hidden_state)

        loss: torch.Tensor | None = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (torch.long, torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(
                    logits.squeeze() if self.num_labels == 1 else logits,
                    labels.squeeze() if self.num_labels == 1 else labels,
                )
            elif self.config.problem_type == "single_label_classification":
                loss = CrossEntropyLoss()(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss = BCEWithLogitsLoss()(logits, labels)

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    logits,
                    encoder_outputs.last_hidden_state,
                    encoder_outputs.hidden_states,
                    encoder_outputs.sae_outputs,
                    encoder_outputs.attentions,
                ]
                if v is not None
            )

        return ESMCSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            sae_outputs=encoder_outputs.sae_outputs,
            attentions=encoder_outputs.attentions,
        )


# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------


@auto_docstring
class ESMCForTokenClassification(ESMCPreTrainedModel):
    """ESMC with a per-token classification head.

    Useful for tasks such as secondary structure prediction, contact-map
    prediction, or per-residue labelling.
    """

    def __init__(self, config: ESMCConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.esmc = ESMCModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.d_model, config.num_labels)
        self.post_init()

    def add_sae_models(self, sae_models: list[_ESMCSAELayer]) -> None:
        """Proxy to :meth:`ESMCModel.add_sae_models`."""
        self.esmc.add_sae_models(sae_models)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        compute_sae: bool = True,
        normalize_sae: bool = False,
    ) -> tuple[torch.Tensor, ...] | ESMCTokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token labels.  Indices must be in ``[0, config.num_labels - 1]``.
            Positions with index ``-100`` are ignored in the loss.
        output_attentions (`bool`, *optional*):
            Whether to return per-block attention weights. Forwarded to the
            backbone; raises on the ``flash_attention_2`` path.
        compute_sae (`bool`, *optional*, defaults to ``True``):
            Whether to run registered SAE models. Has no effect when none are registered.
        normalize_sae (`bool`, *optional*, defaults to ``False``):
            When ``True``, scale SAE features by ``idf / max`` normalization buffers.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.esmc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
            compute_sae=compute_sae,
            normalize_sae=normalize_sae,
        )

        sequence_output = self.dropout(encoder_outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss: torch.Tensor | None = None
        if labels is not None:
            loss = CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, self.num_labels), labels.to(logits.device).view(-1)
            )

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    logits,
                    encoder_outputs.last_hidden_state,
                    encoder_outputs.hidden_states,
                    encoder_outputs.sae_outputs,
                    encoder_outputs.attentions,
                ]
                if v is not None
            )

        return ESMCTokenClassifierOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            sae_outputs=encoder_outputs.sae_outputs,
            attentions=encoder_outputs.attentions,
        )


__all__ = [
    "ESMCModel",
    "ESMCForMaskedLM",
    "ESMCForSequenceClassification",
    "ESMCForTokenClassification",
    "ESMCPreTrainedModel",
]
