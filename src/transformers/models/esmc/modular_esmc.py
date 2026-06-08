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
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from ... import initialization as init
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
from ..esm.modeling_esm import (
    eager_attention_forward,
    rotate_half,
)
from .configuration_esmc import ESMCConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ESMCConfig"

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
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Per-layer attention weights of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``.
            Returned when ``output_attentions=True``.  Not available on the
            ``flash_attention_2`` path.
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None
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
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Per-layer attention weights of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``.
            Returned when ``output_attentions=True``.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None
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
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Per-layer attention weights of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``.
            Returned when ``output_attentions=True``.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None
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
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Per-layer attention weights of shape
            ``(batch_size, num_heads, sequence_length, sequence_length)``.
            Returned when ``output_attentions=True``.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


# ---------------------------------------------------------------------------
# Rotary position embedding helpers
# ---------------------------------------------------------------------------


class ESMCRotaryEmbedding(nn.Module):
    """Rotary position embeddings (RoPE), config-driven, returning ``(cos, sin)``.

    Follows the standard Transformers rotary convention (cf.
    ``EsmRotaryEmbedding`` / ``LlamaRotaryEmbedding``): ``inv_freq`` is a
    non-persistent fp32 buffer and ``forward`` builds full-head-dim ``cos`` /
    ``sin`` in fp32 before casting to the input dtype.
    """

    inv_freq: torch.Tensor

    def __init__(self, config: ESMCConfig, device=None):
        super().__init__()
        self.config = config
        self.register_buffer("inv_freq", self._compute_inv_freq(config, device), persistent=False)

    @staticmethod
    def _compute_inv_freq(config: ESMCConfig, device=None) -> torch.Tensor:
        dim = config.d_model // config.n_heads
        return 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim)
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # force fp32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply Rotary Position Embedding to ``q`` and ``k`` in the activation dtype.

    This deliberately differs from the otherwise-identical
    :func:`~transformers.models.esm.modeling_esm.apply_rotary_pos_emb` (whose
    ``rotate_half`` it reuses): that helper upcasts ``q``/``k`` to fp32 for the
    rotation, but the reference ESMC implementation applies RoPE in the
    activation dtype. Upcasting here would make bf16 inference diverge from the
    published ESMC numerics — at bf16 it is the single source of fork-vs-port
    drift, accumulating over the residual stream (~0.3 over 80 layers on
    ESMC-6B). The rotation is a no-op-difference in fp32 (``q`` is already fp32),
    so fp32 stays bit-exact. See ``modeling_esm`` for the argument semantics.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Feed-forward network helpers
# ---------------------------------------------------------------------------


def _swiglu_hidden_dim(expansion_ratio: float, d_model: int) -> int:
    """Round hidden dim to the nearest multiple of 256 after applying expansion_ratio."""
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class _PyTorchLayerNormLinear(nn.Module):
    """LayerNorm followed by a Linear projection.

    Parameters are named ``layer_norm_weight``, ``layer_norm_bias`` and
    ``weight`` to match the published ESMC checkpoint layout.
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
        x = F.layer_norm(x, (self.d_in,), self.layer_norm_weight, self.layer_norm_bias, self.eps)
        return F.linear(x, self.weight)


class _PyTorchLayerNormMLP(nn.Module):
    """LayerNorm + SwiGLU MLP.

    Parameters are named ``layer_norm_weight``, ``layer_norm_bias``,
    ``fc1_weight`` and ``fc2_weight`` to match the published ESMC checkpoint
    layout.
    """

    def __init__(self, hidden_size: int, ffn_hidden_size: int, eps: float = 1e-5) -> None:
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
    """LayerNorm + SwiGLU MLP."""
    assert not bias, "ESMC was trained with bias=False; bias=True not supported"
    hidden = _swiglu_hidden_dim(expansion_ratio, d_model)
    return _PyTorchLayerNormMLP(hidden_size=d_model, ffn_hidden_size=hidden)


def _make_attn_layernorm_qkv(d_model: int, bias: bool) -> nn.Module:
    """LayerNorm + fused QKV projection."""
    assert not bias, "ESMC was trained with bias=False; bias=True not supported"
    return _PyTorchLayerNormLinear(d_model, d_model * 3)


def _make_attn_out_proj(d_model: int, bias: bool) -> nn.Module:
    """Attention output projection."""
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
        # ESMC is a bidirectional encoder: never apply causal masking. Without
        # this, the sdpa/flash interfaces default `is_causal` to True when no
        # attention_mask is passed (unpadded inputs), silently masking causally.
        self.is_causal = False

        assert not bias, "ESMC was trained with bias=False; bias=True not supported"
        self.layernorm_qkv = _make_attn_layernorm_qkv(d_model, bias)
        self.out_proj = _make_attn_out_proj(d_model, bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(context, attn_weights)``.

        Attention is computed by the backend selected through
        ``config._attn_implementation`` (``eager`` / ``sdpa`` /
        ``flash_attention_2`` / ...). QK-LayerNorm and RoPE (via
        ``position_embeddings``) are applied here; the backend only computes
        ``softmax(QKᵀ)V``. ``attn_weights`` is ``None`` unless the backend
        exposes the probabilities — ``output_attentions=True`` forces the
        ``eager`` interface so they are observable.
        """
        b, s, _ = hidden_states.shape
        qkv = self.layernorm_qkv(hidden_states)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = self.q_ln(q).to(q.dtype)
        k = self.k_ln(k).to(q.dtype)

        # (B, S, D) -> (B, H, S, Dh)
        q = q.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

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

        self.attn = MultiHeadAttention(config, d_model, n_heads, bias=bias, qk_layernorm=qk_layernorm)

        if ffn_type == "swiglu":
            self.ffn = _swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn = _gelu_ln_ffn(d_model, expansion_ratio, bias)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type!r}. Choose 'swiglu' or 'gelu'.")

        self.scaling_factor = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
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
            x,
            attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
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
                    residue_scaling_factor=math.sqrt(n_layers / 36) if scale_residue else 1.0,
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
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
                x,
                attention_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                **kwargs,
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
        # The fused LN+projection modules are handled explicitly: the base
        # `_init_weights` matches norms by class-name substring ("LayerNorm"),
        # which would otherwise mis-initialize their (Linear) `weight` to ones.
        if isinstance(module, _PyTorchLayerNormLinear):
            init.ones_(module.layer_norm_weight)
            init.zeros_(module.layer_norm_bias)
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, _PyTorchLayerNormMLP):
            init.ones_(module.layer_norm_weight)
            init.zeros_(module.layer_norm_bias)
            init.normal_(module.fc1_weight, mean=0.0, std=std)
            init.normal_(module.fc2_weight, mean=0.0, std=std)
        elif isinstance(module, ESMCRotaryEmbedding):
            init.copy_(module.inv_freq, module._compute_inv_freq(module.config))
        else:
            # nn.Linear / nn.Embedding / nn.LayerNorm via the base initializer.
            super()._init_weights(module)


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
        self.rotary_emb = ESMCRotaryEmbedding(config)
        self.transformer = TransformerStack(
            config,
            config.d_model,
            config.n_heads,
            config.n_layers,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
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
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Collect per-layer hidden states only when the caller asks for them.
        layers_to_collect: list[int] = list(range(self.config.n_layers + 1)) if output_hidden_states else []

        user_supplied_sequence_id = sequence_id is not None
        if not user_supplied_sequence_id and attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id

        x = self.embed(input_ids)
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)

        if user_supplied_sequence_id:
            if self.config._attn_implementation == "flash_attention_2" and (sequence_id > 0).any():
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
            attn_bias = torch.zeros(same_chain.shape, dtype=x.dtype, device=x.device).masked_fill_(
                ~same_chain, torch.finfo(x.dtype).min
            )
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
            position_embeddings=position_embeddings,
            layers_to_collect=layers_to_collect,
            output_attentions=output_attentions,
        )

        collected_tensor: torch.Tensor | None = (
            torch.stack(collected, dim=0) if collected else None  # type: ignore[arg-type]
        )
        hidden_states_tensor = collected_tensor if output_hidden_states else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    hidden_states_tensor,
                    attentions,
                ]
                if v is not None
            )

        return ESMCOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states_tensor,
            attentions=attentions,
        )


# ---------------------------------------------------------------------------
# LM head
# ---------------------------------------------------------------------------


def _esmc_lm_head(d_model: int, output_dim: int, hidden_dim: int | None = None) -> nn.Sequential:
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

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | ESMCMaskedLMOutput:
        r"""
        sequence_id (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Integer chain-ID tensor forwarded to the encoder for chain-aware
            attention masking. See :meth:`ESMCModel.forward` for the encoding.
        output_attentions (`bool`, *optional*):
            Whether to return per-block attention weights. Forwarded to the
            backbone; raises on the ``flash_attention_2`` path.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for masked language modelling loss.  Positions with label ``-100``
            are ignored.  Other positions must be in ``[0, config.vocab_size)``.

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.esmc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        logits = self.lm_head(encoder_outputs.last_hidden_state)

        loss: torch.Tensor | None = None
        if labels is not None:
            loss = CrossEntropyLoss(ignore_index=-100)(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss,
                    logits,
                    encoder_outputs.last_hidden_state,
                    encoder_outputs.hidden_states,
                    encoder_outputs.attentions,
                ]
                if v is not None
            )

        return ESMCMaskedLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
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

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | ESMCSequenceClassifierOutput:
        r"""
        output_attentions (`bool`, *optional*):
            Whether to return per-block attention weights. Forwarded to the
            backbone; raises on the ``flash_attention_2`` path.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for sequence classification loss.  Indices must be in
            ``[0, config.num_labels - 1]``.  For regression pass a float
            tensor of shape ``(batch_size,)``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.esmc(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
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
                loss = CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
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
                    encoder_outputs.attentions,
                ]
                if v is not None
            )

        return ESMCSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
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

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | ESMCTokenClassifierOutput:
        r"""
        output_attentions (`bool`, *optional*):
            Whether to return per-block attention weights. Forwarded to the
            backbone; raises on the ``flash_attention_2`` path.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token labels.  Indices must be in ``[0, config.num_labels - 1]``.
            Positions with index ``-100`` are ignored in the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.esmc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
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
                    encoder_outputs.attentions,
                ]
                if v is not None
            )

        return ESMCTokenClassifierOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


__all__ = [
    "ESMCModel",
    "ESMCForMaskedLM",
    "ESMCForSequenceClassification",
    "ESMCForTokenClassification",
    "ESMCPreTrainedModel",
]
