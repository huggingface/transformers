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

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from ... import initialization as init
from ...masking_utils import create_bidirectional_mask  # type: ignore[import]
from ...modeling_outputs import (  # type: ignore[import]
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel  # type: ignore[import]
from ...utils import (  # type: ignore[import]
    auto_docstring,
    can_return_tuple,
    logging,
)
from ..esm.modeling_esm import EsmClassificationHead, EsmRotaryEmbedding, eager_attention_forward

# ESMC applies RoPE in the activation dtype (no fp32 upcast), matching the reference
# implementation's bf16 numerics. Llama's `apply_rotary_pos_emb` is exactly that
# no-upcast variant; `esm`'s upcasts q/k to fp32 and would drift bf16 inference (the
# dominant fork-vs-port divergence on the ESMFold2 backbone, ~0.3 over 80 layers on
# ESMC-6B), so we reuse Llama's rather than esm's. `rotate_half` is pulled in by the
# modular converter as a dependency of the imported function.
from ..llama.modeling_llama import apply_rotary_pos_emb
from ..modernbert.modeling_modernbert import ModernBertMLP
from .configuration_esmc import ESMCConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "ESMCConfig"

# ---------------------------------------------------------------------------
# Rotary position embedding helpers
# ---------------------------------------------------------------------------


class ESMCRotaryEmbedding(EsmRotaryEmbedding):
    """Rotary position embeddings (RoPE), returning ``(cos, sin)``.

    Identical to :class:`~transformers.models.esm.modeling_esm.EsmRotaryEmbedding`
    (``inv_freq = 1 / theta^(arange(0, dim, 2) / dim)`` with ``dim = d_model // n_heads``,
    ``cos`` / ``sin`` built in fp32), with two ESMC-specific tweaks:

    * ``inv_freq`` is a **non-persistent** buffer (recomputed from the config, never
      stored in the checkpoint).
    * ``_apply`` is overridden so ``inv_freq`` stays fp32 even when the module is cast
      to bf16/fp16 (e.g. when ESMFold2 loads its bundled ESMC backbone in bf16).
      ``nn.Module._apply`` would otherwise round the rotary frequencies to bf16, which
      drifts the RoPE angles and is the dominant source of bf16 fork-vs-port divergence
      in the ESMFold2 backbone.
    """

    def __init__(self, config: ESMCConfig, device=None):
        super().__init__(config, device=device)
        inv_freq, _ = self.compute_default_rope_parameters(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        inv_freq, _ = self.compute_default_rope_parameters(self.config, device=self.inv_freq.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        return result


# ---------------------------------------------------------------------------
# Feed-forward network helpers
# ---------------------------------------------------------------------------


def _swiglu_hidden_dim(expansion_ratio: float, d_model: int) -> int:
    """Round hidden dim to the nearest multiple of 256 after applying expansion_ratio."""
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class ESMCMLP(ModernBertMLP):
    """SwiGLU feed-forward network, reusing :class:`ModernBertMLP`'s gated forward.

    ``ModernBertMLP`` computes ``Wo(act(input) * gate)`` with ``input, gate =
    Wi(x).chunk(2, dim=-1)`` — exactly ESMC's SwiGLU once ``act`` is SiLU. ESMC was
    trained with ``bias=False`` and no MLP dropout, and rounds the hidden dim to a
    multiple of 256. The pre-MLP LayerNorm lives in :class:`ESMCLayer` (``mlp_norm``);
    the published checkpoint's ``ffn.fc1_weight`` / ``ffn.fc2_weight`` map onto
    ``mlp.Wi`` / ``mlp.Wo`` via ``conversion_mapping.py``.
    """

    def __init__(self, d_model: int, expansion_ratio: float = 8 / 3) -> None:
        nn.Module.__init__(self)
        ffn_hidden_size = _swiglu_hidden_dim(expansion_ratio, d_model)
        self.Wi = nn.Linear(d_model, 2 * ffn_hidden_size, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Identity()
        self.Wo = nn.Linear(ffn_hidden_size, d_model, bias=False)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class ESMCAttention(nn.Module):
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
        if bias:
            raise ValueError("ESMC was trained with bias=False; bias=True is not supported.")
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

        # Fused QKV projection (``Wqkv``) and output projection (``Wo``), matching
        # ModernBERT's attention layout. The pre-attention LayerNorm lives in
        # :class:`ESMCLayer` (``attn_norm``); the published checkpoint's fused
        # ``attn.layernorm_qkv`` / ``attn.out_proj`` map onto these via
        # ``conversion_mapping.py``.
        self.Wqkv = nn.Linear(d_model, d_model * 3, bias=bias)
        self.Wo = nn.Linear(d_model, d_model, bias=bias)

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
        qkv = self.Wqkv(hidden_states)
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
        return self.Wo(attn_output), attn_weights


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class ESMCLayer(nn.Module):
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
    """

    def __init__(
        self,
        config: ESMCConfig,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        expansion_ratio: float = 8 / 3,
        residue_scaling_factor: float = 1.0,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        if bias:
            raise ValueError("ESMC was trained with bias=False; bias=True is not supported.")

        # Pre-norm layout (cf. ModernBERT): the LayerNorms that the published ESMC
        # checkpoint fuses into ``attn.layernorm_qkv`` / ``ffn`` live here as
        # ``attn_norm`` / ``mlp_norm`` (remapped in ``conversion_mapping.py``).
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = ESMCAttention(config, d_model, n_heads, bias=bias, qk_layernorm=qk_layernorm)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = ESMCMLP(d_model, expansion_ratio)

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
            self.attn_norm(x),
            attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )
        x = x + attn_out / self.scaling_factor
        x = x + self.mlp(self.mlp_norm(x)) / self.scaling_factor
        return x, attn_weights


class ESMCEncoder(nn.Module):
    """Stack of :class:`ESMCLayer` layers with a final LayerNorm.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        scale_residue: When ``True`` apply ESM3 residue scaling
            ``sqrt(n_layers / 36)`` to each block.
        bias: Bias flag forwarded to every sub-module.
        qk_layernorm: QK LayerNorm flag forwarded to every block.
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
        expansion_ratio: float = 8 / 3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ESMCLayer(
                    config,
                    d_model,
                    n_heads,
                    residue_scaling_factor=math.sqrt(n_layers / 36) if scale_residue else 1.0,
                    expansion_ratio=expansion_ratio,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
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
    config_class = ESMCConfig
    base_model_prefix = "esmc"
    supports_gradient_checkpointing = False
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True
    _no_split_modules = ["ESMCLayer"]
    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$"]

    def _init_weights(self, module: nn.Module):
        if isinstance(module, ESMCRotaryEmbedding):
            inv_freq, _ = module.compute_default_rope_parameters(module.config)
            init.copy_(module.inv_freq, inv_freq)
        else:
            # nn.Linear / nn.Embedding / nn.LayerNorm via the base initializer.
            super()._init_weights(module)


# ---------------------------------------------------------------------------
# Base encoder model
# ---------------------------------------------------------------------------


@auto_docstring
class ESMCModel(ESMCPreTrainedModel):
    def __init__(self, config: ESMCConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary_emb = ESMCRotaryEmbedding(config)
        self.transformer = ESMCEncoder(
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
        **kwargs,
    ) -> tuple[torch.Tensor, ...] | BaseModelOutput:
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
            Forces a manual-SDPA path inside :class:`ESMCAttention` so the
            attention probabilities are observable; raises on the
            ``flash_attention_2`` path.

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ESMCModel

        >>> model = ESMCModel.from_pretrained("biohub/ESMC-300M")
        >>> tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-300M")
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

        # Standard Transformers convention: a tuple of per-layer hidden states (the
        # live activations collected in the encoder), not a stacked tensor. Consumers
        # that want a single tensor (e.g. ESMFold2's LM projection) stack it themselves.
        hidden_states = collected if output_hidden_states else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    hidden_states,
                    attentions,
                ]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )


# ---------------------------------------------------------------------------
# LM head
# ---------------------------------------------------------------------------


class ESMCMaskedLMHead(nn.Module):
    """Masked-language-modelling head: Linear → GELU → LayerNorm → Linear.

    The published checkpoints store this head as an ``nn.Sequential`` (keys
    ``lm_head.{0,2,3}``); the ``esmc`` entry in ``conversion_mapping.py`` remaps
    them onto ``dense`` / ``layer_norm`` / ``decoder`` on load (and back on save).
    """

    def __init__(self, d_model: int, output_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else d_model
        self.dense = nn.Linear(d_model, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return self.decoder(hidden_states)


# ---------------------------------------------------------------------------
# Masked language model
# ---------------------------------------------------------------------------


@auto_docstring
class ESMCForMaskedLM(ESMCPreTrainedModel):
    def __init__(self, config: ESMCConfig):
        super().__init__(config)
        self.esmc = ESMCModel(config)
        self.lm_head = ESMCMaskedLMHead(config.d_model, config.vocab_size)
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head.decoder = new_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...] | MaskedLMOutput:
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

        >>> model = ESMCForMaskedLM.from_pretrained("biohub/ESMC-300M")
        >>> tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-300M")
        >>> inputs = tokenizer(["MLKNVQ<mask>LV"], return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> outputs.logits.shape
        torch.Size([1, 11, 64])
        ```
        """
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

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# ---------------------------------------------------------------------------
# Classification heads
# ---------------------------------------------------------------------------


class ESMCClassificationHead(EsmClassificationHead):
    """Dense classification head applied to the ``<cls>`` token representation.

    Identical to :class:`~transformers.models.esm.modeling_esm.EsmClassificationHead`
    (``<cls>`` token -> dropout -> ``tanh(dense)`` -> dropout -> ``out_proj``); ESMC just
    sources the dropout rate from ``classifier_dropout`` instead of ``hidden_dropout_prob``.
    """

    def __init__(self, config: ESMCConfig):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)


# ---------------------------------------------------------------------------
# Sequence classification
# ---------------------------------------------------------------------------


@auto_docstring
class ESMCForSequenceClassification(ESMCPreTrainedModel):
    def __init__(self, config: ESMCConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.esmc = ESMCModel(config)
        self.classifier = ESMCClassificationHead(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...] | SequenceClassifierOutput:
        r"""
        output_attentions (`bool`, *optional*):
            Whether to return per-block attention weights. Forwarded to the
            backbone; raises on the ``flash_attention_2`` path.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for sequence classification loss.  Indices must be in
            ``[0, config.num_labels - 1]``.  For regression pass a float
            tensor of shape ``(batch_size,)``.
        """
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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# ---------------------------------------------------------------------------
# Token classification
# ---------------------------------------------------------------------------


@auto_docstring
class ESMCForTokenClassification(ESMCPreTrainedModel):
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
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...] | TokenClassifierOutput:
        r"""
        output_attentions (`bool`, *optional*):
            Whether to return per-block attention weights. Forwarded to the
            backbone; raises on the ``flash_attention_2`` path.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Per-token labels.  Indices must be in ``[0, config.num_labels - 1]``.
            Positions with index ``-100`` are ignored in the loss.
        """
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

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
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
