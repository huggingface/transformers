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

from ...masking_utils import create_bidirectional_mask, packed_sequence_mask_function
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.generic import is_flash_attention_requested
from ..esm.modeling_esm import EsmClassificationHead, eager_attention_forward
from ..llama.modeling_llama import LlamaMLP, LlamaRotaryEmbedding, apply_rotary_pos_emb
from .configuration_esmc import ESMCConfig


logger = logging.get_logger(__name__)


class ESMCRotaryEmbedding(LlamaRotaryEmbedding):
    def _apply(self, fn, recurse=True):
        # Little bit of hackery compared to Llama to match the original's dtypes
        result = super()._apply(fn, recurse=recurse)
        inv_freq, _ = self.compute_default_rope_parameters(self.config, device=self.inv_freq.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        return result


class ESMCMLP(LlamaMLP):
    pass


class ESMCAttention(nn.Module):
    def __init__(self, config: ESMCConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = False

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_ln = nn.LayerNorm(config.hidden_size, bias=False) if config.qk_layernorm else nn.Identity()
        self.k_ln = nn.LayerNorm(config.hidden_size, bias=False) if config.qk_layernorm else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(context, attn_weights)``.

        Attention is computed by the backend selected through
        ``config._attn_implementation``. ``attn_weights`` is ``None`` unless the backend
        exposes the probabilities -- ``output_attentions=True`` forces the ``eager``
        interface so they are observable.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_ln(self.q_proj(hidden_states)).to(hidden_states.dtype)
        key_states = self.k_ln(self.k_proj(hidden_states)).to(hidden_states.dtype)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if not output_attentions:
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
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class ESMCLayer(GradientCheckpointingLayer):
    """Single transformer block: pre-norm attention + pre-norm FFN with residual scaling."""

    def __init__(self, config: ESMCConfig):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.self_attn = ESMCAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.mlp = ESMCMLP(config)
        self.scaling_factor = math.sqrt(config.num_hidden_layers / 36) if config.scale_residue else 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        attn_output, attn_weights = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )
        # ESM3 residue scaling on each residual branch.
        hidden_states = residual + attn_output / self.scaling_factor
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states)) / self.scaling_factor
        return hidden_states, attn_weights


@auto_docstring
class ESMCPreTrainedModel(PreTrainedModel):
    config_class = ESMCConfig
    base_model_prefix = "esmc"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True
    _no_split_modules = ["ESMCLayer"]
    # ``inv_freq`` / ``original_inv_freq`` are non-persistent rotary buffers; ``_extra_state``
    # keys come from the published checkpoint's fused TransformerEngine layout.
    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$", r"\.inv_freq$", r"\.original_inv_freq$"]


@auto_docstring
class ESMCModel(ESMCPreTrainedModel):
    def __init__(self, config: ESMCConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = ESMCRotaryEmbedding(config)
        self.layers = nn.ModuleList([ESMCLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, bias=False)
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, ...] | BaseModelOutput:
        r"""
        sequence_id (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Integer chain-ID tensor for chain-aware attention masking. Tokens with the same
            non-negative integer value can attend to each other; tokens with different values
            cannot (cross-chain masking). Padding positions should be set to ``-1`` and inputs
            must be **right-padded** (RoPE uses absolute positions starting at 0). When provided,
            ``attention_mask`` is ignored. Passing ``sequence_id`` builds a custom attention
            mask, which requires ``torch>=2.6``. Multi-chain inputs additionally require a
            non-flash ``attn_implementation`` (``'sdpa'`` / ``'eager'`` / ``'flex_attention'``);
            flash attention only supports the single-chain case.
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
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        hidden_states = self.embed(input_ids)
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if sequence_id is not None:
            # Chain-aware attention: a token attends only to tokens sharing its
            # ``sequence_id`` (block-diagonal), expressed as an additional ``and`` mask
            # over the bidirectional mask. ``attention_mask`` is ignored here -- padding
            # is encoded as ``sequence_id == -1``. Flash attention can't represent a
            # block-diagonal mask without varlen, so multi-chain inputs are rejected.
            if is_flash_attention_requested(self.config) and (sequence_id > 0).any():
                raise ValueError(
                    "Multi-chain ``sequence_id`` (any value > 0) is not supported with "
                    "flash attention. Re-load the model with attn_implementation='sdpa' "
                    "(or 'eager' / 'flex_attention') for chain-aware attention masking."
                )
            attn_bias = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=None,
                and_mask_function=packed_sequence_mask_function(sequence_id),
            )
        else:
            attn_bias = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )

        all_hidden_states: tuple[torch.Tensor, ...] = () if output_hidden_states else None
        all_attentions: tuple[torch.Tensor, ...] = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, attn_weights = layer(
                hidden_states,
                attn_bias,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                **kwargs,
            )
            if output_attentions and attn_weights is not None:
                all_attentions += (attn_weights,)

        last_hidden_state = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (last_hidden_state,)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class ESMCMaskedLMHead(nn.Module):
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


@auto_docstring
class ESMCForMaskedLM(ESMCPreTrainedModel):
    def __init__(self, config: ESMCConfig):
        super().__init__(config)
        self.esmc = ESMCModel(config)
        self.lm_head = ESMCMaskedLMHead(config.hidden_size, config.vocab_size)
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
        **kwargs: Unpack[TransformersKwargs],
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


class ESMCClassificationHead(EsmClassificationHead):
    def __init__(self, config: ESMCConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)


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
        **kwargs: Unpack[TransformersKwargs],
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


@auto_docstring
class ESMCForTokenClassification(ESMCPreTrainedModel):
    def __init__(self, config: ESMCConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.esmc = ESMCModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
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
        **kwargs: Unpack[TransformersKwargs],
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
