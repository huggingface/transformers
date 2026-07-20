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
"""PyTorch ESMC model."""

import math
from collections.abc import Callable

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from ...masking_utils import create_bidirectional_mask, packed_sequence_mask_function
from ...modeling_layers import GenericForTokenClassification
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
)
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.generic import is_flash_attention_requested, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..esm.modeling_esm import EsmClassificationHead, EsmForSequenceClassification, eager_attention_forward
from ..llama.configuration_llama import LlamaConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from ..nomic_bert.modeling_nomic_bert import NomicBertPreTrainedModel


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="biohub/ESMC-6B")
@strict
class EsmcConfig(LlamaConfig):
    r"""
    mask_token_id (`int`, *optional*, defaults to 32):
        Index of the mask token in the vocabulary (``"<mask>"``), used for masked language modelling.
    classifier_dropout (`float`, *optional*, defaults to 0.1):
        Dropout ratio for the classification head.
    expansion_ratio (`float`, *optional*, defaults to `8/3`):
        Hidden-dim expansion ratio for the SwiGLU feed-forward network. When
        `intermediate_size` is not given it is derived from this as
        `expansion_ratio * hidden_size` rounded up to a multiple of 256.
    scale_residue (`bool`, *optional*, defaults to `True`):
        Whether to apply ESM3 residual scaling (`1 / sqrt(num_hidden_layers / 36)`
        per block) to stabilise deep networks.

    Examples:

    ```python
    >>> from transformers import EsmcConfig, EsmcModel

    >>> # Initializing an ESMC biohub/ESMC-6B style configuration
    >>> configuration = EsmcConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = EsmcModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "esmc"
    attribute_map = {
        "d_model": "hidden_size",
        "n_heads": "num_attention_heads",
        "n_layers": "num_hidden_layers",
    }

    # Llama fields re-declared with ESMC defaults.
    vocab_size: int = 64
    hidden_size: int = 2560
    intermediate_size: int | None = None
    num_hidden_layers: int = 80
    num_attention_heads: int = 40
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    pad_token_id: int | None = 1
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    mlp_bias: bool = False

    # ESMC-specific fields.
    mask_token_id: int | None = 32
    classifier_dropout: float | None = 0.1
    expansion_ratio: float | None = 8 / 3
    scale_residue: bool | None = True

    # Llama fields that do not apply to ESMC
    rms_norm_eps = AttributeError()
    pretraining_tp = AttributeError()
    use_cache = AttributeError()
    keys_to_ignore_at_inference = AttributeError()

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        # ``attribute_map`` (``d_model``/``n_heads``) is only applied inside the base ``__post_init__``,
        # so the GQA-free key/value head count and head dim must be (re-)derived afterwards -- ESMC never
        # uses grouped-query attention, so ``num_key_value_heads`` always tracks ``num_attention_heads``.
        self.num_key_value_heads = self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.intermediate_size is None:
            self.intermediate_size = int(((self.expansion_ratio * self.hidden_size) + 255) // 256 * 256)


class EsmcRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class EsmcMLP(LlamaMLP):
    pass


class EsmcAttention(LlamaAttention):
    """Multi-head self-attention with QK-LayerNorm and RoPE."""

    def __init__(self, config: EsmcConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.is_causal = False
        # QK-LayerNorm is inherent to ESMC; every released checkpoint carries q_norm/k_norm weights.
        self.q_norm = nn.LayerNorm(config.hidden_size, bias=False)
        self.k_norm = nn.LayerNorm(config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states)).to(hidden_states.dtype)
        key_states = self.k_norm(self.k_proj(hidden_states)).to(hidden_states.dtype)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

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
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class EsmcLayer(LlamaDecoderLayer):
    """Single transformer block: pre-norm attention + pre-norm FFN with residual scaling."""

    def __init__(self, config: EsmcConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        # LayerNorm instead of Llama's RMSNorm.
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        # ESM3 residual scaling to stabilise deep networks.
        self.scaling_factor = math.sqrt(config.num_hidden_layers / 36) if config.scale_residue else 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        attn_output, _ = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # ESM3 residue scaling on each residual branch.
        hidden_states = residual + attn_output / self.scaling_factor
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states)) / self.scaling_factor
        return hidden_states


@auto_docstring
class EsmcPreTrainedModel(NomicBertPreTrainedModel):
    config_class = EsmcConfig
    base_model_prefix = "esmc"
    _no_split_modules = ["EsmcLayer"]
    _can_record_outputs = {
        "hidden_states": EsmcLayer,
        "attentions": EsmcAttention,
    }
    # ``inv_freq`` / ``original_inv_freq`` are non-persistent rotary buffers; ``_extra_state``
    # keys come from the published checkpoint's fused TransformerEngine layout.
    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$", r"\.inv_freq$", r"\.original_inv_freq$"]

    def _init_weights(self, module):
        raise AttributeError()


@auto_docstring
class EsmcModel(EsmcPreTrainedModel):
    def __init__(self, config: EsmcConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = EsmcRotaryEmbedding(config)
        self.layers = nn.ModuleList([EsmcLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
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

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, EsmcModel

        >>> model = EsmcModel.from_pretrained("biohub/ESMC-300M")
        >>> tokenizer = AutoTokenizer.from_pretrained("biohub/ESMC-300M")
        >>> inputs = tokenizer(["MLKNVQVQLV"], return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> outputs.last_hidden_state.shape
        torch.Size([1, 12, 960])
        ```
        """
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if sequence_id is not None:
            if is_flash_attention_requested(self.config):
                raise ValueError(
                    "`sequence_id` (chain-aware attention) is not supported with flash attention, "
                    "which can't represent the block-diagonal chain mask. Re-load the model with "
                    "attn_implementation='sdpa' (or 'eager' / 'flex_attention')."
                )
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=None,
                and_mask_function=packed_sequence_mask_function(sequence_id),
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        output = self.norm(hidden_states)

        return BaseModelOutput(last_hidden_state=output)


class EsmcMaskedLMHead(nn.Module):
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
class EsmcForMaskedLM(EsmcPreTrainedModel):
    def __init__(self, config: EsmcConfig):
        super().__init__(config)
        self.esmc = EsmcModel(config)
        self.lm_head = EsmcMaskedLMHead(config.hidden_size, config.vocab_size)
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
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, ...] | MaskedLMOutput:
        r"""
        sequence_id (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Integer chain-ID tensor forwarded to the encoder for chain-aware
            attention masking. See :meth:`EsmcModel.forward` for the encoding.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for masked language modelling loss.  Positions with label ``-100``
            are ignored.  Other positions must be in ``[0, config.vocab_size)``.

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, EsmcForMaskedLM
        >>> import torch

        >>> model = EsmcForMaskedLM.from_pretrained("biohub/ESMC-300M")
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
            return_dict=True,
            **kwargs,
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


class EsmcClassificationHead(EsmClassificationHead):
    def __init__(self, config: EsmcConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)


class EsmcForSequenceClassification(EsmForSequenceClassification):
    # Same `<cls>`-token classification head + problem-type loss as ESM-2; only the
    # backbone (no `add_pooling_layer`) and the `classifier_dropout`-sourced head differ.
    def __init__(self, config: EsmcConfig):
        EsmcPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.esmc = EsmcModel(config)
        self.classifier = EsmcClassificationHead(config)
        self.post_init()


class EsmcForTokenClassification(GenericForTokenClassification, EsmcPreTrainedModel):
    # ``dropout`` (from ``config.classifier_dropout``) + a ``score`` linear over the
    # per-token hidden states, identical to the ESM token-classification head.
    pass


__all__ = [
    "EsmcConfig",
    "EsmcModel",
    "EsmcForMaskedLM",
    "EsmcForSequenceClassification",
    "EsmcForTokenClassification",
    "EsmcPreTrainedModel",
]
