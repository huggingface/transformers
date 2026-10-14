# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import copy

import torch
from torch import nn

from ...activations import ACT2FN
from ...configuration_utils import strict
from ...masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask
from ...modeling_outputs import BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..eurobert.configuration_eurobert import EuroBertConfig
from ..eurobert.modeling_eurobert import (
    EuroBertAttention,
    EuroBertDecoderLayer,
    EuroBertMLP,
    EuroBertModel,
    EuroBertPreTrainedModel,
)
from ..llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding


@auto_docstring(checkpoint="dmis-lab/UltraBERT-150M")
@strict
class UltrabertConfig(EuroBertConfig):
    r"""
    attention_windows (`int` or `list[int | None]`, *optional*):
        Inclusive attention radius on either side of each token. `None` or zero uses full attention.
        A scalar applies to every layer.
    rope_theta (`float` or `list[float]`, *optional*, defaults to 10000.0):
        Rotary embedding base for each layer. A scalar applies to every layer.
    prediction_head_size (`int`, *optional*):
        Intermediate dimension of the two-layer masked language modeling prediction head.
        If unset, the prediction head is an identity.
    """

    model_type = "ultrabert"

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    pad_token_id: int | None = 0
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 0
    use_cache: bool = False
    max_position_embeddings: int = 32768
    attention_windows: int | list[int | None] | None = None
    rope_theta: float | list[float] = 10000.0
    prediction_head_size: int | None = None
    classifier_dropout: float = 0.0
    rope_parameters = AttributeError()
    pretraining_tp = AttributeError()
    classifier_pooling = AttributeError()
    mask_token_id = AttributeError()

    def __post_init__(self, **kwargs):
        if not isinstance(self.attention_windows, (list, tuple)):
            self.attention_windows = [self.attention_windows] * self.num_hidden_layers
        if not isinstance(self.rope_theta, (list, tuple)):
            self.rope_theta = [self.rope_theta] * self.num_hidden_layers
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        if len(self.attention_windows) != self.num_hidden_layers or len(self.rope_theta) != self.num_hidden_layers:
            raise ValueError("attention_windows and rope_theta must have one entry per hidden layer")
        if any(window is not None and window < 0 for window in self.attention_windows):
            raise ValueError("attention_windows must contain nonnegative radii or None")
        if any(base <= 0 for base in self.rope_theta):
            raise ValueError("rope_theta must contain positive bases")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")


class UltrabertRMSNorm(LlamaRMSNorm):
    pass


class UltrabertRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: UltrabertConfig, base: float):
        config = copy(config)
        config.rope_parameters = {"rope_type": "default", "rope_theta": base}
        super().__init__(config)


class UltrabertAttention(EuroBertAttention):
    pass


class UltrabertMLP(EuroBertMLP):
    pass


class UltrabertDecoderLayer(EuroBertDecoderLayer):
    pass


class UltrabertPreTrainedModel(EuroBertPreTrainedModel):
    base_model_prefix = "ultrabert"


class UltrabertModel(EuroBertModel):
    def __init__(self, config: UltrabertConfig):
        super().__init__(config)
        self.rotary_emb = nn.ModuleDict(
            {str(base).replace(".", "_"): UltrabertRotaryEmbedding(config, base) for base in set(config.rope_theta)}
        )
        self._mask_configs = {}
        for window in set(config.attention_windows):
            mask_config = copy(config)
            mask_config.sliding_window = window
            self._mask_configs[window] = mask_config

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        attention_masks = {}
        for window, mask_config in self._mask_configs.items():
            mask_config._attn_implementation = self.config._attn_implementation
            mask_function = create_bidirectional_sliding_window_mask if window else create_bidirectional_mask
            attention_masks[window] = mask_function(
                config=mask_config, inputs_embeds=inputs_embeds, attention_mask=attention_mask
            )
            if (
                self.config._attn_implementation == "sdpa"
                and attention_masks[window] is not None
                and attention_masks[window].dtype == torch.bool
            ):
                attention_masks[window] = torch.zeros_like(
                    attention_masks[window], dtype=inputs_embeds.dtype
                ).masked_fill_(~attention_masks[window], torch.finfo(inputs_embeds.dtype).min)

        position_embeddings = {
            base: self.rotary_emb[str(base).replace(".", "_")](inputs_embeds, position_ids)
            for base in set(self.config.rope_theta)
        }
        hidden_states = inputs_embeds
        for i, encoder_layer in enumerate(self.layers):
            window = self.config.attention_windows[i]
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_masks[window],
                position_ids=position_ids,
                position_embeddings=position_embeddings[self.config.rope_theta[i]],
                sliding_window=window + 1 if window else None,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class UltrabertPredictionHead(nn.Module):
    def __init__(self, config: UltrabertConfig):
        super().__init__()
        if config.prediction_head_size is not None:
            self.linear1 = nn.Linear(config.hidden_size, config.prediction_head_size, bias=False)
            self.linear2 = nn.Linear(config.prediction_head_size, config.hidden_size, bias=False)
            self.act_fn = ACT2FN["gelu_pytorch_tanh"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "linear1"):
            hidden_states = self.act_fn(self.linear2(self.act_fn(self.linear1(hidden_states))))
        return hidden_states


@auto_docstring
class UltrabertForMaskedLM(UltrabertPreTrainedModel):
    _tied_weights_keys = {"cls.weight": "ultrabert.embed_tokens.weight"}

    def __init__(self, config: UltrabertConfig):
        super().__init__(config)
        self.ultrabert = UltrabertModel(config)
        self.head = UltrabertPredictionHead(config)
        self.cls = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.cls

    def set_output_embeddings(self, new_embeddings):
        self.cls = new_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MaskedLMOutput:
        outputs = self.ultrabert(
            input_ids, attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=inputs_embeds, **kwargs
        )
        logits = self.cls(self.head(outputs.last_hidden_state))
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return MaskedLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


@auto_docstring
class UltrabertForSequenceClassification(UltrabertPreTrainedModel):
    def __init__(self, config: UltrabertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.ultrabert = UltrabertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutput:
        outputs = self.ultrabert(
            input_ids, attention_mask=attention_mask, position_ids=position_ids, inputs_embeds=inputs_embeds, **kwargs
        )
        logits = self.classifier(self.dropout(outputs.last_hidden_state[:, 0]))
        loss = None
        if labels is not None:
            loss = self.loss_function(labels=labels, pooled_logits=logits, config=self.config, **kwargs)
        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


__all__ = [
    "UltrabertConfig",
    "UltrabertPreTrainedModel",
    "UltrabertModel",
    "UltrabertForMaskedLM",
    "UltrabertForSequenceClassification",
]
