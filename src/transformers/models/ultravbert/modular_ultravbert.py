# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict
from torch.nn import CrossEntropyLoss

from ... import initialization as init
from ...configuration_utils import SubConfigSpec
from ...modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple
from ..auto import AutoConfig
from ..modernvbert.configuration_modernvbert import ModernVBertConfig
from ..modernvbert.modeling_modernvbert import (
    ModernVBertBaseModelOutput,
    ModernVBertConnector,
    ModernVBertForMaskedLM,
    ModernVBertForSequenceClassification,
    ModernVBertForTokenClassification,
    ModernVBertMaskedLMOutput,
    ModernVBertModel,
    ModernVBertPreTrainedModel,
)
from ..ultrabert.modeling_ultrabert import UltrabertPredictionHead, UltrabertRMSNorm


@auto_docstring(checkpoint="athrael-soju/UltraVBERT")
@strict
class UltraVBertConfig(ModernVBertConfig):
    r"""
    pixel_shuffle_factor (`int`, *optional*, defaults to 4):
        Side of the square of vision patches that the connector merges into one image token.
    initializer_cutoff_factor (`float`, *optional*, defaults to 2.0):
        The cutoff factor for the truncated_normal_initializer for initializing all weight matrices.
    classifier_pooling (`Literal["cls", "mean"]`, *optional*, defaults to `"cls"`):
        The pooling strategy to use for classification tasks.
    connector_norm (`bool`, *optional*, defaults to `True`):
        Whether the connector RMS-normalizes the image token embeddings.
    """

    model_type = "ultravbert"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="ultrabert"),
        "vision_config": SubConfigSpec(config_class=AutoConfig, model_type="siglip_vision_model"),
    }

    image_token_id: int = 49191
    connector_norm: bool = True
    classifier_bias = AttributeError()


class UltraVBertBaseModelOutput(ModernVBertBaseModelOutput):
    pass


class UltraVBertMaskedLMOutput(ModernVBertMaskedLMOutput):
    pass


class UltraVBertRMSNorm(UltrabertRMSNorm):
    pass


class UltraVBertConnector(ModernVBertConnector):
    def __init__(self, config):
        super().__init__(config)
        # Raw projections run about two orders of magnitude above UltraBERT's token embeddings, which reach the first
        # layer unnormalized; without the norm, image tokens swamp the residual stream.
        self.norm = (
            UltraVBertRMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
            if config.connector_norm
            else nn.Identity()
        )

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.pixel_shuffle_factor)
        return self.norm(self.modality_projection(image_hidden_states))


@auto_docstring
class UltraVBertPreTrainedModel(ModernVBertPreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)

        def init_weight(module: nn.Module, std: float):
            cutoff_factor = getattr(self.config, "initializer_cutoff_factor", 2.0)
            init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if module.bias is not None:
                    init.zeros_(module.bias)

        if isinstance(module, UltraVBertConnector):
            out_std = self.config.initializer_range / math.sqrt(2.0 * self.config.text_config.num_hidden_layers)
            init_weight(module.modality_projection, out_std)
        elif isinstance(module, UltraVBertForMaskedLM):
            out_std = self.config.initializer_range / math.sqrt(2.0 * self.config.text_config.num_hidden_layers)
            init_weight(module.head.decoder, out_std)
        elif isinstance(module, (UltraVBertForSequenceClassification, UltraVBertForTokenClassification)):
            final_out_std = self.config.initializer_range / math.sqrt(self.config.text_config.hidden_size)
            init_weight(module.classifier, final_out_std)


@auto_docstring
class UltraVBertModel(ModernVBertModel):
    @can_return_tuple
    @auto_docstring(checkpoint="athrael-soju/UltraVBERT")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | UltraVBertBaseModelOutput:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*):
            Mask over the real pixels of each image tile.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )


class UltraVBertPredictionHead(UltrabertPredictionHead):
    pass


@auto_docstring
class UltraVBertForMaskedLM(ModernVBertForMaskedLM):
    _tied_weights_keys = {"head.decoder.weight": "model.text_model.embed_tokens.weight"}

    def __init__(self, config: UltraVBertConfig):
        super().__init__(config)
        self.head = UltraVBertPredictionHead(config.text_config)
        del self.projection_head
        del self.lm_head

    def get_output_embeddings(self):
        return self.head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.head.decoder = new_embeddings

    @can_return_tuple
    @auto_docstring(checkpoint="athrael-soju/UltraVBERT")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | UltraVBertMaskedLMOutput:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*):
            Mask over the real pixels of each image tile.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            **kwargs,
        )
        logits = self.head(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = CrossEntropyLoss()(logits.view(-1, self.vocab_size), labels.view(-1))

        return UltraVBertMaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


class UltraVBertForSequenceClassification(ModernVBertForSequenceClassification):
    def __init__(self, config: UltraVBertConfig):
        super().__init__(config)
        self.head = nn.Identity()

    @can_return_tuple
    @auto_docstring(checkpoint="athrael-soju/UltraVBERT")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | SequenceClassifierOutput:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*):
            Mask over the real pixels of each image tile.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            labels=labels,
            **kwargs,
        )


class UltraVBertForTokenClassification(ModernVBertForTokenClassification):
    def __init__(self, config: UltraVBertConfig):
        super().__init__(config)
        self.head = nn.Identity()

    @can_return_tuple
    @auto_docstring(checkpoint="athrael-soju/UltraVBERT")
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_attention_mask: torch.BoolTensor | None = None,
        image_hidden_states: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | TokenClassifierOutput:
        r"""
        pixel_attention_mask (`torch.BoolTensor` of shape `(batch_size, num_images, image_size, image_size)`, *optional*):
            Mask over the real pixels of each image tile.
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            image_hidden_states=image_hidden_states,
            labels=labels,
            **kwargs,
        )


__all__ = [
    "UltraVBertConfig",
    "UltraVBertPreTrainedModel",
    "UltraVBertModel",
    "UltraVBertForMaskedLM",
    "UltraVBertForSequenceClassification",
    "UltraVBertForTokenClassification",
]
