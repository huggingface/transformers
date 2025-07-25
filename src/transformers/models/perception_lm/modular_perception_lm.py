# coding=utf-8
# Copyright 2025 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch PerceptionLM model."""

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...utils import (
    auto_docstring,
    can_return_tuple,
    logging,
)
from ..auto import AutoModel
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast,
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaModelOutputWithPast,
    LlavaPreTrainedModel,
)
from .configuration_perception_lm import PerceptionLMConfig


logger = logging.get_logger(__name__)


class PerceptionLMAdaptiveAvgPooling(nn.Module):
    def __init__(self, pooling_ratio=2):
        super().__init__()
        self.pooling_ratio = pooling_ratio

    def forward(self, hidden_states):
        b, num_tokens, c = hidden_states.shape
        h = int(math.sqrt(num_tokens))
        if h * h != num_tokens:
            raise ValueError(f"num_tokens {num_tokens} is expected to be a square number")

        shape = (h // self.pooling_ratio, h // self.pooling_ratio)
        hidden_states = hidden_states.permute(0, 2, 1).reshape(b, -1, h, h)
        hidden_states = F.adaptive_avg_pool2d(hidden_states, shape)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states


class PerceptionLMMultiModalProjector(nn.Module):
    def __init__(self, config: PerceptionLMConfig):
        super().__init__()
        input_size = config.vision_config.model_args["embed_dim"]
        output_size = config.text_config.hidden_size
        self.linear_1 = nn.Linear(
            in_features=input_size,
            out_features=output_size,
            bias=True,
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            in_features=output_size,
            out_features=output_size,
            bias=True,
        )
        self.pooling = (
            PerceptionLMAdaptiveAvgPooling(config.projector_pooling_ratio)
            if config.projector_pooling_ratio > 1
            else nn.Identity()
        )

    def forward(self, features):
        features = features.permute(1, 0, 2)  # NLD -> LND
        features = self.linear_1(features)
        features = self.gelu(features)
        features = self.linear_2(features)
        features = features.permute(1, 0, 2)  # LND -> NLD
        features = self.pooling(features)
        return features


class PerceptionLMPreTrainedModel(LlavaPreTrainedModel):
    base_model_prefix = "model"


class PerceptionLMModelOutputWithPast(LlavaModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        Image hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_videos, sequence_length, hidden_size)`.
        Video hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    video_hidden_states: Optional[torch.FloatTensor] = None


class PerceptionLMCausalLMOutputWithPast(LlavaCausalLMOutputWithPast):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        Image hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_videos, sequence_length, hidden_size)`.
        Video hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    video_hidden_states: Optional[torch.FloatTensor] = None


@auto_docstring
class PerceptionLMModel(LlavaModel):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: PerceptionLMConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = PerceptionLMMultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_tiles, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_tiles, num_patches, embed_dim)`).
        """
        image_outputs = self.vision_tower(pixel_values.flatten(0, 1))
        image_outputs = image_outputs.last_hidden_state
        if self.config.vision_use_cls_token:
            image_outputs = image_outputs[:, 1:, :]
        image_features = self.multi_modal_projector(image_outputs)
        return image_features

    def check_mask_feature_size_match(self, media_mask, media_features):
        media_token_count = media_mask.sum()
        media_feature_size = media_features.size()[:-1].numel()
        if media_token_count != media_feature_size:
            raise ValueError(
                f"The number of tokens in the media mask ({media_token_count}) does not match the number of features in the media features ({media_feature_size}. Features shape: {media_features.shape})"
            )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[tuple, PerceptionLMModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if (pixel_values is not None or pixel_values_videos is not None) and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both (pixel_values or pixel_values_videos) and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values.to(inputs_embeds),
            )
            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            self.check_mask_feature_size_match(special_image_mask, image_features)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = image_features.to(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        video_features = None
        if pixel_values_videos is not None:
            video_features = self.get_image_features(
                pixel_values=pixel_values_videos.to(inputs_embeds),
            )
            special_video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1)
            self.check_mask_feature_size_match(special_video_mask, video_features)
            special_video_mask = special_video_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            video_features = video_features.to(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )
        return PerceptionLMModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            past_key_values=outputs.past_key_values,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            video_hidden_states=(video_features if pixel_values_videos is not None else None),
        )


@auto_docstring
class PerceptionLMForConditionalGeneration(LlavaForConditionalGeneration):
    _checkpoint_conversion_mapping = {}

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_videos"] = pixel_values_videos
        return model_inputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[tuple, PerceptionLMCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **lm_kwargs,
            )

        return PerceptionLMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            video_hidden_states=outputs.video_hidden_states,
        )

    def get_image_features(self, **kwargs):
        raise AttributeError("Not needed for PerceptionLM")

    def language_model(self):
        raise AttributeError("Not needed for PerceptionLM")

    def vision_tower(self):
        raise AttributeError("Not needed for PerceptionLM")

    def multi_modal_projector(self):
        raise AttributeError("Not needed for PerceptionLM")


__all__ = [
    "PerceptionLMForConditionalGeneration",
    "PerceptionLMPreTrainedModel",
    "PerceptionLMModel",
]
