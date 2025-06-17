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
from typing import List, Optional, Tuple, Union

import timm
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.generation.utils import GenerationMixin

# from ...generation import GenerationMixin
from ...modeling_utils import PreTrainedModel
from ...utils import (
    auto_docstring,
    logging,
)
from ..auto import AutoModel
from ..llava.modeling_llava import (
    LlavaCausalLMOutputWithPast as PerceptionLMCausalLMOutputWithPast,
)
from ..llava.modeling_llava import (
    LlavaModel,
    LlavaPreTrainedModel,
)
from .configuration_perception_lm import PerceptionEncoderConfig, PerceptionLMConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PerceptionLMConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/Perception-LM-1B"


class PerceptionEncoder(PreTrainedModel):
    def __init__(self, config: PerceptionEncoderConfig):
        super().__init__(config)
        self.use_cls_token = config.use_cls_token
        self.eva_pe = timm.create_model(
            config.architecture,
            img_size=config.img_size,
            depth=config.depth,
            num_classes=config.num_classes,
            global_pool=config.global_pool,
            use_post_transformer_norm=config.use_post_transformer_norm,
            init_values=config.init_values,
            ref_feat_shape=config.ref_feat_shape,
            embed_dim=config.width,
        )

    def forward(self, x):
        x = self.eva_pe(x)
        if self.use_cls_token:
            return x[:, 1:, :]
        else:
            return x


class AdaptiveAvgPooling(nn.Module):
    def __init__(self, pooling_ratio=2):
        super(AdaptiveAvgPooling, self).__init__()
        self.pooling_ratio = pooling_ratio

    def forward(self, hidden_states):
        b, num_tokens, c = hidden_states.shape
        h = int(math.sqrt(num_tokens))
        if h * h != num_tokens:
            raise ValueError(
                f"num_tokens {num_tokens} is expected to be a square number"
            )

        shape = (h // self.pooling_ratio, h // self.pooling_ratio)
        hidden_states = hidden_states.permute(0, 2, 1).reshape(b, -1, h, h)
        hidden_states = F.adaptive_avg_pool2d(hidden_states, shape)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states


class PerceptionLMMultiModalProjector(nn.Module):
    def __init__(self, config: PerceptionLMConfig):
        super().__init__()
        input_size = config.vision_config.width
        output_size = config.text_config.hidden_size
        self.projector = nn.ModuleList(
            [
                nn.Linear(
                    in_features=input_size,
                    out_features=output_size,
                    bias=True,
                ),
                nn.GELU(),
                nn.Linear(
                    in_features=output_size,
                    out_features=output_size,
                    bias=True,
                ),
            ]
        )
        self.pooling = (
            AdaptiveAvgPooling(config.projector_pooling_ratio)
            if config.projector_pooling_ratio > 1
            else nn.Identity()
        )

    def forward(self, features):
        features = features.permute(1, 0, 2)  # NLD -> LND
        for layer in self.projector:
            features = layer(features)
        features = features.permute(1, 0, 2)  # LND -> NLD
        features = self.pooling(features)
        return features


class PerceptionLMPreTrainedModel(LlavaPreTrainedModel):
    base_model_prefix = ""


@auto_docstring
class PerceptionLMModel(LlavaModel):
    def __init__(self, config: PerceptionLMConfig):
        super().__init__(config)
        del self.vision_tower
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
        image_features = self.multi_modal_projector(image_outputs)
        return image_features

    def check_mask_feature_size_match(self, media_mask, media_features):
        media_token_count = media_mask.sum()
        media_feature_size = media_features.size()[:-1].numel()
        if media_token_count != media_feature_size:
            raise ValueError(
                f"The number of tokens in the media mask ({media_token_count}) does not match the number of features in the media features ({media_feature_size}. Features shape: {media_features.shape})"
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[Tuple, PerceptionLMCausalLMOutputWithPast]:
        r"""
            Args:
                input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                    Indices of input sequence tokens in the vocabulary.
                pixel_values (`torch.FloatTensor` of shape `(batch_size, num_tiles, channels, height, width)`, *optional*):
                    Pixel values for input images.
                pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, channels, height, width)`, *optional*):
                    Pixel values for input videos.
                attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                    Mask to avoid performing attention on padding token indices.
                position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                    Indices of positions of each input sequence token in the position embeddings.
                past_key_values (`List[torch.FloatTensor]`, *optional*):
                    List of precomputed key and value hidden states for each layer, used for fast autoregressive generation.
                inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                    Optionally, instead of passing `input_ids`, you can choose to directly pass an embedded representation.
                use_cache (`bool`, *optional*):
                    Whether or not to use past key values to speed up decoding.
                output_attentions (`bool`, *optional*):
                    Whether or not to return the attentions tensors of all attention layers.
                output_hidden_states (`bool`, *optional*):
                    Whether or not to return the hidden states of all layers.
                return_dict (`bool`, *optional*):
                    Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
                cache_position (`torch.LongTensor`, *optional*):
                    Position indices for cached key/value states, used for efficient generation.
                logits_to_keep (`int` or `torch.Tensor`, *optional*):
                    If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                    `input_ids` (special case). If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the
                    sequence length dimension.
                lm_kwargs:
                    Additional keyword arguments passed to the language model.
        Example:
        (TODO: fix example)
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, PerceptionLMForConditionalGeneration

        >>> model = PerceptionLMForConditionalGeneration.from_pretrained("facebook/Perception-LM-1B")
        >>> processor = AutoProcessor.from_pretrained("facebook/Perception-LM-1B")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if (
            pixel_values is not None or pixel_values_videos is not None
        ) and inputs_embeds is not None:
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
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            image_features = image_features.to(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        if pixel_values_videos is not None:
            video_features = self.get_image_features(
                pixel_values=pixel_values_videos.to(inputs_embeds),
            )
            special_video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1)
            self.check_mask_feature_size_match(special_video_mask, video_features)
            special_video_mask = special_video_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            video_features = video_features.to(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                special_video_mask, video_features
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )
        return outputs, image_features


@auto_docstring
class PerceptionLMForConditionalGeneration(
    PerceptionLMPreTrainedModel, GenerationMixin
):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: PerceptionLMConfig, **super_kwargs):
        super().__init__(config, **super_kwargs)
        self.model = PerceptionLMModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[Tuple, PerceptionLMCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).
        Returns:
        Example:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, PerceptionLMForConditionalGeneration
        >>> model = PerceptionLMForConditionalGeneration.from_pretrained("facebook/Perception-LM-1B")
        >>> processor = AutoProcessor.from_pretrained("facebook/Perception-LM-1B")
        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""

        outputs, image_features = self.model(
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
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
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
            image_hidden_states=image_features if pixel_values is not None else None,
        )


__all__ = [
    "PerceptionLMForConditionalGeneration",
    "PerceptionLMPreTrainedModel",
    "PerceptionEncoder",
]
