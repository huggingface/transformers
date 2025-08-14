# coding=utf-8
# Copyright 2025 the Cohere Inc. team. All rights reserved.
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
"""PyTorch AyaVision model."""

from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from transformers.models.aya_vision.modeling_aya_vision import (
    AyaVisionCausalLMOutputWithPast,
    AyaVisionForConditionalGeneration,
    AyaVisionModel,
    AyaVisionModelOutputWithPast,
)
from transformers.models.got_ocr2.image_processing_got_ocr2_fast import GotOcr2ImageProcessorFast

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    auto_docstring,
    logging,
)
from ...utils.generic import check_model_inputs
from .configuration_cohere2_vision import Cohere2VisionConfig


logger = logging.get_logger(__name__)


class Cohere2VisionMultiModalProjector(nn.Module):
    def __init__(self, config: Cohere2VisionConfig):
        super().__init__()
        self.config = config
        self.downsample_factor = config.downsample_factor
        self.intermediate_size = config.alignment_intermediate_size
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * (config.downsample_factor**2), self.intermediate_size, bias=True
        )
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(self.intermediate_size // 2, config.text_config.hidden_size, bias=True)

    def pixel_shuffle(self, image_features):  # B, S, D
        batch_size, seq_length, feature_dim = image_features.shape
        height = width = int(seq_length**0.5)
        image_features = image_features.reshape(image_features.shape[0], width, height, -1)
        channels = image_features.shape[-1]
        image_features = image_features.reshape(
            batch_size, width, int(height / self.downsample_factor), int(channels * self.downsample_factor)
        )
        image_features = image_features.permute(0, 2, 1, 3)
        image_features = image_features.reshape(
            batch_size, int(height / self.downsample_factor), int(width / self.downsample_factor), -1
        )
        image_features = image_features.permute(0, 2, 1, 3)
        return image_features

    def forward(self, image_features):
        image_features = self.pixel_shuffle(image_features)
        hidden_states = self.linear_1(image_features)

        # Split along last dimension and apply SwiGLU
        x, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act(gate) * x

        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class Cohere2VisionModelOutputWithPast(AyaVisionModelOutputWithPast):
    pass


class Cohere2VisionCausalLMOutputWithPast(AyaVisionCausalLMOutputWithPast):
    pass


class Cohere2VisionModel(AyaVisionModel):
    _checkpoint_conversion_mapping = {}

    def get_image_features(self, pixel_values: torch.FloatTensor):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_patches, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (List[`torch.Tensor`]): List of image feature tensor, each contains all the visual feature of all patches
            and are of shape `(num_patches, image_length, embed_dim)`).
        """

        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, Cohere2VisionModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        return Cohere2VisionModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class Cohere2VisionForConditionalGeneration(AyaVisionForConditionalGeneration):
    _checkpoint_conversion_mapping = {}

    def get_image_features(self, pixel_values: torch.FloatTensor):
        return self.model.get_image_features(pixel_values=pixel_values)

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Cohere2VisionCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Cohere2VisionForConditionalGeneration
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("CohereLabs/command-a-vision-07-2025", use_fast=True)
        >>> model = Cohere2VisionForConditionalGeneration.from_pretrained("CohereLabs/command-a-vision-07-2025", device_map="auto")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {
        ...                 "type": "image",
        ...                 "url": "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg",
        ...             },
        ...             {"type": "text", "text": "what is in this image?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ... ).to(model.device)

        >>> gen_tokens = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.3)
        >>> processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            image_sizes=image_sizes,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return Cohere2VisionCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )


@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(max_image_tiles: int) -> list[tuple[int, int]]:
    """
    Computes all allowed aspect ratios for a given maximum number of input tiles.

    This function calculates all possible arrangements of tiles that can be formed
    within the constraint of the maximum number of tiles. Each arrangement is
    represented by its aspect ratio (width/height) and the corresponding tile configuration.

    Args:
        max_image_tiles (`int`):
            The maximum number of tiles allowed.

    Returns:
        `list[tuple[int, int]]`: A list of tuples, each tuple representing a valid (width, height)
        configuration in terms of number of tiles.

    Example:
        >>> get_all_supported_aspect_ratios(4)
        [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1), (4, 1)]

    """
    aspect_ratios = []
    for width in range(1, max_image_tiles + 1):
        for height in range(1, max_image_tiles + 1):
            if width * height <= max_image_tiles:
                aspect_ratios.append((width, height))
    return aspect_ratios


def get_optimal_tiled_canvas(
    original_image_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    min_image_tiles: int,
    max_image_tiles: int,
) -> tuple[int, int]:
    possible_resolutions = get_all_supported_aspect_ratios(max_image_tiles)
    possible_resolutions = sorted(possible_resolutions, key=lambda x: x[0] * x[1])
    image_height, image_width = original_image_size
    patch_size_height, patch_size_width = target_tile_size  # (height == width)

    candidate_resolutions = np.array(possible_resolutions) * patch_size_height
    original_size = np.stack([image_height, image_width])
    required_scales = candidate_resolutions / original_size
    required_scale = np.min(required_scales, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if np.all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        best_grid = possible_resolutions[np.argmax(required_scale)]
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        best_grid = possible_resolutions[np.argmin(required_scale)]
    return best_grid


@auto_docstring
class Cohere2VisionImageProcessorFast(GotOcr2ImageProcessorFast):
    size = {"height": 512, "width": 512}
    min_patches = 1
    max_patches = 12
    crop_to_patches = True
    patch_size = 16


__all__ = [
    "Cohere2VisionForConditionalGeneration",
    "Cohere2VisionPreTrainedModel",  # noqa: F822
    "Cohere2VisionModel",
    "Cohere2VisionImageProcessorFast",
]
