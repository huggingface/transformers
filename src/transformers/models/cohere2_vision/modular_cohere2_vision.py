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

import numpy as np
import torch
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from transformers.models.aya_vision.modeling_aya_vision import (
    AyaVisionCausalLMOutputWithPast,
    AyaVisionForConditionalGeneration,
    AyaVisionModel,
    AyaVisionModelOutputWithPast,
    AyaVisionPreTrainedModel,
)
from transformers.models.got_ocr2.image_processing_got_ocr2 import GotOcr2ImageProcessor

from ...cache_utils import Cache
from ...image_utils import PILImageResampling, SizeDict
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPooling
from ...modeling_utils import slice_logits_to_keep
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
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


class Cohere2VisionPreTrainedModel(AyaVisionPreTrainedModel):
    base_model_prefix = "model"


class Cohere2VisionModel(AyaVisionModel):
    def __init__(self, config: Cohere2VisionConfig):
        super().__init__(config)

        self.multi_modal_projector = Cohere2VisionMultiModalProjector(config)

    @can_return_tuple
    @auto_docstring(
        custom_intro="Obtains image last hidden states from the vision tower and apply multimodal projection."
    )
    def get_image_features(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        image_outputs = self.vision_tower(pixel_values, return_dict=True, **kwargs)
        selected_image_feature = image_outputs.last_hidden_state
        pooler_output = self.multi_modal_projector(selected_image_feature)
        image_outputs.pooler_output = pooler_output.reshape(
            selected_image_feature.shape[0], -1, self.config.text_config.hidden_size
        )

        return image_outputs

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | Cohere2VisionModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and mm_encoder_outputs is not None:
            raise ValueError("You cannot specify both pixel_values and mm_encoder_outputs at the same time")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        mm_encoder_outputs = mm_encoder_outputs if mm_encoder_outputs is not None else {}
        if mm_encoder_outputs.get("image") is None and pixel_values is not None:
            mm_encoder_outputs["image"] = self.get_image_features(pixel_values, return_dict=True)

        if mm_encoder_outputs.get("image") is not None:
            image_features = mm_encoder_outputs["image"].pooler_output
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
            **kwargs,
        )

        return Cohere2VisionModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if mm_encoder_outputs.get("image") is not None else None,
        )


class Cohere2VisionForConditionalGeneration(AyaVisionForConditionalGeneration):
    def __init__(self, config: Cohere2VisionConfig):
        super().__init__(config)
        self.model = Cohere2VisionModel(config)

    @auto_docstring
    def get_image_features(
        self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs]
    ) -> tuple | BaseModelOutputWithPooling:
        return self.model.get_image_features(pixel_values=pixel_values, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_sizes: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Cohere2VisionCausalLMOutputWithPast:
        r"""
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
            image_sizes=image_sizes,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        hidden_states = outputs[0]
        hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(hidden_states)

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
    # tiles following (width, height) order to align with aspect ratio convention
    tile_size = np.stack([image_width, image_height])
    required_scales = candidate_resolutions / tile_size
    required_scale = np.min(required_scales, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if np.all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        best_grid = possible_resolutions[np.argmax(required_scale)]
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = np.where(required_scale < 1.0, 10e9, required_scale)
        best_grid = possible_resolutions[np.argmin(required_scale)]
    return best_grid  # (width, height)


class Cohere2VisionImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    crop_to_patches (`bool`, *optional*, defaults to `False`):
        Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the
        `preprocess` method.
    min_patches (`int`, *optional*, defaults to 1):
        The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
        set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.
    max_patches (`int`, *optional*, defaults to 12):
        The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
        set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.
    """

    crop_to_patches: bool
    min_patches: int
    max_patches: int


@auto_docstring
class Cohere2VisionImageProcessor(GotOcr2ImageProcessor):
    size = {"height": 512, "width": 512}
    min_patches = 1
    max_patches = 12
    crop_to_patches = True
    patch_size = 16
    valid_kwargs = Cohere2VisionImageProcessorKwargs

    def crop_image_to_patches(
        self,
        images: "torch.Tensor",
        min_patches: int,
        max_patches: int,
        use_thumbnail: bool = True,
        patch_size: SizeDict | None = None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
    ):
        """
        Crop the images to patches and return a list of cropped images.
        The number of patches and their grid arrangement are determined by the original image size,
        the target patch size and the minimum and maximum number of patches.
        The aspect ratio of the patches grid is chosen to be the closest to the original image aspect ratio.

        Args:
            images (`torch.Tensor`):
                The images to be cropped.
            min_patches (`int`):
                The minimum number of patches to be extracted from the image.
            max_patches (`int`):
                The maximum number of patches to be extracted from the image.
            use_thumbnail (`bool`, *optional*, defaults to `True`):
                Whether to add a thumbnail image to the list of cropped patches.
            patch_size (`SizeDict`, *optional*):
                The size of the output patches.
            resample (`PILImageResampling | tvF.InterpolationMode | int | None`, *optional*):
                Resampling filter to use when resizing.
        """
        patch_size_height, patch_size_width = patch_size.height, patch_size.width
        original_height, original_width = images.shape[-2:]
        # find the closest aspect ratio to the target
        num_columns, num_rows = get_optimal_tiled_canvas(
            (original_height, original_width), (patch_size_height, patch_size_width), min_patches, max_patches
        )

        # calculate the target width and height
        target_width = patch_size_width * num_columns
        target_height = patch_size_height * num_rows
        num_blocks = num_columns * num_rows

        # resize the image so that each patch is of patch_size
        resized_image = self.resize(images, SizeDict(height=target_height, width=target_width), resample=resample)
        # split the image into patches
        processed_images = []
        for i in range(num_blocks):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * patch_size_width,
                row * patch_size_height,
                (column + 1) * patch_size_width,
                (row + 1) * patch_size_height,
            )
            # split the image
            patch_image = resized_image[..., box[1] : box[3], box[0] : box[2]]
            processed_images.append(patch_image)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = self.resize(images, patch_size, resample=resample)
            processed_images.append(thumbnail_img)

        processed_images = torch.stack(processed_images, dim=0).transpose(0, 1).contiguous()

        return processed_images

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """
        A utility that returns number patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of patches per image.
        """
        images_kwargs = images_kwargs or {}
        min_patches = images_kwargs.get("min_patches", self.min_patches)
        max_patches = images_kwargs.get("max_patches", self.max_patches)
        patch_size = images_kwargs.get("patch_size", self.size)
        crop_to_patches = images_kwargs.get("crop_to_patches", self.crop_to_patches)

        num_patches = 1
        if crop_to_patches and max_patches > 1:
            if isinstance(patch_size, dict):
                patch_height, patch_width = patch_size["height"], patch_size["width"]
            else:
                patch_height, patch_width = patch_size.height, patch_size.width
            num_columns, num_rows = get_optimal_tiled_canvas(
                (height, width), (patch_height, patch_width), min_patches, max_patches
            )
            if num_columns * num_rows > 1:
                num_patches += num_columns * num_rows

        return num_patches


__all__ = [
    "Cohere2VisionForConditionalGeneration",
    "Cohere2VisionPreTrainedModel",
    "Cohere2VisionModel",
    "Cohere2VisionImageProcessor",
]
