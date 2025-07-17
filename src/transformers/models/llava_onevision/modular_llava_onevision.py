# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, Union

import torch
from torch import nn

from transformers.models.llava_next.image_processing_llava_next_fast import LlavaNextImageProcessorFast
from transformers.models.llava_next_video.modeling_llava_next_video import (
    LlavaNextVideoCausalLMOutputWithPast,
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoModel,
    LlavaNextVideoModelOutputWithPast,
    LlavaNextVideoPreTrainedModel,
    TransformersKwargs,
    get_anyres_image_grid_shape,
    image_size_to_num_patches,
    unpad_image,
)

from ...cache_utils import Cache
from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import DefaultFastImageProcessorKwargs, group_images_by_shape, reorder_images
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
)


if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


logger = logging.get_logger(__name__)


class LlavaOnevisionFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    image_grid_pinpoints (`list[list[int]]`, *optional*):
        A list of possible resolutions to use for processing high resolution images. The best resolution is selected
        based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
        method.
    do_pad (`bool`, *optional*):
        Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
        number of patches in the batch. Padding will be applied to the bottom and right with zeros.
    """

    image_grid_pinpoints: Optional[list[list[int]]]
    do_pad: Optional[bool]


class LlavaOnevisionImageProcessorFast(LlavaNextImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    crop_size = None
    default_to_square = False
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    image_grid_pinpoints = [[384, 384], [384, 768], [384, 1152], [384, 1536], [384, 1920], [384, 2304], [768, 384], [768, 768], [768, 1152], [768, 1536], [768, 1920], [768, 2304], [1152, 384], [1152, 768], [1152, 1152], [1152, 1536], [1152, 1920], [1152, 2304], [1536, 384], [1536, 768], [1536, 1152], [1536, 1536], [1536, 1920], [1536, 2304], [1920, 384], [1920, 768], [1920, 1152], [1920, 1536], [1920, 1920], [1920, 2304], [2304, 384], [2304, 768], [2304, 1152], [2304, 1536], [2304, 1920], [2304, 2304]]  # fmt: skip
    model_input_names = ["pixel_values_videos"]

    # Copied from transformers.models.llava.image_processing_llava_fast.LlavaImageProcessorFast.pad_to_square
    def pad_to_square(
        self,
        images: "torch.Tensor",
        background_color: Union[int, tuple[int, int, int]] = 0,
    ) -> "torch.Tensor":
        """
        Pads an image to a square based on the longest edge.

        Args:
            images (`np.ndarray`):
                The images to pad.
            background_color (`int` or `tuple[int, int, int]`, *optional*, defaults to 0):
                The color to use for the padding. Can be an integer for single channel or a
                tuple of integers representing for multi-channel images. If passed as integer
                in mutli-channel mode, it will default to `0` in subsequent channels.
        Returns:
            `torch.Tensor`: The padded images.
        """
        height, width = get_image_size(images, ChannelDimension.FIRST)

        if height == width:
            return images

        num_channels = images.shape[1] if len(images.shape) == 4 else images.shape[0]
        if isinstance(background_color, int):
            background_color = [background_color] + [0] * (num_channels - 1)
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )

        max_dim = max(height, width)
        paste_x_left = (max_dim - width) // 2
        paste_y_left = (max_dim - height) // 2
        paste_x_right = max_dim - width - paste_x_left
        paste_y_right = max_dim - height - paste_y_left
        padded_images = F.pad(
            images, padding=[paste_x_left, paste_y_left, paste_x_right, paste_y_right], fill=background_color
        )

        return padded_images

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[LlavaOnevisionFastImageProcessorKwargs]) -> BatchFeature:
        if isinstance(images, (tuple, list)) and isinstance(images[0], (tuple, list)):
            # if the first element is a list, we assume that all elements are lists
            batch_num_images = [len(x) for x in images]
        elif isinstance(images, (tuple, list)):
            # treat this as a single-image case for backward compatibility
            batch_num_images = [1] * len(images)
        else:
            batch_num_images = [1]
        kwargs["batch_num_images"] = batch_num_images
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        image_grid_pinpoints: list[list[int]],
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        do_pad: bool,
        batch_num_images: list[int],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        processed_images = []
        image_sizes = []

        # only single image patching is supported
        need_patching = [n == 1 for n in batch_num_images for _ in range(n)]

        # Determine the size tuple
        if size and size.height and size.width:
            size_tuple = (size.height, size.width)
        else:
            size_tuple = (size.shortest_edge, size.shortest_edge)

        # Determine the patch size
        if crop_size and crop_size.height:
            patch_size = crop_size.height
        elif size and size.height:
            patch_size = size.height
        else:
            patch_size = size.shortest_edge

        for i, image in enumerate(images):
            if need_patching[i]:
                image_patches = self._get_image_patches(
                    image,
                    image_grid_pinpoints,
                    size=size_tuple,
                    patch_size=patch_size,
                    interpolation=interpolation,
                )
            else:
                padded_image = self.pad_to_square(
                    images=image, background_color=tuple(int(x * 255) for x in self.image_mean)
                )
                image_patches = [padded_image]

            # Group images by size for batched processing
            processed_image_patches_grouped = {}
            grouped_image_patches, grouped_image_patches_index = group_images_by_shape(
                image_patches, disable_grouping=disable_grouping
            )
            for shape, stacked_image_patches in grouped_image_patches.items():
                if do_resize:
                    stacked_image_patches = self.resize(
                        image=stacked_image_patches,
                        size=size,
                        interpolation=interpolation,
                    )
                if do_center_crop:
                    stacked_image_patches = self.center_crop(stacked_image_patches, crop_size)
                # Fused rescale and normalize
                stacked_image_patches = self.rescale_and_normalize(
                    stacked_image_patches, do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
                processed_image_patches_grouped[shape] = stacked_image_patches
            processed_image_patches = reorder_images(processed_image_patches_grouped, grouped_image_patches_index)
            processed_image_patches = (
                torch.stack(processed_image_patches, dim=0) if return_tensors else processed_image_patches
            )
            processed_images.append(processed_image_patches)
            image_sizes.append(get_image_size(image, ChannelDimension.FIRST))

        if do_pad:
            processed_images = self._pad_for_batching(processed_images)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes, "batch_num_images": batch_num_images},
            tensor_type=return_tensors,
        )


class LlavaOnevisionModelOutputWithPast(LlavaNextVideoModelOutputWithPast):
    pass


class LlavaOnevisionCausalLMOutputWithPast(LlavaNextVideoCausalLMOutputWithPast):
    pass


class LlavaOnevisionPreTrainedModel(LlavaNextVideoPreTrainedModel):
    pass


class LlavaOnevisionModel(LlavaNextVideoModel):
    def __init__(self, config):
        super().__init__(config)
        del self.vision_resampler

    def pack_image_features(self, image_features, image_sizes, image_newline=None, vision_aspect_ratio="anyres_max_9"):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`list[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
            vision_aspect_ratio (`str`, *optional*, "anyres_max_9"):
                Aspect ratio used when processong image features. The default value is "anyres_max_9".
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`list[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                max_num_patches = int(vision_aspect_ratio.strip("anyres_max_"))
                channels, curr_height, curr_width = image_feature.shape
                ratio = math.sqrt(curr_height * curr_width / (max_num_patches * height**2))
                if ratio > 1.1:
                    image_feature = image_feature[None]
                    image_feature = nn.functional.interpolate(
                        image_feature, [int(curr_height // ratio), int(curr_width // ratio)], mode="bilinear"
                    )[0]
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
                image_feature = image_feature.flatten(0, 1)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features[0].device)
        return new_image_features, feature_lens

    def apply_pooling(self, image_features):
        height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
        batch_frames, seq_len, dim = image_features.shape
        image_features = image_features.view(batch_frames, height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2).contiguous()

        height, width = image_features.shape[2:]
        scaled_shape = [math.ceil(height / 2), math.ceil(width / 2)]
        image_features = nn.functional.interpolate(image_features, size=scaled_shape, mode="bilinear")

        image_features = image_features.permute(0, 2, 3, 1)
        image_features = image_features.view(batch_frames, -1, dim)
        return image_features

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        vision_aspect_ratio: Optional[str] = None,
        batch_num_images: Optional[torch.LongTensor] = None,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_patches, channels, height, width)`)
               The tensors corresponding to the input images.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            vision_feature_layer (`Union[int, list[int]]`):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
            batch_num_images (`torch.LongTensor`, *optional*):
                Number of images in each sample.
        Returns:
            image_features (list[`torch.Tensor`]): List of image feature tensor, each contains all the visual feature of all patches
            and are of shape `(num_patches, image_length, embed_dim)`).
        """
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        vision_aspect_ratio = (
            vision_aspect_ratio if vision_aspect_ratio is not None else self.config.vision_aspect_ratio
        )

        # ! infer image_num_patches from image_sizes
        if batch_num_images is None:
            # treat this as a single-image case for backward compatibility
            need_patching = [True] * len(image_sizes)
        else:
            need_patching = [n == 1 for n in batch_num_images for _ in range(n)]
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            if should_patch
            else 1
            for imsize, should_patch in zip(image_sizes, need_patching)
        ]
        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        image_features = self.vision_tower(pixel_values, output_hidden_states=True)
        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_features.hidden_states[vision_feature_layer]
        else:
            hs_pool = [image_features.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)

        image_features, feature_lens = self.pack_image_features(
            image_features,
            image_sizes,
            image_newline=self.image_newline,
            vision_aspect_ratio=vision_aspect_ratio,
        )
        return image_features

    def get_video_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, list[int]],
        vision_feature_select_strategy: str,
    ):
        """
        Obtains video last hidden states from the vision tower, apply multimodal projection and pooling.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_frames, channels, height, width)`)
               The tensors corresponding to the input video.
            vision_feature_layer (`Union[int, list[int]], *optional*, defaults to -2`):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            video_features (list[`torch.Tensor`]): List of video feature tensor, each contains all the visual feature of all patches
            and are of shape `(num_videos, video_length, embed_dim)`).
        """
        batch_size, frames, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * frames, channels, height, width)
        video_features = self.vision_tower(pixel_values, output_hidden_states=True)

        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_video_feature = video_features.hidden_states[vision_feature_layer]
        else:
            hs_pool = [video_features.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_video_feature = torch.cat(hs_pool, dim=-1)

        if vision_feature_select_strategy == "default":
            selected_video_feature = selected_video_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_video_feature = selected_video_feature
        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)

        return video_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_sizes_videos: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        vision_aspect_ratio: Optional[str] = None,
        batch_num_images: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, LlavaOnevisionModelOutputWithPast]:
        r"""
        image_sizes_videos (`torch.LongTensor` of shape `(batch_size, frames, 2)`, *optional*):
            The sizes of the videos in the batch, being (height, width) for each frame in the video.
        vision_aspect_ratio (`str`, *optional*, defaults to `"anyres_max_9"`):
            Aspect ratio used when processong image features. The default value is "anyres_max_9".
        batch_num_images (`torch.LongTensor`, *optional*):
            Number of images in each sample.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        vision_aspect_ratio = (
            vision_aspect_ratio if vision_aspect_ratio is not None else self.config.vision_aspect_ratio
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Images are processed with Anyres
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                batch_num_images=batch_num_images,
            )
            image_features = torch.cat(image_features, dim=0)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                special_image_mask = special_image_mask.all(-1)
            else:
                special_image_mask = input_ids == self.config.image_token_id

            n_image_tokens = (special_image_mask).sum()
            special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_features = image_features.shape[0]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Video are simply embedded and further pooled to decrease seq len
        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_newline = (
                self.image_newline[None, None, :].repeat(video_features.shape[0], 1, 1).to(video_features.device)
            )
            video_features = torch.cat((video_features, image_newline), dim=1)
            video_features = video_features.flatten(0, 1)

            if input_ids is None:
                special_video_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
                special_video_mask = special_video_mask.all(-1)
            else:
                special_video_mask = input_ids == self.config.video_token_id

            n_video_tokens = (special_video_mask).sum()
            special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_video_mask].numel() != video_features.numel():
                n_video_features = video_features.shape[0]
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
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
            **kwargs,
        )

        return LlavaOnevisionModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            video_hidden_states=video_features if pixel_values_videos is not None else None,
        )


class LlavaOnevisionForConditionalGeneration(LlavaNextVideoForConditionalGeneration):
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_sizes_videos: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        vision_aspect_ratio: Optional[str] = None,
        batch_num_images: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaOnevisionCausalLMOutputWithPast]:
        r"""
        image_sizes_videos (`torch.LongTensor` of shape `(batch_size, frames, 2)`, *optional*):
            The sizes of the videos in the batch, being (height, width) for each frame in the video.
        vision_aspect_ratio (`str`, *optional*, defaults to `"anyres_max_9"`):
            Aspect ratio used when processong image features. The default value is "anyres_max_9".
        batch_num_images (`torch.LongTensor`, *optional*):
            Number of images in each sample.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> import torch
        >>> from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

        >>> model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype="float16", device_map="cuda:0")
        >>> processor = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

        >>> conversation = [
        ...     {
        ...       "role": "user",
        ...       "content": [
        ...           {"type": "text", "text": "What is shown in this image?"},
        ...           {"type": "image"},
        ...         ],
        ...     },
        ... ]
        >>> prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        >>> image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> raw_image = Image.open(requests.get(image_file, stream=True).raw)
        >>> inputs = processor(text=prompt, images=raw_image, return_tensors='pt').to(0, torch.float16)

        >>> output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        >>> processor.batch_decode(output, skip_special_tokens=True)[0]
        "user\n\nWhat is shown in this image?\nassistant\ncat"
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        vision_aspect_ratio = (
            vision_aspect_ratio if vision_aspect_ratio is not None else self.config.vision_aspect_ratio
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_sizes=image_sizes,
            image_sizes_videos=image_sizes_videos,
            vision_aspect_ratio=vision_aspect_ratio,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            batch_num_images=batch_num_images,
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

        return LlavaOnevisionCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            video_hidden_states=outputs.video_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        pixel_values_videos=None,
        image_sizes_videos=None,
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
            model_inputs["image_sizes"] = image_sizes
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["image_sizes_videos"] = image_sizes_videos

        return model_inputs


__all__ = [
    "LlavaOnevisionImageProcessorFast",
    "LlavaOnevisionModel",
    "LlavaOnevisionForConditionalGeneration",
    "LlavaOnevisionPreTrainedModel",
]
