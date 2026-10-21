# Copyright 2026 Xiaomi Corporation and the HuggingFace Inc. team. All rights reserved.
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
"""Image processor for MiMo V2 multimodal models."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


class MiMoV2ImageProcessorKwargs(ImagesKwargs, total=False):
    """
    patch_size (`int`, *optional*, defaults to 14):
        Spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        Number of repeated image frames in each temporal patch.
    merge_size (`int`, *optional*, defaults to 2):
        Number of spatial patches merged along each axis.
    """

    patch_size: int
    temporal_patch_size: int
    merge_size: int


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Resize to multiples of `factor` within the pixel budget while preserving the aspect ratio.

    The MiMo V2 series first scales images smaller than one spatial merge block up to that block size.
    """
    if min(height, width, factor, min_pixels) <= 0 or max_pixels < min_pixels:
        raise ValueError("Image dimensions and pixel limits must be positive and min_pixels <= max_pixels.")
    if min(height, width) < factor:
        scale = factor / min(height, width)
        height, width = round(height * scale), round(width * scale)
    elif max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio must not exceed 200.")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        resized_height = math.floor(height / scale / factor) * factor
        resized_width = math.floor(width / scale / factor) * factor
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * scale / factor) * factor
        resized_width = math.ceil(width * scale / factor) * factor
    if min(resized_height, resized_width) == 0:
        raise ValueError("Pixel budget is too small for the image aspect ratio.")
    return resized_height, resized_width


def _preprocess_frames(
    frames: torch.Tensor,
    size: SizeDict,
    resample: PILImageResampling,
    do_resize: bool,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: float | list[float],
    image_std: float | list[float],
    patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    """Apply the shared float32 resize and normalization to channel-first image or video frames."""
    frames = frames.float()
    if do_resize:
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError("size must contain shortest_edge and longest_edge pixel budgets.")
        resized_size = smart_resize(
            *frames.shape[-2:],
            factor=patch_size * merge_size,
            min_pixels=size.shortest_edge,
            max_pixels=size.longest_edge,
        )
        interpolation_modes = {PILImageResampling.BILINEAR: "bilinear", PILImageResampling.BICUBIC: "bicubic"}
        if resample not in interpolation_modes:
            raise ValueError("The MiMo V2 series supports bilinear and bicubic resampling.")
        if tuple(frames.shape[-2:]) != resized_size:
            frames = F.interpolate(
                frames, size=resized_size, mode=interpolation_modes[resample], align_corners=False, antialias=False
            )
    if do_normalize:
        # Match XiaomiMiMo/MiMo-V2.6-Flash-RL by normalizing in source-pixel space.
        scale = 1 / rescale_factor if do_rescale else 1.0
        mean = (torch.as_tensor(image_mean, dtype=torch.float32, device=frames.device) * scale).view(-1, 1, 1)
        std = (torch.as_tensor(image_std, dtype=torch.float32, device=frames.device) * scale).view(-1, 1, 1)
        frames = (frames - mean) / std
    elif do_rescale:
        frames = frames * rescale_factor
    return frames


def _patchify_frames(
    frames: torch.Tensor, patch_size: int, temporal_patch_size: int, merge_size: int
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Flatten frames in spatial-merge, channel, temporal, patch order and return the temporal/spatial grid."""
    num_frames, num_channels, height, width = frames.shape
    if min(patch_size, temporal_patch_size, merge_size) <= 0:
        raise ValueError("Patch and merge sizes must be positive.")
    if num_frames % temporal_patch_size or height % (patch_size * merge_size) or width % (patch_size * merge_size):
        raise ValueError("Frame dimensions must be divisible by their temporal and spatial patch factors.")
    grid_t = num_frames // temporal_patch_size
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = frames.reshape(
        grid_t,
        temporal_patch_size,
        num_channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flattened_patches = patches.reshape(grid_t * grid_h * grid_w, num_channels * temporal_patch_size * patch_size**2)
    return flattened_patches, (grid_t, grid_h, grid_w)


@auto_docstring(
    custom_intro="Constructs an image processor for the MiMo V2 VL series.",
    checkpoint="XiaomiMiMo/MiMo-V2.6-Flash-RL",
)
class MiMoV2ImageProcessor(TorchvisionBackend):
    size = {"shortest_edge": 4 * 28 * 28, "longest_edge": 4096 * 28 * 28}
    default_to_square = False
    resample = PILImageResampling.BILINEAR
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    valid_kwargs = MiMoV2ImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[MiMoV2ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[MiMoV2ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def process_image(
        self,
        image: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(image, np.ndarray) and any(stride < 0 for stride in image.strides):
            image = np.ascontiguousarray(image)
        return super().process_image(
            image, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device, **kwargs
        )

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float],
        image_std: float | list[float],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        pixel_values = []
        image_grid_thw = []
        for image in images:
            frames = _preprocess_frames(
                image.unsqueeze(0),
                size=size,
                resample=resample,
                do_resize=do_resize,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                patch_size=patch_size,
                merge_size=merge_size,
            )
            patches, grid_thw = _patchify_frames(
                frames.expand(temporal_patch_size, -1, -1, -1),
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=merge_size,
            )
            pixel_values.append(patches)
            image_grid_thw.append(grid_thw)
        return BatchFeature(
            {"pixel_values": torch.cat(pixel_values), "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        size = SizeDict(**images_kwargs.get("size", self.size))
        if images_kwargs.get("do_resize", self.do_resize):
            height, width = smart_resize(height, width, patch_size * merge_size, size.shortest_edge, size.longest_edge)
        return (height // patch_size) * (width // patch_size)


__all__ = ["MiMoV2ImageProcessor"]
