# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""video processor class for KimiK2.5."""

import math

import torch
import torchvision.transforms.v2.functional as tvF

from ...image_processing_utils import BatchFeature
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import group_videos_by_shape, reorder_videos


# Same resize as in image processing
def navit_resize(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    max_patches: int,
    max_size_per_side: int,
):
    num_patches_w = max(1.0, width // patch_size)
    num_patches_h = max(1.0, height // patch_size)
    current_patch_count = num_patches_w * num_patches_h

    # Scale to satisfy total patch budget (affects both dims, hence sqrt)
    scale_for_total_patches = math.sqrt(max_patches / current_patch_count)

    # Scale to satisfy per-side patch budget
    scale_for_width_patches = (max_size_per_side * patch_size) / width
    scale_for_height_patches = (max_size_per_side * patch_size) / height

    # Use the most restrictive scale, never upscale
    scale = min(1.0, scale_for_total_patches, scale_for_width_patches, scale_for_height_patches)

    # Make sure the resized size doesn't go beyond predefined `max`
    new_width, new_height = max(1, int(width * scale)), max(1, int(height * scale))
    new_width = min(new_width, max_size_per_side * patch_size)
    new_height = min(new_height, max_size_per_side * patch_size)

    # Calculate the padding to make the height and width divisible by the merge kernel size and patch size.
    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_height % factor) % factor + new_height
    pad_width = (factor - new_width % factor) % factor + new_width

    return (new_height, new_width), (pad_height, pad_width)


class Kimi_K25VideoProcessorInitKwargs(VideosKwargs, total=False):
    r"""
    max_patches (`int`, *optional*, defaults to `16384`):
        The max limit to resize resize the video.
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    merge_kernel_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    temporal_patch_size (`int`, *optional*, defaults to 4):
        The temporal patch size of the vision encoder.
    """

    max_patches: int
    patch_size: int
    merge_size: int
    temporal_patch_size: int


class Kimi_K25VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"max_height": 512, "max_width": 512}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 4
    merge_size = 2
    max_patches = 4096
    fps = 2
    do_sample_frames = True
    valid_kwargs = Kimi_K25VideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw", "num_chunks_per_video"]

    def __init__(self, **kwargs: Unpack[Kimi_K25VideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(
        self,
        size: SizeDict | None = None,
        **kwargs,
    ) -> dict:
        if size is not None:
            if size.max_height is None or size.max_width is None or (size.max_height != size.max_width):
                raise ValueError(
                    f"size must contain 'max_height' and 'max_width' keys with identical values but got {size}."
                )
        super()._validate_preprocess_kwargs(size=size, **kwargs)

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        max_patches: int,
        return_tensors: str | TensorType | None,
        **kwargs,
    ):
        # Split video to chunks based on temporal patch size
        chunked_videos, num_chunks_per_video = [], []
        for video in videos:
            for chunk in range(0, video.shape[0], temporal_patch_size):
                video_chunk = video[chunk : chunk + temporal_patch_size]
                chunked_videos.append(video_chunk)
            num_chunks_per_video.append(math.ceil(video.shape[0] / temporal_patch_size))

        # Group videos by size for batched resizing and padding
        grouped_videos, grouped_videos_index = group_videos_by_shape(chunked_videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            height, width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            resized_height, resized_width = height, width
            if do_resize:
                (resized_height, resized_width), (pad_height, pad_width) = navit_resize(
                    height,
                    width,
                    patch_size=patch_size,
                    merge_kernel_size=merge_size,
                    max_patches=max_patches,
                    max_size_per_side=size.max_height,
                )
                stacked_videos = self.resize(
                    image=stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_videos = self.pad(stacked_videos, pad_size=SizeDict(height=pad_height, width=pad_width))
                stacked_videos = torch.stack(stacked_videos, dim=0)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            batch_size, time, channels, height, width = stacked_videos.shape
            grid_h, grid_w = height // patch_size, width // patch_size
            patches = stacked_videos.reshape(batch_size, time, channels, grid_h, patch_size, grid_w, patch_size)
            patches = patches.permute(0, 1, 3, 5, 2, 4, 6)

            processed_videos_grouped[shape] = patches.reshape(batch_size, -1, channels, patch_size, patch_size)
            processed_grids[shape] = [[time, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
                "num_chunks_per_video": num_chunks_per_video,
            },
            tensor_type=return_tensors,
        )


__all__ = ["Kimi_K25VideoProcessor"]
