# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""video processor class for Qwen3-VL."""

import math
from typing import Optional, Union

import numpy as np
import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ChannelDimension, PILImageResampling, SizeDict, get_image_size
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, add_start_docstrings, logging
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


logger = logging.get_logger(__name__)


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
):
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class Qwen3VLVideoProcessorInitKwargs(VideosKwargs):
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    min_frames: Optional[int]
    max_frames: Optional[int]


@add_start_docstrings(
    "Constructs a fast Qwen3-VL image processor that dynamically resizes videos based on the original videos.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        patch_size (`int`, *optional*, defaults to 16):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """,
)
class Qwen3VLVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 128 * 32 * 32, "longest_edge": 32 * 32 * 768}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    fps = 2
    min_frames = 4
    max_frames = 768
    do_sample_frames = True
    valid_kwargs = Qwen3VLVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Qwen3VLVideoProcessorInitKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (
            self.size.get("shortest_edge", None) is None or self.size.get("longest_edge", None) is None
        ):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        return super()._further_process_kwargs(size=size, **kwargs)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        **kwargs,
    ):
        """
        Default sampling function which uniformly samples the desired number of frames between 0 and total number of frames.
        If `fps` is passed along with metadata, `fps` frames per second are sampled uniformty. Arguments `num_frames`
        and `fps` are mutually exclusive.

        Args:
            video (`torch.Tensor`):
                Video that need to be sampled.
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. Defaults to `self.num_frames`.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.fps`.
        Returns:
            torch.Tensor:
                Sampled video frames.
        """
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        total_num_frames = metadata.total_num_frames
        fps = fps if fps is not None else self.fps

        # If num_frames is not given but fps is, calculate num_frames from fps
        if num_frames is None and fps is not None:
            if metadata.fps is None:
                metadata.fps = 24
                logger.warning_once(
                    "Asked to sample `fps` frames per second but no video metadata was provided which is required when sampling with `fps`. "
                    "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                )
            num_frames = int(total_num_frames / metadata.fps * fps)
            num_frames = min(min(max(num_frames, self.min_frames), self.max_frames), total_num_frames)

        if num_frames is None:
            num_frames = min(max(total_num_frames, self.min_frames), self.max_frames)

        indices = np.linspace(0, total_num_frames - 1, num_frames).round().astype(int)

        return indices

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Optional[SizeDict] = None,
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            B, T, C, H, W = stacked_videos.shape
            num_frames, height, width = T, H, W
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.view(B * T, C, H, W)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
                stacked_videos = stacked_videos.view(B, T, C, resized_height, resized_width)
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
            patches = stacked_videos

            # Check that videos have `num_frames` divisible by `temporal_patch_size`
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Qwen3VLVideoProcessor"]
