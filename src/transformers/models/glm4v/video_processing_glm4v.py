# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
"""video processor class for GLM-4.1V."""

import math
from typing import Optional, Union

import numpy as np

from ...image_processing_utils import (
    BatchFeature,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    SizeDict,
    get_image_size,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_vision_available,
)
from .image_processing_glm4v import smart_resize


if is_torch_available():
    import torch

from ...utils.import_utils import requires
from ...video_processing_utils import (
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    BaseVideoProcessor,
)
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


if is_vision_available():
    from ...image_utils import PILImageResampling

import torch.nn.functional as F


class Glm4vVideoProcessorInitKwargs(VideosKwargs):
    max_image_size: dict[str, int] = None
    patch_size: Optional[int] = None
    temporal_patch_size: Optional[int] = None
    merge_size: Optional[int] = None
    image_mean: Optional[list[float]] = None
    image_std: Optional[list[float]] = None


@add_start_docstrings(
    "Constructs a fast GLM-4V image processor that dynamically resizes videos based on the original videos.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """,
)
@requires(backends=("torchvision",))
class Glm4vVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 2 * 30000}
    max_image_size = {"longest_edge": 28 * 28 * 2 * 30000}
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = True
    patch_size = 14
    temporal_patch_size = 2
    max_duration = 300
    merge_size = 2
    valid_kwargs = Glm4vVideoProcessorInitKwargs
    num_frames = 16
    fps = 2

    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Glm4vVideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def sample_frames(
        self,
        video: torch.Tensor,
        metadata: Union[VideoMetadata, dict],
    ):
        total_frames = video.shape[0]
        video_fps = getattr(metadata, "fps", 2.0)
        meta_frames = getattr(metadata, "total_num_frames", total_frames)
        max_frame_idx = meta_frames - 1
        duration = getattr(metadata, "duration", None)
        if duration is None:
            duration = round(max_frame_idx / video_fps) + 1

        if duration <= self.max_duration:
            n = int(math.floor(duration * self.fps))
            frame_indices = [min(max_frame_idx, int(math.ceil(i * video_fps / self.fps))) for i in range(n)]
        else:
            num_samples = int(self.max_duration * self.fps)
            if num_samples >= meta_frames:
                frame_indices = list(range(meta_frames))
            else:
                target_seconds = np.linspace(0, duration, num_samples, endpoint=True)
                frame_indices = [min(max_frame_idx, int(math.ceil(t * video_fps))) for t in target_seconds]

        seen, uniq = set(), []
        for idx in frame_indices:
            if idx not in seen:
                seen.add(idx)
                uniq.append(idx)

        if len(uniq) & 1:
            uniq.append(uniq[-1])

        frame_indices = uniq
        sampled_video = video[frame_indices]
        full_second_idxs = [int(idx / video_fps) for idx in frame_indices]
        second_idxs = full_second_idxs[::2]  # mrope
        return sampled_video, second_idxs

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        video_metadata: Optional[Union[list[VideoMetadata], list[dict]]] = None,
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        do_sample_frames: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        timestamps_list = []
        if do_sample_frames:
            if video_metadata is None or (isinstance(video_metadata, list) and video_metadata[0] is None):
                raise ValueError(
                    "Frame sampling is enabled but no video metadata was found. "
                    "Please pass in `VideoMetadata` object per each input video or set `do_sample_frames=False`"
                )
            processed_videos = []
            for video, metadata in zip(videos, video_metadata):
                video, timestamps = self.sample_frames(video, metadata)
                timestamps_list.append(timestamps)
                processed_videos.append(video)
        else:
            raise AssertionError("Must set `do_sample_frames=True` to sample frames from GLM-4.1V Model.")

        grouped_videos, grouped_videos_index = group_videos_by_shape(processed_videos)
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
                    max_pixels=self.max_image_size["longest_edge"],
                )
                stacked_videos = stacked_videos.view(B * T, C, H, W)
                stacked_videos = F.interpolate(
                    stacked_videos, size=(resized_height, resized_width), mode="bicubic", align_corners=False
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
        total_frames = video_grid_thw[0][0].item()
        h = video_grid_thw[0][1].item()
        w = video_grid_thw[0][2].item()
        video_grid_thw = [[1, h, w] for _ in range(total_frames)]
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "timestamps": timestamps_list,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Glm4vVideoProcessor"]
