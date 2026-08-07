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
"""Video processor class for Ovis2.5."""

import math
from collections.abc import Iterable

import torch
import torchvision.transforms.v2.functional as tvF

from ...image_processing_utils import BatchFeature
from ...image_utils import ChannelDimension, PILImageResampling, SizeDict, get_image_size
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, auto_docstring
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import group_videos_by_shape, reorder_videos
from .image_processing_ovis2_5 import smart_resize


def _resolve_size(
    size: int | Iterable[int] | dict[str, int] | SizeDict | None,
    default_size: dict[str, int] | SizeDict,
    min_pixels: int | None,
    max_pixels: int | None,
) -> SizeDict:
    if size is None:
        size = default_size
    if isinstance(size, SizeDict):
        shortest_edge = size.shortest_edge
        longest_edge = size.longest_edge
    elif isinstance(size, dict):
        shortest_edge = size.get("shortest_edge")
        longest_edge = size.get("longest_edge")
    else:
        raise ValueError("`size` must be a dictionary or `SizeDict` with `shortest_edge` and `longest_edge`.")

    shortest_edge = min_pixels if min_pixels is not None else shortest_edge
    longest_edge = max_pixels if max_pixels is not None else longest_edge
    if shortest_edge is None or longest_edge is None:
        raise ValueError("`size` must contain `shortest_edge` and `longest_edge`.")
    return SizeDict(shortest_edge=shortest_edge, longest_edge=longest_edge)


def _validate_patch_grid(height: int, width: int, patch_size: int, merge_size: int) -> None:
    factor = patch_size * merge_size
    if height % factor != 0 or width % factor != 0:
        raise ValueError(
            "Ovis2.5 images must have height and width divisible by "
            f"`patch_size * merge_size` ({factor}), got ({height}, {width})."
        )


class Ovis2_5VideoProcessorKwargs(VideosKwargs, total=False):
    r"""
    min_pixels (`int`, *optional*, defaults to `448 * 448`):
        The minimum number of pixels in each resized video frame.
    max_pixels (`int`, *optional*, defaults to `1344 * 1792`):
        The maximum number of pixels in each resized video frame.
    patch_size (`int`, *optional*, defaults to 16):
        The spatial patch size used by the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size used by the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The spatial merge size between the vision encoder and language model.
    """

    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int


@auto_docstring
class Ovis2_5VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    size = {"shortest_edge": 448 * 448, "longest_edge": 1344 * 1792}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    # The original implementation receives an already sampled frame list.
    do_sample_frames = False
    valid_kwargs = Ovis2_5VideoProcessorKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Ovis2_5VideoProcessorKwargs]):
        size = _resolve_size(
            kwargs.pop("size", None),
            self.size,
            kwargs.pop("min_pixels", None),
            kwargs.pop("max_pixels", None),
        )
        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(
        self,
        size: SizeDict | dict[str, int] | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ) -> dict:
        size = _resolve_size(size, self.size, min_pixels, max_pixels)
        kwargs = super()._standardize_kwargs(size=size, **kwargs)
        return kwargs

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_convert_rgb: bool,
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
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        if any(video.shape[0] == 0 for video in videos):
            raise ValueError("Ovis2.5 does not support videos with zero frames.")

        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            height, width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = self.resize(
                    image=stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            else:
                resized_height, resized_width = height, width
            _validate_patch_grid(resized_height, resized_width, patch_size, merge_size)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            patches = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            num_frames = patches.shape[1]
            if pad := -num_frames % temporal_patch_size:
                repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
                patches = torch.cat((patches, repeats), dim=1)

            batch_size, padded_num_frames, channel = patches.shape[:3]
            grid_t = padded_num_frames // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
            patches = patches.reshape(
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
            # [batch, time, merged_h, merged_w, merge_h, merge_w, channel, temporal, patch_h, patch_w]
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids_ordered = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.long)
        return BatchFeature(
            data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
            tensor_type=return_tensors,
        )

    def get_number_of_video_patches(self, num_frames: int, height: int, width: int, videos_kwargs=None) -> int:
        """Return the number of pre-merge vision patches for one video."""
        if num_frames <= 0:
            raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
        videos_kwargs = videos_kwargs or {}
        patch_size = videos_kwargs.get("patch_size", self.patch_size)
        temporal_patch_size = videos_kwargs.get("temporal_patch_size", self.temporal_patch_size)
        merge_size = videos_kwargs.get("merge_size", self.merge_size)
        do_resize = videos_kwargs.get("do_resize", self.do_resize)
        if do_resize:
            min_pixels = videos_kwargs.get("min_pixels", self.size["shortest_edge"])
            max_pixels = videos_kwargs.get("max_pixels", self.size["longest_edge"])
            height, width = smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        _validate_patch_grid(height, width, patch_size, merge_size)
        grid_t = math.ceil(num_frames / temporal_patch_size)
        return grid_t * (height // patch_size) * (width // patch_size)


__all__ = ["Ovis2_5VideoProcessor"]
