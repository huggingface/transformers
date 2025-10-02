# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Video processor class for InstructBLIPVideo
"""

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling, SizeDict
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import group_videos_by_shape, reorder_videos


class InstructBlipVideoVideoProcessorInitKwargs(VideosKwargs): ...


class InstructBlipVideoVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    default_to_square = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = False  # Set to False for BC, recommended to set `True` in new models
    valid_kwargs = InstructBlipVideoVideoProcessorInitKwargs
    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs: Unpack[InstructBlipVideoVideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                stacked_videos = self.resize(stacked_videos, size=size, interpolation=interpolation)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_center_crop:
                stacked_videos = self.center_crop(stacked_videos, crop_size)
            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_videos_grouped[shape] = stacked_videos

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_videos = torch.stack(processed_videos, dim=0) if return_tensors else processed_videos

        return BatchFeature(data={"pixel_values": processed_videos}, tensor_type=return_tensors)


__all__ = ["InstructBlipVideoVideoProcessor"]
