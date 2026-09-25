# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import torch
import torchvision.transforms.v2.functional as tvF

from ...feature_extraction_utils import BatchFeature
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling, SizeDict
from ...utils import TensorType
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import group_videos_by_shape, reorder_videos


class CogVLM2VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 224}
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = False
    return_metadata = False
    model_input_names = ["pixel_values_videos"]

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                stacked_videos = self.resize(stacked_videos, size=size, resample=resample)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_center_crop:
                stacked_videos = self.center_crop(stacked_videos, crop_size)
            stacked_videos = self.rescale_and_normalize(
                stacked_videos,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
            )
            processed_videos_grouped[shape] = stacked_videos

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        frame_counts = torch.tensor([video.shape[0] for video in processed_videos], dtype=torch.long)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        return BatchFeature(
            data={
                "pixel_values_videos": pixel_values_videos,
                "video_frame_counts": frame_counts,
            },
            tensor_type=return_tensors,
        )


__all__ = ["CogVLM2VideoProcessor"]
