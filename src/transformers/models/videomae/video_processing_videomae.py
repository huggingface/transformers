# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""Video processor class for VideoMAE."""

from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling
from ...video_processing_utils import BaseVideoProcessor


class VideoMAEVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = False  # Set to False for backward compatibility with image processor workflows.
    model_input_names = ["pixel_values"]

    def preprocess(self, videos, **kwargs):
        batch = super().preprocess(videos, **kwargs)
        batch["pixel_values"] = batch.pop("pixel_values_videos")
        return batch


__all__ = ["VideoMAEVideoProcessor"]
