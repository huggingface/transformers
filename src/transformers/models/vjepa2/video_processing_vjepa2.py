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
"""Fast Video processor class for VJEPA2."""

from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from ...processing_utils import Unpack, VideosKwargs
from ...video_processing_utils import BaseVideoProcessor


class VJEPA2VideoProcessorInitKwargs(VideosKwargs): ...


class VJEPA2VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": int(256 * 256 / 224)}
    crop_size = 256
    do_resize = True
    do_rescale = True
    do_center_crop = True
    do_normalize = True
    valid_kwargs = VJEPA2VideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs: Unpack[VJEPA2VideoProcessorInitKwargs]):
        crop_size = kwargs.get("crop_size", 256)
        if not isinstance(crop_size, int):
            if not isinstance(crop_size, dict) or "height" not in crop_size:
                raise ValueError("crop_size must be an integer or a dictionary with a 'height' key")
            crop_size = crop_size["height"]
        resize_size = int(crop_size * 256 / 224)
        kwargs["size"] = {"shortest_edge": resize_size}
        super().__init__(**kwargs)


__all__ = ["VJEPA2VideoProcessor"]
