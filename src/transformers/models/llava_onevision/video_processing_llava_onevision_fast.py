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
"""Video processor class for LLaVa-Onevision."""

from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    PILImageResampling,
)
from ...processing_utils import VideosKwargs
from ...video_processing_utils_fast import (
    BaseVideoProcessorFast,
)


class LlavaOnevisionFastVideoProcessorInitKwargs(VideosKwargs): ...


class LlavaOnevisionVideoProcessorFast(BaseVideoProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    rescale_factor = 1 / 255
    default_to_square = False
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs):
        super().__init__(model_init_kwargs=LlavaOnevisionFastVideoProcessorInitKwargs, **kwargs)


__all__ = ["LlavaOnevisionVideoProcessorFast"]
