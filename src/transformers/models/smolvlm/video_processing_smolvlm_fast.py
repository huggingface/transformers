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
"""Video processor class for Video-LLaVA."""

from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
)
from ...processing_utils import VideosKwargs
from ...video_processing_utils_fast import (
    BaseVideoProcessorFast,
)


class SmolVLMFastVideoProcessorInitKwargs(VideosKwargs): ...


class SmolVLMVideoProcessorFast(BaseVideoProcessorFast):
    resample = PILImageResampling.LANCZOS
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"longest_edge": 4 * 364}
    max_image_size = {"longest_edge": 364}
    default_to_square = False
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_image_splitting = True
    do_pad = True
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs):
        super().__init__(model_init_kwargs=SmolVLMFastVideoProcessorInitKwargs, **kwargs)


__all__ = ["SmolVLMVideoProcessorFast"]
