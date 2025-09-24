# coding=utf-8
# Copyright 2025 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
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
"""Video processor class for PerceptionLM."""

from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling
from ...processing_utils import Unpack, VideosKwargs
from ...video_processing_utils import BaseVideoProcessor


class PerceptionLMFastVideoProcessorInitKwargs(VideosKwargs): ...


class PerceptionLMVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 448, "width": 448}
    do_resize = True
    do_center_crop = False
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    valid_kwargs = PerceptionLMFastVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs: Unpack[PerceptionLMFastVideoProcessorInitKwargs]):
        super().__init__(**kwargs)


__all__ = ["PerceptionLMVideoProcessor"]
