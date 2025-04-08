# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for CLIP."""

from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...utils import add_start_docstrings


@add_start_docstrings(
    "Constructs a fast CLIP image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class CLIPImageProcessorFast(BaseImageProcessorFast):
    # To be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True


__all__ = ["CLIPImageProcessorFast"]
