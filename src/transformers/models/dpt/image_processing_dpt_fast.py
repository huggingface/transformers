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
"""Fast Image processor class for DPT."""

from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling
from ...utils import add_start_docstrings


@add_start_docstrings(
    "Constructs a fast DPT image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class DPTImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    default_to_square = None
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None


__all__ = ["DPTImageProcessorFast"]
