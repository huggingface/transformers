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
"""Fast Image processor class for Pvt."""

from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    get_size_dict,
)
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from ...processing_utils import Unpack
from ...utils import add_start_docstrings


class PvtFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    pass


@add_start_docstrings(
    "Constructs a fast Pvt image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class PvtImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 224, "width": 224}
    default_to_square = None
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None
    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs: Unpack[PvtFastImageProcessorKwargs]) -> None:
        size = kwargs.pop("size", None)
        size = size if size is not None else {"height": 224, "width": 224}
        if isinstance(size, (tuple, list)):
            size = size[::-1]
        self.size = get_size_dict(size)
        super().__init__(**kwargs)


__all__ = ["PvtImageProcessorFast"]
