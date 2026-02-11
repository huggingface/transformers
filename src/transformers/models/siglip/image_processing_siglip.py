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
"""Image processor class for SigLIP."""

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring


@auto_docstring(custom_intro="Constructs a SigLIP image processor.")
class SiglipImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
    default_to_square = False
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)


__all__ = ["SiglipImageProcessor"]
