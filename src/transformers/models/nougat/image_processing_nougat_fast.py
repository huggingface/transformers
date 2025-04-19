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
"""Fast Image processor class for Nougat."""
from typing import Optional, Dict

from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...processing_utils import Unpack
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    PILImageResampling,
)
from ...utils import (
    add_start_docstrings,
    is_torch_available,
    is_vision_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
)

if is_torch_available():
    import torch

if is_vision_available():
    from ...image_utils import PILImageResampling

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

if is_torchvision_available():
    from ...image_utils import pil_torch_interpolation_mapping


logger = logging.get_logger(__name__)


class NougatFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_crop_margin: Optional[bool]
    do_thumbnail: Optional[bool]
    do_align_long_axis: Optional[bool]


@add_start_docstrings(
    "Constructs a fast Nougat image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class NougatImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 896, "width": 672}
    default_to_square = None
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None
    do_crop_margin = None
    do_thumbnail = None
    do_align_long_axis = None
    valid_kwargs = NougatFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[NougatFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def python_find_non_zero(self, image: torch.Tensor) -> torch.Tensor:
        non_zero_indices = torch.nonzero(image, as_tuple=False)
        idxvec = non_zero_indices[:, [1, 0]]
        idxvec = idxvec.view(-1, 1, 2)
        return idxvec

    def python_bounding_rect(self, coordinates):
        min_values = torch.min(coordinates, axis=(0, 1)).astype(int)
        max_values = torch, max(coordinates, axis=(0, 1)).astype(int)
        x_min, y_min = min_values[0], min_values[1]
        width = max_values[0] - x_min + 1
        height = max_values[1] - y_min + 1
        return x_min, y_min, width, height

    def crop_margin(
        self,
        image: torch.Tensor,
        gray_threshold: int = 200,
    ) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Expected image to be in CHW format with 3 channels")

        grayscale = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]

        if grayscale.max() <= 1.0:
            grayscale = grayscale * 255.0

        mask = grayscale >= gray_threshold

        if not mask.any():
            return image

        coords = self.python_find_non_zero(mask)
        y_min, x_min = coords.min(dim=0).values
        y_max, x_max = coords.max(dim=0).values

        cropped = image[:, y_min : y_max + 1, x_min : x_max + 1]
        return cropped

    def align_long_axis(
        self,
        image: torch.Tensor,
        size: Dict[str, int],
    ) -> torch.Tensor:
        input_height, input_width = image.shape[-2], image.shape[-1]
        output_height, output_width = size["height"], size["width"]

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = torch.rot90(image, k=1, dims=[1, 2])

        return image

    def thumbnail(
        self,
        image: torch.Tensor,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        **kwargs,
    ) -> torch.Tensor:
        input_height, input_width = image.shape[-2], image.shape[-1]
        output_height, output_width = size["height"], size["width"]

        height = min(input_height, output_height)
        width = min(input_width, output_width)

        if height == input_height and width == input_width:
            return image

        if input_height > input_width:
            width = int(input_width * height / input_height)
        elif input_width > input_height:
            height = int(input_height * width / input_width)

        interpolation = pil_torch_interpolation_mapping(resample)
        size = {
            "height": height,
            "width": width,
        }

        return super().resize(
            image,
            size=size,
            interpolation=interpolation,
        )


__all__ = ["NougatImageProcessorFast"]
