# coding=utf-8
# Copyright 2025 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for Chameleon."""

from typing import Optional

import numpy as np
import PIL
import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring
class ChameleonImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.LANCZOS
    image_mean = [1.0, 1.0, 1.0]
    image_std = [1.0, 1.0, 1.0]
    size = {"shortest_edge": 512}
    default_to_square = False
    crop_size = {"height": 512, "width": 512}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    rescale_factor = 0.0078
    do_normalize = True
    do_convert_rgb = True

    def convert_to_rgb(self, image: ImageInput) -> ImageInput:
        """
        Convert image to RGB by blending the transparency layer if it's in RGBA format.
        If image is not `PIL.Image`, it si simply returned without modifications.

        Args:
            image (`ImageInput`):
                Image to convert.
        """

        if not isinstance(image, PIL.Image.Image):
            return image
        elif image.mode == "RGB":
            return image

        img_rgba = np.array(image.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (img_rgba[:, :, 3] < 255).any():
            return image.convert("RGB")

        # There is a transparency layer, blend it with a white background.
        # Calculate the alpha proportion for blending.
        alpha = img_rgba[:, :, 3] / 255.0
        img_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * img_rgba[:, :, :3]
        return PIL.Image.fromarray(img_rgb.astype("uint8"), "RGB")

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"] = None,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        if interpolation == F.InterpolationMode.LANCZOS:
            logger.warning_once(
                "You have used fast image processor with LANCZOS resample which not yet supported for torch.Tensor. "
                "BICUBIC resample will be used as an alternative. Please fall back to slow image processor if you "
                "want full consistency with the original model."
            )
            interpolation = F.InterpolationMode.BICUBIC

        return super().resize(
            image=image,
            size=size,
            interpolation=interpolation,
            **kwargs,
        )


__all__ = ["ChameleonImageProcessorFast"]
