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
"""Image processor class for Chameleon."""

import numpy as np
import PIL.Image

from ...image_processing_backends import PilBackend
from ...image_utils import (
    ImageInput,
    PILImageResampling,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring


@auto_docstring
class ChameleonImageProcessorPil(PilBackend):
    """PIL backend for Chameleon with custom convert_to_rgb."""

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

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)

    def convert_to_rgb(self, image: ImageInput) -> ImageInput:
        """
        Convert image to RGB by blending the transparency layer if it's in RGBA format.
        If image is not `PIL.Image`, it is simply returned without modifications.
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
        alpha = img_rgba[:, :, 3] / 255.0
        img_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * img_rgba[:, :, :3]
        return PIL.Image.fromarray(img_rgb.astype("uint8"), "RGB")


__all__ = ["ChameleonImageProcessorPil"]
