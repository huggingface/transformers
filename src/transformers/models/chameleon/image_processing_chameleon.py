# coding=utf-8
# Copyright 2024 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ImageInput
from ...utils import TensorType, is_vision_available


if is_vision_available():
    import PIL


# TODO: support batch? Rewrite code, use existing functionality to resize, ...
class ChameleonImageProcessor(BaseImageProcessor):
    r"""
    Constructs an Chameleon image processor.

    Args:
        kwargs (Dict, *optional*):
            foo bar
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        images = self._whiten_transparency(images)
        images = self._vqgan_input_from(images)
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
        # Check if it's already in RGB format.
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")

        # There is a transparency layer, blend it with a white background.

        # Calculate the alpha proportion for blending.
        alpha = vals_rgba[:, :, 3] / 255.0
        # Blend with white background.
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def _vqgan_input_from(self, img: PIL.Image, target_image_size=512) -> np.array:
        # Resize with aspect ratio preservation.
        s = min(img.size)
        scale = target_image_size / s
        new_size = (round(scale * img.size[0]), round(scale * img.size[1]))
        img = img.resize(new_size, PIL.Image.LANCZOS)

        # Center crop.
        x0 = (img.width - target_image_size) // 2
        y0 = (img.height - target_image_size) // 2
        img = img.crop((x0, y0, x0 + target_image_size, y0 + target_image_size))

        # Convert to tensor.
        np_img = np.array(img) / 255.0  # Normalize to [0, 1]
        np_img = np_img * 2 - 1  # Scale to [-1, 1]
        np_img = np.transpose(np_img, (2, 0, 1))  # CHW format.
        return np_img
