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
"""Fast Image processor class for SAM."""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    PILImageResampling,
)
from ...utils import auto_docstring


class SamFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    SAM-specific kwargs for fast image processor.

    do_pad (`bool`, *optional*, defaults to `self.do_pad`):
        Whether to pad the image to the specified `pad_size`.
    pad_size (`Dict[str, int]`, *optional*, defaults to `self.pad_size`):
        Size of the output image after padding.
    mask_size (`Dict[str, int]`, *optional*, defaults to `self.mask_size`):
        Controls the size of the segmentation map after resize.
    mask_pad_size (`Dict[str, int]`, *optional*, defaults to `self.mask_pad_size`):
        Controls the size of the padding applied to the segmentation map.
    """

    do_pad: Optional[bool]
    pad_size: Optional[Dict[str, int]]
    mask_size: Optional[Dict[str, int]]
    mask_pad_size: Optional[Dict[str, int]]


@auto_docstring()
class SamImageProcessorFast(BaseImageProcessorFast):
    # Correct SAM defaults
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"longest_edge": 1024}
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_pad = True
    pad_size = {"height": 1024, "width": 1024}
    do_convert_rgb = True
    # SAM-specific attributes that slow processor has
    mask_size = {"longest_edge": 256}
    mask_pad_size = {"height": 256, "width": 256}
    valid_kwargs = SamFastImageProcessorKwargs

    def resize(self, image, size, **kwargs):
        """Essential: BaseImageProcessorFast doesn't support longest_edge."""
        if hasattr(size, "longest_edge") and size.longest_edge:
            h, w = image.shape[-2:]
            scale = size.longest_edge / max(h, w)
            from ...image_utils import SizeDict

            size = SizeDict(height=int(h * scale + 0.5), width=int(w * scale + 0.5))
        return super().resize(image, size, **kwargs)

    def _preprocess(self, images, **kwargs):
        """Essential: Add padding capability."""
        if not kwargs.get("do_pad", self.do_pad):
            return super()._preprocess(images, **kwargs)

        kwargs_copy = kwargs.copy()
        kwargs_copy["return_tensors"] = None
        result = super()._preprocess(images, **kwargs_copy)

        pad_size = kwargs.get("pad_size", self.pad_size)
        target_h, target_w = pad_size["height"], pad_size["width"]
        result["pixel_values"] = [
            F.pad(
                img,
                (
                    0,
                    max(0, target_w - img.shape[-1]),
                    0,
                    max(0, target_h - img.shape[-2]),
                ),
            )
            for img in result["pixel_values"]
        ]

        if kwargs.get("return_tensors"):
            result["pixel_values"] = torch.stack(result["pixel_values"])
        return result

    def preprocess(self, images, **kwargs):
        """Essential: Add SAM-required metadata."""
        from ...image_utils import get_image_size

        if not isinstance(images, list):
            images = [images]

        original_sizes = []
        for img in images:
            if hasattr(img, "size") and hasattr(img.size, "__getitem__"):  # PIL Image
                original_sizes.append([img.size[1], img.size[0]])  # (height, width)
            else:  # Tensor or numpy array
                try:
                    h, w = get_image_size(img)
                    original_sizes.append([h, w])
                except Exception:
                    original_sizes.append([224, 224])  # Safe fallback

        result = super().preprocess(images, **kwargs)

        # Calculate reshaped sizes
        size = kwargs.get("size", self.size)
        if "longest_edge" in size:
            longest_edge = size["longest_edge"]
            reshaped_sizes = [
                [
                    int(h * longest_edge / max(h, w) + 0.5),
                    int(w * longest_edge / max(h, w) + 0.5),
                ]
                for h, w in original_sizes
            ]
        else:
            reshaped_sizes = original_sizes

        result["original_sizes"] = original_sizes
        result["reshaped_input_sizes"] = reshaped_sizes
        return result


__all__ = ["SamImageProcessorFast"]
