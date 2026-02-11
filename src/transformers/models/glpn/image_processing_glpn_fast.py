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
"""Fast Image processor class for GLPN."""

from typing import Optional

import torch
import torchvision.transforms.v2.functional as tvF

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, group_images_by_shape, reorder_images
from ...image_utils import (
    PILImageResampling,
    SizeDict,
)
from ...utils import (
    TensorType,
    auto_docstring,
    requires_backends,
)
from .image_processing_glpn import GLPNImageProcessorKwargs


@auto_docstring
class GLPNImageProcessorFast(BaseImageProcessorFast):
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    resample = PILImageResampling.BILINEAR
    size_divisor = 32
    valid_kwargs = GLPNImageProcessorKwargs

    def _validate_preprocess_kwargs(self, **kwargs):
        # pop `do_resize` to not raise an error as `size` is not None
        kwargs.pop("do_resize", None)
        return super()._validate_preprocess_kwargs(**kwargs)

    def resize(
        self,
        image: "torch.Tensor",
        size_divisor: int,
        interpolation: Optional["tvF.InterpolationMode"] = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            interpolation (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.
            antialias (`bool`, *optional*, defaults to `True`):
                Whether to use antialiasing.

        Returns:
            `torch.Tensor`: The resized image.
        """
        height, width = image.shape[-2:]
        # Rounds the height and width down to the closest multiple of size_divisor
        new_h = height // size_divisor * size_divisor
        new_w = width // size_divisor * size_divisor
        return super().resize(
            image, SizeDict(height=new_h, width=new_w), interpolation=interpolation, antialias=antialias
        )

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size_divisor: int | None = None,
        interpolation: Optional["tvF.InterpolationMode"] = None,
        do_rescale: bool = True,
        rescale_factor: float | None = 1 / 255,
        do_normalize: bool = False,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        disable_grouping: bool | None = None,
        return_tensors: str | TensorType | None = None,
        resample: PILImageResampling | None = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_groups = {}

        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(stacked_images, size_divisor=size_divisor, interpolation=interpolation)
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_groups[shape] = stacked_images

        processed_images = reorder_images(processed_groups, grouped_index)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def post_process_depth_estimation(self, outputs, target_sizes=None):
        """
        Convert raw model outputs to final depth predictions.
        Mirrors slow GLPN: PyTorch interpolate w/ bicubic, align_corners=False.
        """
        requires_backends(self, "torch")
        predicted_depth = outputs.predicted_depth

        results = []
        target_sizes = target_sizes or [None] * predicted_depth.shape[0]
        for depth, target_size in zip(predicted_depth, target_sizes):
            if target_size is not None:
                # Add batch and channel dimensions for interpolation
                depth_4d = depth[None, None, ...]
                resized = torch.nn.functional.interpolate(
                    depth_4d, size=target_size, mode="bicubic", align_corners=False
                )
                depth = resized.squeeze(0).squeeze(0)
            results.append({"predicted_depth": depth})

        return results


__all__ = ["GLPNImageProcessorFast"]
