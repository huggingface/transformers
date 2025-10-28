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
"""Fast Image processor class for GLPN."""

from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
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
    """
    Fast image processor for GLPN using the Torch/TorchVision backend.

    Performs:
    - Crop H,W down to the nearest multiple of `size_divisor` (default 32)
    - Rescale [0,255] → [0,1]
    - (No normalization by default)
    """

    do_resize = True
    do_rescale = True
    do_normalize = False
    resample = PILImageResampling.BILINEAR
    size_divisor = 32
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    interpolation = F.InterpolationMode.BILINEAR
    valid_kwargs = GLPNImageProcessorKwargs

    def __init__(self, **kwargs) -> None:
        if "ensure_multiple_of" in kwargs and "size_divisor" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["size_divisor"] = kwargs.pop("ensure_multiple_of")
        # ensure resample default for validation
        kwargs.setdefault("resample", PILImageResampling.BILINEAR)
        kwargs.setdefault("size", {"height": 480, "width": 640})
        super().__init__(**kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: Optional[dict] = None,
        size_divisor: Optional[int] = None,
        interpolation: Optional["F.InterpolationMode"] = None,
        do_rescale: bool = True,
        rescale_factor: Optional[float] = 1 / 255,
        do_normalize: bool = False,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        disable_grouping: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        resample: Optional[PILImageResampling] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        GLPN fast preprocessing:
        - crop to floored multiple of size_divisor
        - rescale [0,1]
        - normalize (off by default)
        """
        # avoid validation error: inject dummy size/resample for validate_preprocess_arguments

        if resample is None and interpolation is None:
            resample = self.resample

        grouped_images, grouped_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_groups = {}
        sd = size_divisor if size_divisor is not None else self.size_divisor

        for shape, stacked_images in grouped_images.items():
            if do_resize:
                # Calculate target size (nearest multiple of size_divisor)
                _, _, h, w = stacked_images.shape
                new_h = (h // sd) * sd
                new_w = (w // sd) * sd

                if (new_h, new_w) != (h, w):
                    target_size = SizeDict(height=new_h, width=new_w)
                    stacked_images = self.resize(
                        stacked_images, size=target_size, interpolation=interpolation, antialias=True
                    )
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_groups[shape] = stacked_images

        reordered = reorder_images(processed_groups, grouped_index)

        # Pad to max size if there are heterogeneous shapes
        shapes = {tuple(img.shape) for img in reordered}
        if len(shapes) > 1:
            reordered = self.pad(reordered, pad_size=None)

        processed = torch.stack(reordered, dim=0) if return_tensors else reordered

        return BatchFeature(data={"pixel_values": processed}, tensor_type=return_tensors)

    # ensure only slow keys are serialized
    def to_dict(self):
        output_dict = super().to_dict()

        keys_to_keep = {
            "image_processor_type",
            "_processor_class",
            "do_resize",
            "size_divisor",
            "resample",
            "do_rescale",
            "default_to_square",
            "data_format",
        }

        for key in list(output_dict.keys()):
            if key not in keys_to_keep:
                output_dict[key] = None

        return output_dict

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
