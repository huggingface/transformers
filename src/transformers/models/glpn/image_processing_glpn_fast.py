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
)
from ...utils import (
    TensorType,
    auto_docstring,
    requires_backends,
)


@auto_docstring
class GLPNImageProcessorFast(BaseImageProcessorFast):
    """
    Fast image processor for GLPN using the Torch/TorchVision backend.

    Performs:
    - Crop H,W down to the nearest multiple of `size_divisor` (default 32)
    - Rescale [0,255] → [0,1]
    - (No normalization by default)
    """

    # Persist ONLY the same keys as the slow processor
    do_resize = True
    do_rescale = True
    do_normalize = False
    resample = PILImageResampling.BILINEAR
    size_divisor = 32
    # Don't persist an explicit `size` for GLPN (slow doesn't)
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 480, "width": 640}  # only for validation; we still crop, not resize
    interpolation = F.InterpolationMode.BILINEAR
    # valid_kwargs = GLPNImageProcessorKwargs

    # If BaseImageProcessorFast supports it, this makes persistence explicit:
    try:
        config_keys = {"do_resize", "size_divisor", "resample", "do_rescale"}
    except Exception:
        pass

    def __init__(self, **kwargs) -> None:
        if "ensure_multiple_of" in kwargs and "size_divisor" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["size_divisor"] = kwargs.pop("ensure_multiple_of")
        # ensure resample default for validation
        kwargs.setdefault("resample", PILImageResampling.BILINEAR)
        super().__init__(**kwargs)

    @staticmethod
    def _crop_to_multiple(
        images: torch.Tensor,
        size_divisor: int = 32,
    ) -> torch.Tensor:
        """
        Crop images (B,C,H,W) by flooring H and W to nearest multiple of `size_divisor`.
        No resampling; purely geometric crop to match slow GLPN behavior.
        """
        _, _, h, w = images.shape
        new_h = (h // size_divisor) * size_divisor
        new_w = (w // size_divisor) * size_divisor
        if (new_h, new_w) == (h, w):
            return images
        # Use top-left crop to mirror typical behavior; slow doesn't center-crop.
        return images[..., :new_h, :new_w]

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
        if size is None:
            size = {"height": 480, "width": 640}
        if resample is None and interpolation is None:
            resample = self.resample

        grouped_images, grouped_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_groups = {}
        sd = size_divisor if size_divisor is not None else self.size_divisor

        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self._crop_to_multiple(stacked_images, sd)
            if do_rescale:
                stacked_images = self.rescale(stacked_images, rescale_factor)
            if do_normalize:
                stacked_images = self.normalize(stacked_images, image_mean, image_std)
            processed_groups[shape] = stacked_images

        reordered = reorder_images(processed_groups, grouped_index)

        if return_tensors:
            # Detect heterogeneous shapes
            shapes = {tuple(img.shape) for img in reordered}
            if len(shapes) == 1:
                # All images same shape -> safe to stack
                processed = torch.stack(reordered, dim=0)
                tensor_type = return_tensors
            else:
                # Keep as list of tensors - can't stack due to heterogeneous shapes
                processed = reordered  # Already torch tensors, keep them that way
                tensor_type = None  # Signal BatchFeature not to try converting
        else:
            processed = reordered
            tensor_type = None

        return BatchFeature(data={"pixel_values": processed}, tensor_type=tensor_type)

    # ensure only slow keys are serialized
    def to_dict(self):
        d = super().to_dict()

        # Keep only these keys with their values (everything else gets set to None)
        keys_to_keep = {
            "image_processor_type",
            "_processor_class",  # Identity metadata
            "do_resize",
            "size_divisor",
            "resample",
            "do_rescale",  # Core GLPN params
            "default_to_square",
            "data_format",  # Fast processor params
        }

        # Set all other keys to None (don't persist their values)
        for key in list(d.keys()):
            if key not in keys_to_keep:
                d[key] = None

        return d

    @torch.no_grad()
    def post_process_depth_estimation(self, outputs, target_sizes=None):
        """
        Convert raw model outputs to final depth predictions.
        Mirrors slow GLPN: PyTorch interpolate w/ bicubic, align_corners=False.
        """
        requires_backends(self, "torch")
        predicted_depth = outputs.predicted_depth  # shape: (B, H, W) or (B, 1, H, W)

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
