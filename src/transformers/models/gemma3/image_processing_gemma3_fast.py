# coding=utf-8
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
"""Fast Image processor class for SigLIP."""

import itertools
import math
from typing import Optional, Union

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ImageInput, SizeDict
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)


if is_vision_available():
    from ...image_utils import PILImageResampling

if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

logger = logging.get_logger(__name__)


class Gemma3FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    do_pan_and_scan (`bool`, *optional*):
        Whether to apply `pan_and_scan` to images.
    pan_and_scan_min_crop_size (`int`, *optional*):
        Minimum size of each crop in pan and scan.
    pan_and_scan_max_num_crops (`int`, *optional*):
        Maximum number of crops per image in pan and scan.
    pan_and_scan_min_ratio_to_activate (`float`, *optional*):
        Minimum aspect ratio to activate pan and scan.
    """

    do_pan_and_scan: Optional[bool]
    pan_and_scan_min_crop_size: Optional[int]
    pan_and_scan_max_num_crops: Optional[int]
    pan_and_scan_min_ratio_to_activate: Optional[float]


@auto_docstring
class Gemma3ImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pan_and_scan = None
    pan_and_scan_min_crop_size = None
    pan_and_scan_max_num_crops = None
    pan_and_scan_min_ratio_to_activate = None
    valid_kwargs = Gemma3FastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[Gemma3FastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def pan_and_scan_batched(
        self,
        images: "torch.Tensor",
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
    ):
        """
        Pan and Scan an image, by cropping into smaller images when the aspect ratio exceeds
        minimum allowed ratio.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            pan_and_scan_min_crop_size (`int`, *optional*):
                Minimum size of each crop in pan and scan.
            pan_and_scan_max_num_crops (`int`, *optional*):
                Maximum number of crops per image in pan and scan.
            pan_and_scan_min_ratio_to_activate (`float`, *optional*):
                Minimum aspect ratio to activate pan and scan.
        """
        height, width = images.shape[-2:]

        # Square or landscape image.
        if width >= height:
            # Only apply PaS if the image is sufficiently exaggerated
            if width / height < pan_and_scan_min_ratio_to_activate:
                return []

            # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
            num_crops_w = int(math.floor(width / height + 0.5))  # Half round up rounding.
            num_crops_w = min(int(math.floor(width / pan_and_scan_min_crop_size)), num_crops_w)

            # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
            num_crops_w = max(2, num_crops_w)
            num_crops_w = min(pan_and_scan_max_num_crops, num_crops_w)
            num_crops_h = 1

        # Portrait image.
        else:
            # Only apply PaS if the image is sufficiently exaggerated
            if height / width < pan_and_scan_min_ratio_to_activate:
                return []

            # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
            num_crops_h = int(math.floor(height / width + 0.5))
            num_crops_h = min(int(math.floor(height / pan_and_scan_min_crop_size)), num_crops_h)

            # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
            num_crops_h = max(2, num_crops_h)
            num_crops_h = min(pan_and_scan_max_num_crops, num_crops_h)
            num_crops_w = 1

        crop_size_w = int(math.ceil(width / num_crops_w))
        crop_size_h = int(math.ceil(height / num_crops_h))

        # Don't apply PaS if crop size is too small.
        if min(crop_size_w, crop_size_h) < pan_and_scan_min_crop_size:
            return []

        crop_positions_w = [crop_size_w * i for i in range(num_crops_w)]
        crop_positions_h = [crop_size_h * i for i in range(num_crops_h)]

        return [
            images[..., pos_h : pos_h + crop_size_h, pos_w : pos_w + crop_size_w]
            for pos_h, pos_w in itertools.product(crop_positions_h, crop_positions_w)
        ]

    def _process_images_for_pan_and_scan(
        self,
        images: list["torch.Tensor"],
        do_pan_and_scan: bool,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
    ):
        pas_images = self.pan_and_scan_batched(
            images=images,
            pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
            pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
            pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
        )
        num_crops = [len(pas_images) for _ in images]
        return pas_images, num_crops

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Gemma3FastImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[list["torch.Tensor"]],
        do_resize: bool,
        size: SizeDict,
        do_pan_and_scan: Optional[bool],
        pan_and_scan_min_crop_size: Optional[int],
        pan_and_scan_max_num_crops: Optional[int],
        pan_and_scan_min_ratio_to_activate: Optional[float],
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:
        # Group images by size for batched processing
        processed_images_grouped = {}
        num_crops_grouped = {}
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        for shape_images, stacked_images in grouped_images.items():
            if do_pan_and_scan:
                pas_images, num_crops = self._process_images_for_pan_and_scan(
                    images=stacked_images,
                    do_pan_and_scan=do_pan_and_scan,
                    pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
                    pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
                    pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
                )
                # Add the thumbnails to the image patches
                stacked_images = [stacked_images] + pas_images
                # Group images by size for batched resizing (this will typically group thumbnails together and cropped patches together)
                processed_image_patches_grouped = {}
                grouped_image_patches, grouped_image_patches_index = group_images_by_shape(
                    stacked_images, disable_grouping=disable_grouping
                )
                for shape, stacked_image_patches in grouped_image_patches.items():
                    stacked_image_patches = self.resize(
                        image=stacked_image_patches,
                        size=size,
                        interpolation=interpolation,
                    )
                    processed_image_patches_grouped[shape] = stacked_image_patches
                processed_image_patches = reorder_images(processed_image_patches_grouped, grouped_image_patches_index)
                # Transpose to have the thumbnails with their corresponding patches
                stacked_images = torch.stack(processed_image_patches, dim=0).transpose(0, 1).contiguous()
            else:
                num_crops = [0 for _ in stacked_images]

                if do_resize:
                    stacked_images = self.resize(
                        image=stacked_images,
                        size=size,
                        interpolation=interpolation,
                    )
            num_crops_grouped[shape_images] = num_crops
            processed_images_grouped[shape_images] = stacked_images
        resized_images = reorder_images(processed_images_grouped, grouped_images_index)
        # If pan and scan is enabled, we need to flatten the list of images
        if do_pan_and_scan:
            resized_images = [image for images_list in resized_images for image in images_list]
        num_crops = reorder_images(num_crops_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(
            data={"pixel_values": processed_images, "num_crops": num_crops}, tensor_type=return_tensors
        )


__all__ = ["Gemma3ImageProcessorFast"]
