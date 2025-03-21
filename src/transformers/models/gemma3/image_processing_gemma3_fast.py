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
from functools import partial
from typing import List, Optional, Union

from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    SizeDict,
    get_image_size,
    make_nested_list_of_images,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    add_start_docstrings,
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
    do_pan_and_scan: Optional[bool]
    pan_and_scan_min_crop_size: Optional[int]
    pan_and_scan_max_num_crops: Optional[int]
    pan_and_scan_min_ratio_to_activate: Optional[float]


@add_start_docstrings(
    "Constructs a fast ConvNeXT image processor. Based on [`SiglipImageProcessor`] with incorporation of Pan adn Scan cropping method.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        do_pan_and_scan (`bool`, *optional*):
            Whether to apply `pan_and_scan` to images.
        pan_and_scan_min_crop_size (`int`, *optional*):
            Minimum size of each crop in pan and scan.
        pan_and_scan_max_num_crops (`int`, *optional*):
            Maximum number of crops per image in pan and scan.
        pan_and_scan_min_ratio_to_activate (`float`, *optional*):
            Minimum aspect ratio to activate pan and scan.
    """,
)
class Gemma3ImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
    default_to_square = True
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

    def _prepare_images_structure(
        self,
        images: ImageInput,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        return make_nested_list_of_images(images)

    def _prepare_input_images(
        self,
        images: ImageInput,
        do_convert_rgb: bool = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> List["torch.Tensor"]:
        """
        Prepare the input images for processing.
        """
        batch_images = self._prepare_images_structure(images)
        process_image_fn = partial(
            self._process_image,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )
        # todo: yoni - check if we can parallelize this efficiently
        batch_processed_images = []
        for image_list in batch_images:
            processed_images = []
            for image in image_list:
                processed_images.append(process_image_fn(image))
            batch_processed_images.append(processed_images)

        return batch_processed_images

    def pan_and_scan(
        self,
        image: "torch.Tensor",
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
    ):
        """
        Pan and Scan an image, by cropping into smaller images when the aspect ratio exceeds
        minumum allowed ratio.

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
        height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)

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
            image[:, pos_h : pos_h + crop_size_h, pos_w : pos_w + crop_size_w]
            for pos_h, pos_w in itertools.product(crop_positions_h, crop_positions_w)
        ]

    def _process_images_for_pan_and_scan(
        self,
        images: List["torch.Tensor"],
        do_pan_and_scan: bool,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
    ):
        pas_images_list = []
        num_crops = []
        for image in images:
            pas_images = self.pan_and_scan(
                image=image,
                pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
                pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
                pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
            )
            pas_images_list.extend([image] + pas_images)
            num_crops.append(len(pas_images))
        return pas_images_list, num_crops

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
            do_pan_and_scan (`bool`, *optional*):
                Whether to apply `pan_and_scan` to images.
            pan_and_scan_min_crop_size (`int`, *optional*):
                Minimum size of each crop in pan and scan.
            pan_and_scan_max_num_crops (`int`, *optional*):
                Maximum number of crops per image in pan and scan.
            pan_and_scan_min_ratio_to_activate (`float`, *optional*):
                Minimum aspect ratio to activate pan and scan.
        """,
    )
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Gemma3FastImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: List[List["torch.Tensor"]],
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
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:
        processed_images = []
        batch_num_crops = []

        for images_list in images:
            if do_pan_and_scan:
                images_list, num_crops = self._process_images_for_pan_and_scan(
                    images=images_list,
                    do_pan_and_scan=do_pan_and_scan,
                    pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
                    pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
                    pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
                )
            else:
                num_crops = [[0] for _ in images_list]

            # Group images by size for batched processing
            processed_image_patches_grouped = {}
            grouped_image_patches, grouped_image_patches_index = group_images_by_shape(images_list)
            for shape, stacked_image_patches in grouped_image_patches.items():
                if do_resize:
                    stacked_image_patches = self.resize(
                        image=stacked_image_patches,
                        size=size,
                        interpolation=interpolation,
                    )
                # Fused rescale and normalize
                stacked_image_patches = self.rescale_and_normalize(
                    stacked_image_patches, do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
                processed_image_patches_grouped[shape] = stacked_image_patches
            processed_image_patches = reorder_images(processed_image_patches_grouped, grouped_image_patches_index)
            processed_image_patches = (
                torch.stack(processed_image_patches, dim=0) if return_tensors else processed_image_patches
            )
            processed_images.extend(processed_image_patches)
            batch_num_crops.extend(num_crops)

        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        return BatchFeature(
            data={"pixel_values": processed_images, "num_crops": batch_num_crops}, tensor_type=return_tensors
        )


__all__ = ["Gemma3ImageProcessorFast"]
