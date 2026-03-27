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
"""Image processor class for Gemma3."""

import itertools
import math

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)


# Adapted from transformers.models.gemma3.image_processing_gemma3.Gemma3ImageProcessorKwargs
class Gemma3ImageProcessorKwargs(ImagesKwargs, total=False):
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

    do_pan_and_scan: bool
    pan_and_scan_min_crop_size: int
    pan_and_scan_max_num_crops: int
    pan_and_scan_min_ratio_to_activate: float


@auto_docstring
class Gemma3ImageProcessorPil(PilBackend):
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
    valid_kwargs = Gemma3ImageProcessorKwargs
    model_input_names = ["pixel_values", "num_crops"]

    def __init__(self, **kwargs: Unpack[Gemma3ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Gemma3ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def pan_and_scan(
        self,
        image: np.ndarray,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
    ):
        """
        Pan and Scan an image, by cropping into smaller images when the aspect ratio exceeds
        minimum allowed ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            pan_and_scan_min_crop_size (`int`, *optional*):
                Minimum size of each crop in pan and scan.
            pan_and_scan_max_num_crops (`int`, *optional*):
                Maximum number of crops per image in pan and scan.
            pan_and_scan_min_ratio_to_activate (`float`, *optional*):
                Minimum aspect ratio to activate pan and scan.
        """
        height, width = get_image_size(image, channel_dim="channels_first")

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

        # Images are channels-first (CHW format)
        return [
            image[:, pos_h : pos_h + crop_size_h, pos_w : pos_w + crop_size_w]
            for pos_h, pos_w in itertools.product(crop_positions_h, crop_positions_w)
        ]

    def _process_images_for_pan_and_scan(
        self,
        images: list[np.ndarray],
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

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        do_pan_and_scan: bool | None = None,
        pan_and_scan_min_crop_size: int | None = None,
        pan_and_scan_max_num_crops: int | None = None,
        pan_and_scan_min_ratio_to_activate: float | None = None,
        **kwargs,
    ) -> BatchFeature:
        processed_images = []
        num_crops = []

        for image in images:
            if do_pan_and_scan:
                pas_images = self.pan_and_scan(
                    image=image,
                    pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
                    pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
                    pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
                )
                # Add the original image and its crops
                image_list = [image] + pas_images
                num_crops.append(len(pas_images))
            else:
                image_list = [image]
                num_crops.append(0)

            # Process each image (original + crops if pan_and_scan)
            processed_image_list = []
            for img in image_list:
                if do_resize:
                    img = self.resize(image=img, size=size, resample=resample)
                if do_rescale:
                    img = self.rescale(image=img, scale=rescale_factor)
                if do_normalize:
                    img = self.normalize(image=img, mean=image_mean, std=image_std)
                processed_image_list.append(img)
            processed_images.extend(processed_image_list)

        return BatchFeature(
            data={"pixel_values": processed_images, "num_crops": num_crops}, tensor_type=return_tensors
        )


__all__ = ["Gemma3ImageProcessorPil"]
