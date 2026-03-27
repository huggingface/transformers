# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Got-OCR-2."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
    infer_channel_dimension_format,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)
from ...utils.import_utils import requires
from .image_processing_got_ocr2 import GotOcr2ImageProcessorKwargs, get_optimal_tiled_canvas


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class GotOcr2ImageProcessorPil(PilBackend):
    valid_kwargs = GotOcr2ImageProcessorKwargs
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    crop_to_patches = False
    min_patches = 1
    max_patches = 12

    def __init__(self, **kwargs: Unpack[GotOcr2ImageProcessorKwargs]):
        super().__init__(**kwargs)

    def crop_image_to_patches(
        self,
        image: np.ndarray,
        min_patches: int,
        max_patches: int,
        use_thumbnail: bool = True,
        patch_size: SizeDict | None = None,
        resample: "PILImageResampling | int | None" = None,
    ):
        """
        Crop the image to patches and return a list of cropped images.
        The number of patches and their grid arrangement are determined by the original image size,
        the target patch size and the minimum and maximum number of patches.
        The aspect ratio of the patches grid is chosen to be the closest to the original image aspect ratio.

        Args:
            image (`np.ndarray`):
                The image to be cropped.
            min_patches (`int`):
                The minimum number of patches to be extracted from the image.
            max_patches (`int`):
                The maximum number of patches to be extracted from the image.
            use_thumbnail (`bool`, *optional*, defaults to `True`):
                Whether to add a thumbnail image to the list of cropped patches.
            patch_size (`SizeDict`, *optional*):
                The size of the output patches.
            resample (`PILImageResampling | int | None`, *optional*):
                Resampling filter to use when resizing.
        """
        # Ensure image is in CHW format for processing
        input_data_format = infer_channel_dimension_format(image)
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

        patch_size_height, patch_size_width = patch_size.height, patch_size.width
        original_height, original_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        # find the closest aspect ratio to the target
        num_columns, num_rows = get_optimal_tiled_canvas(
            (original_height, original_width), (patch_size_height, patch_size_width), min_patches, max_patches
        )

        # calculate the target width and height
        target_width = patch_size_width * num_columns
        target_height = patch_size_height * num_rows
        num_blocks = num_columns * num_rows

        # resize the image so that each patch is of patch_size
        resized_image = self.resize(image, SizeDict(height=target_height, width=target_width), resample=resample)
        # split the image into patches
        processed_images = []
        for i in range(num_blocks):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * patch_size_width,
                row * patch_size_height,
                (column + 1) * patch_size_width,
                (row + 1) * patch_size_height,
            )
            # split the image (images are CHW format)
            patch_image = resized_image[..., box[1] : box[3], box[0] : box[2]]
            # Convert back to original format
            patch_image = to_channel_dimension_format(patch_image, input_data_format, ChannelDimension.FIRST)
            processed_images.append(patch_image)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = self.resize(image, patch_size, resample=resample)
            thumbnail_img = to_channel_dimension_format(thumbnail_img, input_data_format, ChannelDimension.FIRST)
            processed_images.append(thumbnail_img)

        return processed_images

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        crop_to_patches: bool = False,
        min_patches: int = 1,
        max_patches: int = 12,
        **kwargs,
    ) -> BatchFeature:
        num_patches = []
        processed_images = []

        for image in images:
            if crop_to_patches and max_patches > 1:
                patches = self.crop_image_to_patches(
                    image,
                    min_patches,
                    max_patches,
                    patch_size=size,
                    resample=resample,
                )
                num_patches.append(len(patches))
                # Normalize and rescale patches
                for patch in patches:
                    if do_rescale:
                        patch = self.rescale(patch, rescale_factor)
                    if do_normalize:
                        patch = self.normalize(patch, image_mean, image_std)
                    processed_images.append(patch)
            else:
                num_patches.append(1)
                if do_resize:
                    image = self.resize(image, size, resample)
                if do_rescale:
                    image = self.rescale(image, rescale_factor)
                if do_normalize:
                    image = self.normalize(image, image_mean, image_std)
                processed_images.append(image)

        return BatchFeature(
            data={"pixel_values": processed_images, "num_patches": num_patches}, tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns number patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of patches per image.
        """
        min_patches = images_kwargs.get("min_patches", self.min_patches) if images_kwargs else self.min_patches
        max_patches = images_kwargs.get("max_patches", self.max_patches) if images_kwargs else self.max_patches
        patch_size = images_kwargs.get("patch_size", self.size) if images_kwargs else self.size
        crop_to_patches = (
            images_kwargs.get("crop_to_patches", self.crop_to_patches) if images_kwargs else self.crop_to_patches
        )

        num_patches = 1
        if crop_to_patches and max_patches > 1:
            if isinstance(patch_size, dict):
                patch_height, patch_width = patch_size["height"], patch_size["width"]
            else:
                patch_height, patch_width = patch_size.height, patch_size.width
            num_columns, num_rows = get_optimal_tiled_canvas(
                (height, width), (patch_height, patch_width), min_patches, max_patches
            )
            if num_columns * num_rows > 1:
                num_patches += num_columns * num_rows

        return num_patches


__all__ = ["GotOcr2ImageProcessorPil"]
