# coding=utf-8
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
"""Fast Image processor class for Got-OCR-2."""

from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, ImageInput, PILImageResampling, SizeDict
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
)
from .image_processing_got_ocr2 import get_optimal_tiled_canvas


if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


class GotOcr2FastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    crop_to_patches (`bool`, *optional*, defaults to `False`):
        Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the
        `preprocess` method.
    min_patches (`int`, *optional*, defaults to 1):
        The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
        set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.
    max_patches (`int`, *optional*, defaults to 12):
        The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
        set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.
    """

    crop_to_patches: Optional[bool]
    min_patches: Optional[int]
    max_patches: Optional[int]


@auto_docstring
class GotOcr2ImageProcessorFast(BaseImageProcessorFast):
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
    valid_kwargs = GotOcr2FastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[GotOcr2FastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[GotOcr2FastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def crop_image_to_patches(
        self,
        images: "torch.Tensor",
        min_patches: int,
        max_patches: int,
        use_thumbnail: bool = True,
        patch_size: Optional[Union[tuple, int, dict]] = None,
        interpolation: Optional["F.InterpolationMode"] = None,
    ):
        """
        Crop the images to patches and return a list of cropped images.
        The number of patches and their grid arrangement are determined by the original image size,
        the target patch size and the minimum and maximum number of patches.
        The aspect ratio of the patches grid is chosen to be the closest to the original image aspect ratio.

        Args:
            images (`torch.Tensor`):
                The images to be cropped.
            min_patches (`int`):
                The minimum number of patches to be extracted from the image.
            max_patches (`int`):
                The maximum number of patches to be extracted from the image.
            use_thumbnail (`bool`, *optional*, defaults to `True`):
                Whether to add a thumbnail image to the list of cropped patches.
            patch_size (`int`, `tuple[int, int]`, `dict`, *optional*):
                The size of the output patches.
                The format of the image data. If `None`, the format is inferred from the input image.

        Returns:
            list[`PIL.Image.Image`] or list[np.ndarray]: The list of cropped images.
        """
        patch_size_height, patch_size_width = patch_size.height, patch_size.width
        original_height, original_width = images.shape[-2:]
        # find the closest aspect ratio to the target
        num_columns, num_rows = get_optimal_tiled_canvas(
            (original_height, original_width), (patch_size_height, patch_size_width), min_patches, max_patches
        )

        # calculate the target width and height
        target_width = patch_size_width * num_columns
        target_height = patch_size_height * num_rows
        num_blocks = num_columns * num_rows

        # resize the image so that each patch is of patch_size
        resized_image = self.resize(
            images, SizeDict(height=target_height, width=target_width), interpolation=interpolation
        )
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
            # split the image
            patch_image = resized_image[..., box[1] : box[3], box[0] : box[2]]
            processed_images.append(patch_image)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = self.resize(images, patch_size, interpolation=interpolation)
            processed_images.append(thumbnail_img)

        processed_images = torch.stack(processed_images, dim=0).transpose(0, 1).contiguous()

        return processed_images

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        crop_to_patches: bool,
        min_patches: int,
        max_patches: int,
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
        if crop_to_patches:
            grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
            processed_images_grouped = {}
            num_patches = {}
            for shape, stacked_images in grouped_images.items():
                stacked_images = self.crop_image_to_patches(
                    stacked_images,
                    min_patches,
                    max_patches,
                    patch_size=size,
                    interpolation=interpolation,
                )
                processed_images_grouped[shape] = stacked_images
                num_patches[shape] = [stacked_images.shape[1]] * stacked_images.shape[0]
            images = reorder_images(processed_images_grouped, grouped_images_index)
            images = [image for images_list in images for image in images_list]
            num_patches = reorder_images(num_patches, grouped_images_index)
        else:
            num_patches = [1] * len(images)

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

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
        min_patches = images_kwargs.get("min_patches", self.min_patches)
        max_patches = images_kwargs.get("max_patches", self.max_patches)
        patch_size = images_kwargs.get("patch_size", self.size)
        crop_to_patches = images_kwargs.get("crop_to_patches", self.crop_to_patches)

        num_patches = 1
        if crop_to_patches and max_patches > 1:
            num_columns, num_rows = get_optimal_tiled_canvas(
                (height, width), (patch_size["height"], patch_size["width"]), min_patches, max_patches
            )
            if num_columns * num_rows > 1:
                num_patches += num_columns * num_rows

        return num_patches


__all__ = ["GotOcr2ImageProcessorFast"]
