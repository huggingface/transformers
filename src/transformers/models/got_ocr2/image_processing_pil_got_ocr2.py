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

from functools import lru_cache

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
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)


# Adapted from transformers.models.got_ocr2.image_processing_got_ocr2.GotOcr2ImageProcessorKwargs
class GotOcr2ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    crop_to_patches (`bool`, *optional*, defaults to `self.crop_to_patches`):
        Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the
        `preprocess` method.
    min_patches (`int`, *optional*, defaults to `self.min_patches`):
        The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
        set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.
    max_patches (`int`, *optional*, defaults to `self.max_patches`):
        The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
        set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.
    """

    crop_to_patches: bool
    min_patches: int
    max_patches: int


# Adapted from transformers.models.got_ocr2.image_processing_got_ocr2.get_all_supported_aspect_ratios
@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(min_image_tiles: int, max_image_tiles: int) -> list[tuple[int, int]]:
    """
    Computes all allowed aspect ratios for a given minimum and maximum number of input tiles.

    This function calculates all possible arrangements of tiles that can be formed
    within the constraint of the minimum and maximum number of tiles. Each arrangement is
    represented by its aspect ratio (width/height) and the corresponding tile configuration.

    Args:
        min_image_tiles (`int`):
            The minimum number of tiles allowed.
        max_image_tiles (`int`):
            The maximum number of tiles allowed.

    Returns:
        `list[tuple[int, int]]`: A list of tuples, each tuple representing a valid (width, height)
        configuration in terms of number of tiles.

    Example:
        >>> get_all_supported_aspect_ratios(1, 4)
        [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (2, 2), (4, 1)]

    """
    aspect_ratios = []
    for width in range(1, max_image_tiles + 1):
        for height in range(1, max_image_tiles + 1):
            if width * height <= max_image_tiles and width * height >= min_image_tiles:
                aspect_ratios.append((width, height))

    aspect_ratios = sorted(aspect_ratios, key=lambda x: x[0] * x[1])

    return aspect_ratios


# Adapted from transformers.models.got_ocr2.image_processing_got_ocr2.get_optimal_tiled_canvas
@lru_cache(maxsize=100)
def get_optimal_tiled_canvas(
    original_image_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    min_image_tiles: int,
    max_image_tiles: int,
) -> tuple[int, int]:
    """
    Given a minimum and maximum number of tiles, find the canvas with the closest aspect ratio to the
    original image aspect ratio.
    In case of tie-breaking condition when two canvases have the same aspect ratio difference, we favor the canvas with
    more tiles, until the area covered by the tiles is more than twice the target area, in order to avoid unnecessarily
    excessive tiling.
    """
    possible_tile_arrangements = get_all_supported_aspect_ratios(min_image_tiles, max_image_tiles)

    original_height, original_width = original_image_size
    target_tile_height, target_tile_width = target_tile_size
    aspect_ratio = original_width / original_height
    area = original_width * original_height

    # find the grid with the best aspect ratio
    best_ratio_diff = float("inf")
    best_grid = (1, 1)
    for grid in possible_tile_arrangements:
        grid_aspect_ratio = grid[0] / grid[1]
        ratio_diff = abs(aspect_ratio - grid_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_grid = grid
        elif ratio_diff == best_ratio_diff:
            # if the aspect ratio difference is the same, we favor the grid with more patches
            # until the area covered by the patches is more than twice the original image area
            if area > 0.5 * target_tile_height * target_tile_width * grid[0] * grid[1]:
                best_grid = grid

    return best_grid


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
        resample: "PILImageResampling | None" = None,
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
        resample: "PILImageResampling | None",
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
