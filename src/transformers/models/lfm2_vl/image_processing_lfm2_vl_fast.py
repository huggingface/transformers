# coding=utf-8
# Copyright 2025 the HuggingFace Inc. team. All rights reserved.
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
import math
from functools import lru_cache
from typing import Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import (
    Unpack,
)
from ...utils import (
    TensorType,
    auto_docstring,
    logging,
)


logger = logging.get_logger(__name__)


def round_by_factor(number: float, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Find the closest aspect ratio from target_ratios to match the input aspect ratio.

    Args:
        aspect_ratio: The aspect ratio to match (width/height).
        target_ratios: List of possible aspect ratios as tuples of (width, height) integers.
        width: Original image width in pixels.
        height: Original image height in pixels.
        image_size: Base size for calculating target area.

    Returns:
        tuple[int, int]: The best matching ratio as (width, height) integers.
    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        # update best ratio if we found a closer match
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        # if equally close, prefer the ratio that better matches the original image area
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio

    return best_ratio


# copied from Siglip2ImageProcessor
@lru_cache(maxsize=256)
def get_image_size_for_max_num_patches(
    image_height: int, image_width: int, patch_size: int, max_num_patches: int, eps: float = 1e-5
) -> tuple[int, int]:
    """
    Determine image size based on max number of patches, ensure dimensions are divisible by patch size and image is at least 1 patch.

    Args:
        image_height (`int`):
            Original image height.
        image_width (`int`):
            Original image width.
        patch_size (`int`):
            Patch size for processing.
        max_num_patches (`int`):
            Maximum number of patches.
        eps (`float`):
            Small threshold for binary search.

    Returns:
        Tuple: (target_height, target_width)
    """

    def get_scaled_image_size(scale: float, size: int, patch_size: int) -> int:
        scaled_size = size * scale
        scaled_size = math.ceil(scaled_size / patch_size) * patch_size  # make divisible by patch_size
        scaled_size = max(patch_size, scaled_size)  # ensure at least 1 patch
        return int(scaled_size)

    # Binary search for optimal scale
    scale_min, scale_max = eps / 10, 100.0
    while (scale_max - scale_min) >= eps:
        scale = (scale_min + scale_max) / 2
        target_height = get_scaled_image_size(scale, image_height, patch_size)
        target_width = get_scaled_image_size(scale, image_width, patch_size)
        num_patches = (target_height / patch_size) * (target_width / patch_size)

        if num_patches <= max_num_patches:
            scale_min = scale
        else:
            scale_max = scale

    scale = scale_min
    target_height = get_scaled_image_size(scale, image_height, patch_size)
    target_width = get_scaled_image_size(scale, image_width, patch_size)
    return target_height, target_width


def convert_image_to_patches(images: "torch.Tensor", patch_size: int) -> "torch.Tensor":
    """
    Convert 3D array image of shape (image_height, image_width, num_channels) into 2D array of patches of shape
    (num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
    """
    batch_size, num_channels, image_height, image_width = images.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = images.reshape(
        batch_size, num_channels, num_patches_height, patch_size, num_patches_width, patch_size
    )
    patched_image = patched_image.permute(0, 2, 4, 3, 5, 1)
    patched_image = patched_image.reshape(batch_size, num_patches_height * num_patches_width, -1)
    return patched_image


def pad_along_first_dim(
    images: "torch.Tensor", target_length: int, pad_value: int = 0
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Pad the array along the first dimension.
    """
    current_length = images.shape[1]
    padding_length = target_length - current_length
    pixel_mask = torch.ones((target_length,), dtype=torch.int32)
    if padding_length > 0:
        paddings = (0, 0, 0, padding_length, 0, 0)
        images = torch.nn.functional.pad(images, paddings, mode="constant", value=pad_value)
        pixel_mask[-padding_length:] = 0
    return images, pixel_mask


class Lfm2VlFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    downsample_factor (`int`, *optional*, defaults to `2`):
        The downsampling factor for images used when resizing the image.
    """

    downsample_factor: Optional[int]
    do_image_splitting: Optional[bool]
    min_tiles: Optional[int]
    max_tiles: Optional[int]
    use_thumbnail: Optional[bool]
    min_image_tokens: Optional[int]
    max_image_tokens: Optional[int]
    encoder_patch_size: Optional[int]
    tile_size: Optional[int]
    max_pixels_tolerance: Optional[float]
    do_pad: Optional[bool]
    return_row_col_info: Optional[bool]


@auto_docstring
class Lfm2VlImageProcessorFast(BaseImageProcessorFast):
    downsample_factor = 2
    do_image_splitting = True
    min_tiles = 2
    max_tiles = 10
    use_thumbnail = True
    min_image_tokens = 64
    max_image_tokens = 256
    encoder_patch_size = 16
    tile_size = 512
    max_pixels_tolerance = 2.0
    do_resize = True
    size = {"height": 512, "width": 512}
    resample = PILImageResampling.BILINEAR
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_pad = True
    return_row_col_info = False
    image_mean = IMAGENET_STANDARD_STD
    image_std = IMAGENET_STANDARD_MEAN
    valid_kwargs = Lfm2VlFastImageProcessorKwargs
    model_input_names = ["pixel_values", "pixel_attention_mask", "spatial_shapes"]

    def __init__(self, **kwargs: Unpack[Lfm2VlFastImageProcessorKwargs]):
        super().__init__(**kwargs)

        max_thumbnail_image_patches = self.max_image_tokens * self.downsample_factor**2
        tile_size_patches = (self.tile_size // self.encoder_patch_size) ** 2 if self.do_image_splitting else 0
        self.max_num_patches = max(
            max_thumbnail_image_patches,
            tile_size_patches,
        )

    @lru_cache(maxsize=256)
    def _target_ratios(self, min_tiles: int, max_tiles: int) -> list[tuple[int, int]]:
        ratios = [
            (w, h)
            for n in range(min_tiles, max_tiles + 1)
            for w in range(1, n + 1)
            for h in range(1, n + 1)
            if min_tiles <= w * h <= max_tiles
        ]
        return sorted(set(ratios), key=lambda x: x[0] * x[1])

    def _get_grid_layout(
        self,
        height: int,
        width: int,
        min_tiles: int,
        max_tiles: int,
        tile_size: int,
    ) -> tuple[int, int]:
        aspect_ratio = width / height
        target_ratios = self._target_ratios(min_tiles, max_tiles)

        # find best matching grid configuration
        grid_width, grid_height = find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, tile_size)

        target_width = tile_size * grid_width
        target_height = tile_size * grid_height
        total_patches = grid_width * grid_height

        return grid_width, grid_height, target_width, target_height, total_patches

    def crop_image_to_patches(
        self,
        image: "torch.Tensor",
        min_tiles: int,
        max_tiles: int,
        tile_size: int,
        use_thumbnail: bool,
        thumbnail_size: tuple[int],
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Processes a high resolution image into patches.
        This method splits a high resolution image into a grid of smaller patches while trying to maintain
        the original aspect ratio. It finds the optimal grid configuration within the specified tile constraints.
        """
        batch_size, num_channels, height, width = image.shape
        grid_width, grid_height, target_width, target_height, total_patches = self._get_grid_layout(
            height, width, min_tiles=min_tiles, max_tiles=max_tiles, tile_size=tile_size
        )
        resized_image = F.resize(
            image, (target_height, target_width), interpolation=interpolation, antialias=antialias
        )

        # split the image into patches
        processed_images = (
            resized_image.unfold(2, size=tile_size, step=tile_size)
            .unfold(3, size=tile_size, step=tile_size)
            .contiguous()
            .view(batch_size, num_channels, -1, tile_size, tile_size)
            .permute(2, 0, 1, 3, 4)
            .reshape(batch_size, -1, num_channels, tile_size, tile_size)
        )

        # Re-order processed images to a nested image structure, so it can be reordered back correctly
        # Note that the images can't be stacked because the thumbnail image is of bigger size than patches
        # Each image in sublist will be of shape (1, C, H, W)
        processed_images = list(processed_images)

        if use_thumbnail and grid_width * grid_height != 1:
            total_patches += 1
            thumbnail_image = F.resize(image, thumbnail_size, interpolation=interpolation, antialias=antialias)
            for i in range(batch_size):
                processed_images[i] = list(processed_images[i]) + list(thumbnail_image[i][None, ...])

        return processed_images, grid_width, grid_height

    # Adapted from Qwen-VL with minor differences
    def smart_resize(
        self,
        height: int,
        width: int,
        downsample_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
    ) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:
        1. Both dimensions (height and width) are divisible by 'encoder_patch_size' * 'downsample_factor'.
           This ensures no padding is needed in the downsampling step.
        2. The total number of pixels is within the range ['smart_resize_min_pixels', 'smart_resize_max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
        """
        total_factor = encoder_patch_size * downsample_factor
        smart_resize_min_pixels = min_image_tokens * encoder_patch_size**2 * downsample_factor**2
        smart_resize_max_pixels = max_image_tokens * encoder_patch_size**2 * downsample_factor**2

        h_bar = max(total_factor, round_by_factor(height, total_factor))
        w_bar = max(total_factor, round_by_factor(width, total_factor))

        if h_bar * w_bar > smart_resize_max_pixels:
            beta = math.sqrt((height * width) / smart_resize_max_pixels)
            math.floor(height / beta / total_factor) * total_factor
            h_bar = max(total_factor, math.floor(height / beta / total_factor) * total_factor)
            w_bar = max(total_factor, math.floor(width / beta / total_factor) * total_factor)
        elif h_bar * w_bar < smart_resize_min_pixels:
            beta = math.sqrt(smart_resize_min_pixels / (height * width))
            h_bar = math.ceil(height * beta / total_factor) * total_factor
            w_bar = math.ceil(width * beta / total_factor) * total_factor

        return w_bar, h_bar

    def _is_image_too_large(
        self,
        height: int,
        width: int,
        max_image_tokens: int,
        encoder_patch_size: int,
        downsample_factor: int,
        max_pixels_tolerance: float,
    ) -> bool:
        """Check if the image is too large to be processed as one tile."""
        total_factor = encoder_patch_size * downsample_factor

        h_bar = max(encoder_patch_size, round_by_factor(height, total_factor))
        w_bar = max(encoder_patch_size, round_by_factor(width, total_factor))
        return h_bar * w_bar > max_image_tokens * encoder_patch_size**2 * downsample_factor**2 * max_pixels_tolerance

    def resize_and_split(
        self,
        images: "torch.Tensor",
        downsample_factor: int,
        min_tiles: int,
        max_tiles: int,
        use_thumbnail: bool,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
        tile_size: int,
        max_pixels_tolerance: float,
        interpolation: "F.InterpolationMode",
    ) -> "torch.Tensor":
        batch_size, _, height, width = images.shape
        do_image_splitting = not min_tiles == max_tiles == 1
        is_image_large = self._is_image_too_large(
            height=height,
            width=width,
            max_image_tokens=max_image_tokens,
            encoder_patch_size=encoder_patch_size,
            downsample_factor=downsample_factor,
            max_pixels_tolerance=max_pixels_tolerance,
        )

        new_width, new_height = self.smart_resize(
            height=height,
            width=width,
            downsample_factor=downsample_factor,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
            encoder_patch_size=encoder_patch_size,
        )

        # Big image will be cropped into patches and small images are just resized
        if is_image_large and do_image_splitting:
            images, num_rows, num_cols = self.crop_image_to_patches(
                images,
                min_tiles=min_tiles,
                max_tiles=max_tiles,
                tile_size=tile_size,
                thumbnail_size=(new_height, new_width),
                use_thumbnail=use_thumbnail,
                interpolation=interpolation,
            )
        else:
            num_rows = num_cols = 1
            images = F.resize(images, (new_height, new_width), interpolation=interpolation)
            # Make a list and treat it as single crop per image so it can be re-grouped back correctly
            images = [[image] for image in images]

        num_rows = [num_rows] * batch_size
        num_cols = [num_cols] * batch_size
        image_sizes = [[new_height, new_width]] * batch_size
        return images, num_rows, num_cols, image_sizes

    def _preprocess(
        self,
        images: ImageInput,
        size: SizeDict,
        interpolation: "F.InterpolationMode",
        do_resize: bool,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, list[float]],
        image_std: Union[float, list[float]],
        downsample_factor: int,
        do_image_splitting: bool,
        min_tiles: int,
        max_tiles: int,
        use_thumbnail: bool,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
        tile_size: int,
        max_pixels_tolerance: float,
        return_tensors: Union[str, TensorType],
        disable_grouping: bool,
        do_pad: bool,
        return_row_col_info: bool,
        **kwargs,
    ) -> BatchFeature:
        if not do_image_splitting:
            min_tiles = 1
            max_tiles = 1
            logger.debug(
                "Image splitting is disabled, setting min_tiles and max_tiles to 1. Set do_image_splitting=True to enable splitting."
            )

        if do_image_splitting and min_tiles > max_tiles:
            raise ValueError("min_tiles must be less than or equal to max_tiles")

        max_thumbnail_image_patches = max_image_tokens * downsample_factor**2
        tile_size_patches = (tile_size // encoder_patch_size) ** 2 if do_image_splitting else 0
        max_num_patches = max(
            max_thumbnail_image_patches,
            tile_size_patches,
        )

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        resized_image_sizes = {}
        rows_grouped, cols_grouped = {}, {}
        for shape, stacked_images in grouped_images.items():
            num_rows = [1] * stacked_images.shape[0]
            num_cols = [1] * stacked_images.shape[0]
            height, width = stacked_images.shape[-2:]
            image_sizes = [[height, width]] * stacked_images.shape[0]
            do_resize = True

            if do_resize:
                stacked_images, num_rows, num_cols, image_sizes = self.resize_and_split(
                    stacked_images,
                    downsample_factor=downsample_factor,
                    min_tiles=min_tiles,
                    max_tiles=max_tiles,
                    use_thumbnail=use_thumbnail,
                    min_image_tokens=min_image_tokens,
                    max_image_tokens=max_image_tokens,
                    encoder_patch_size=encoder_patch_size,
                    tile_size=tile_size,
                    max_pixels_tolerance=max_pixels_tolerance,
                    interpolation=interpolation,
                )

            rows_grouped[shape] = num_rows
            cols_grouped[shape] = num_cols
            resized_image_sizes[shape] = image_sizes
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)
        batch_rows = reorder_images(rows_grouped, grouped_images_index)
        batch_cols = reorder_images(cols_grouped, grouped_images_index)
        resized_image_sizes = reorder_images(resized_image_sizes, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping, is_nested=True
        )

        processed_images_grouped = {}
        processed_masks, processed_spatial_shapes = {}, {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            batch_size, *_, height, width = stacked_images.shape
            num_patches_height = height // encoder_patch_size
            num_patches_width = width // encoder_patch_size

            stacked_images = convert_image_to_patches(stacked_images, encoder_patch_size)
            processed_spatial_shapes[shape] = [[num_patches_height, num_patches_width]] * batch_size

            if do_pad:
                stacked_images, pixel_mask = pad_along_first_dim(stacked_images, max_num_patches)
                processed_masks[shape] = [pixel_mask] * batch_size

            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index, is_nested=True)
        data = {"pixel_values": torch.cat([torch.stack(images) for images in processed_images])}

        if do_pad:
            processed_masks = reorder_images(processed_masks, grouped_images_index, is_nested=True)
            processed_spatial_shapes = reorder_images(processed_spatial_shapes, grouped_images_index, is_nested=True)
            processed_masks = torch.cat([torch.stack(masks) for masks in processed_masks])
            processed_spatial_shapes = torch.cat(
                [torch.tensor(spatial_shape) for spatial_shape in processed_spatial_shapes]
            )
            data.update({"pixel_attention_mask": processed_masks, "spatial_shapes": processed_spatial_shapes})

        if return_row_col_info:
            data["image_rows"] = batch_rows
            data["image_cols"] = batch_cols
            data["image_sizes"] = resized_image_sizes

        encoding = BatchFeature(data=data, tensor_type=return_tensors)
        return encoding


__all__ = ["Lfm2VlImageProcessorFast"]
