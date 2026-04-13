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
"""PIL Image processor class for OVIS2."""

from functools import lru_cache

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    infer_channel_dimension_format,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


# Adapted from transformers.models.ovis2.image_processing_ovis2.Ovis2ImageProcessorKwargs
class Ovis2ImageProcessorKwargs(ImagesKwargs, total=False):
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
    use_covering_area_grid (`bool`, *optional*, defaults to `True`):
        Whether to use the covering area grid to determine the number of patches. Only has an effect if
        `crop_to_patches` is set to `True`. Can be overridden by the `use_covering_area_grid` parameter in the
        `preprocess` method.
    """

    crop_to_patches: bool
    min_patches: int
    max_patches: int
    use_covering_area_grid: bool


# Adapted from transformers.models.ovis2.image_processing_ovis2.get_all_supported_aspect_ratios
@lru_cache(maxsize=10)
def get_all_supported_aspect_ratios(min_image_tiles: int, max_image_tiles: int) -> list[tuple[int, int]]:
    """Computes all allowed aspect ratios for a given minimum and maximum number of input tiles."""
    aspect_ratios = []
    for width in range(1, max_image_tiles + 1):
        for height in range(1, max_image_tiles + 1):
            if width * height <= max_image_tiles and width * height >= min_image_tiles:
                aspect_ratios.append((width, height))
    return sorted(aspect_ratios, key=lambda x: x[0] * x[1])


# Adapted from transformers.models.ovis2.image_processing_ovis2.compute_patch_covering_area
def compute_patch_covering_area(left: int, upper: int, right: int, lower: int, side: int) -> float:
    w = right - left
    h = lower - upper
    w, h = max(w, h), min(w, h)
    if w > side:
        h = h / w * side
        w = side
    return w * h


# Adapted from transformers.models.ovis2.image_processing_ovis2.split_image_into_grid
def split_image_into_grid(h: int, w: int, grid: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    row_height = h // grid[0]
    col_width = w // grid[1]
    return [
        (
            col * col_width,
            row * row_height,
            w if col == grid[1] - 1 else (col + 1) * col_width,
            h if row == grid[0] - 1 else (row + 1) * row_height,
        )
        for row in range(grid[0])
        for col in range(grid[1])
    ]


# Adapted from transformers.models.ovis2.image_processing_ovis2.get_min_tile_covering_grid
@lru_cache(maxsize=100)
def get_min_tile_covering_grid(
    image_size: tuple[int, int],
    target_patch_size: int,
    max_image_tiles: int,
    covering_threshold: float = 0.9,
) -> tuple[int, int]:
    image_height, image_width = image_size
    image_area = image_width * image_height
    candidate_tile_grids = get_all_supported_aspect_ratios(1, max_image_tiles)
    evaluated_grids = []
    sufficient_covering_grids = []

    for tile_grid in candidate_tile_grids:
        tile_regions = split_image_into_grid(image_height, image_width, tile_grid)
        tile_covering_ratio = (
            sum(compute_patch_covering_area(*region, target_patch_size) for region in tile_regions) / image_area
        )
        evaluated_grids.append((tile_grid, tile_covering_ratio))
        if tile_covering_ratio > covering_threshold:
            sufficient_covering_grids.append((tile_grid, tile_covering_ratio))

    if sufficient_covering_grids:
        return min(sufficient_covering_grids, key=lambda x: (x[0][0] * x[0][1], -x[1]))[0]
    return min(evaluated_grids, key=lambda x: (-x[1], x[0][0] * x[0][1]))[0]


# Adapted from transformers.models.ovis2.image_processing_ovis2.get_optimal_tiled_canvas
@lru_cache(maxsize=100)
def get_optimal_tiled_canvas(
    original_image_size: tuple[int, int],
    target_tile_size: tuple[int, int],
    min_image_tiles: int,
    max_image_tiles: int,
) -> tuple[int, int]:
    """Find the canvas with the closest aspect ratio to the original image aspect ratio."""
    possible_tile_arrangements = get_all_supported_aspect_ratios(min_image_tiles, max_image_tiles)
    original_height, original_width = original_image_size
    target_tile_height, target_tile_width = target_tile_size
    aspect_ratio = original_width / original_height
    area = original_width * original_height

    best_ratio_diff = float("inf")
    best_grid = (1, 1)
    for grid in possible_tile_arrangements:
        grid_aspect_ratio = grid[0] / grid[1]
        ratio_diff = abs(aspect_ratio - grid_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_grid = grid
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * target_tile_height * target_tile_width * grid[0] * grid[1]:
                best_grid = grid
    return best_grid


@auto_docstring
class Ovis2ImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    default_to_square = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    crop_to_patches = False
    min_patches = 1
    max_patches = 12
    use_covering_area_grid = True
    valid_kwargs = Ovis2ImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[Ovis2ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Ovis2ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def crop_image_to_patches(
        self,
        image: np.ndarray,
        min_patches: int,
        max_patches: int,
        use_covering_area_grid: bool = True,
        covering_threshold: float = 0.9,
        patch_size: SizeDict | None = None,
        resample: "PILImageResampling | None" = None,
    ):
        """
        Crop the image to patches and return a list of cropped images.
        Mirrors TorchvisionBackend.crop_image_to_patches.
        """
        # Normalize to CHW when called directly (e.g. from tests); _preprocess already receives CHW
        input_data_format = infer_channel_dimension_format(image)
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

        patch_size_height, patch_size_width = patch_size.height, patch_size.width
        original_height, original_width = image.shape[-2:]

        if use_covering_area_grid:
            # Use the original OVIS2 approach: compute the minimal number of tiles that cover at least 90% of the image area
            num_columns, num_rows = get_min_tile_covering_grid(
                (original_height, original_width),
                target_patch_size=patch_size_height,  # square patch size
                max_image_tiles=max_patches,
                covering_threshold=covering_threshold,
            )
        else:
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
            patch_image = resized_image[:, box[1] : box[3], box[0] : box[2]]
            processed_images.append(patch_image)

        if len(processed_images) != 1:
            thumbnail_img = self.resize(image, patch_size, resample=resample)
            processed_images.insert(0, thumbnail_img)

        return processed_images, [num_rows, num_columns]

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        crop_to_patches: bool = False,
        min_patches: int = 1,
        max_patches: int = 12,
        use_covering_area_grid: bool = True,
        **kwargs,
    ) -> BatchFeature:
        if crop_to_patches and max_patches > 1:
            # Crop to patches first
            processed_images = []
            grids = []
            for image in images:
                patches, grid = self.crop_image_to_patches(
                    image,
                    min_patches,
                    max_patches,
                    patch_size=size,
                    use_covering_area_grid=use_covering_area_grid,
                    resample=resample,
                )
                processed_images.extend(patches)
                grids.append(grid)
            images = processed_images
        else:
            grids = [[1, 1] for _ in range(len(images))]

        # Process all images (including patches if any) through the standard pipeline
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size=size, resample=resample)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images, "grids": grids}, tensor_type=return_tensors)


__all__ = ["Ovis2ImageProcessorPil"]
