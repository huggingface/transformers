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

from functools import lru_cache
from typing import Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, filter_out_non_signature_kwargs, is_vision_available, logging


if is_vision_available():
    import PIL


logger = logging.get_logger(__name__)


# Similar to image_processing_mllama.get_all_supported_aspect_ratios
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
        `List[Tuple[int, int]]`: A list of tuples, each tuple representing a valid (width, height)
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


def compute_patch_covering_area(left: int, upper: int, right: int, lower: int, side: int) -> float:
    w = right - left
    h = lower - upper
    w, h = max(w, h), min(w, h)
    if w > side:
        h = h / w * side
        w = side
    return w * h


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
            sum([compute_patch_covering_area(*region, target_patch_size) for region in tile_regions]) / image_area
        )

        evaluated_grids.append((tile_grid, tile_covering_ratio))
        if tile_covering_ratio > covering_threshold:
            sufficient_covering_grids.append((tile_grid, tile_covering_ratio))

    if sufficient_covering_grids:
        # Prefer fewer tiles and higher covering ratio
        return sorted(sufficient_covering_grids, key=lambda x: (x[0][0] * x[0][1], -x[1]))[0][0]
    else:
        # Fallback: prefer higher covering even if below threshold
        return sorted(evaluated_grids, key=lambda x: (-x[1], x[0][0] * x[0][1]))[0][0]


class Ovis2ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Ovis2 image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        crop_to_patches (`bool`, *optional*, defaults to `False`):
            Whether to crop the image to patches. Can be overridden by the `crop_to_patches` parameter in the
            `preprocess` method.
        min_patches (`int`, *optional*, defaults to 1):
            The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
            set to `True`. Can be overridden by the `min_patches` parameter in the `preprocess` method.
        max_patches (`int`, *optional*, defaults to 12):
            The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
            set to `True`. Can be overridden by the `max_patches` parameter in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        use_covering_area_grid (`bool`, *optional*, defaults to `True`):
            Whether to use the covering area grid to determine the number of patches. Only has an effect if
            `crop_to_patches` is set to `True`. Can be overridden by the `use_covering_area_grid` parameter in the
            `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        crop_to_patches: bool = False,
        min_patches: int = 1,
        max_patches: int = 12,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = True,
        use_covering_area_grid: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 384, "width": 384}
        size = get_size_dict(size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.crop_to_patches = crop_to_patches
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb

    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        crop_to_patches: Optional[bool] = None,
        min_patches: Optional[int] = None,
        max_patches: Optional[int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        use_covering_area_grid: bool = True,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Controls the size of the image after `resize`. The shortest edge of the image is resized to
                `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
                is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
                edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
            crop_to_patches (`bool`, *optional*, defaults to `self.crop_to_patches`):
                Whether to crop the image to patches.
            min_patches (`int`, *optional*, defaults to `self.min_patches`):
                The minimum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
                set to `True`.
            max_patches (`int`, *optional*, defaults to `self.max_patches`):
                The maximum number of patches to be extracted from the image. Only has an effect if `crop_to_patches` is
                set to `True`.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            use_covering_area_grid (`bool`, *optional*, defaults to `True`):
                Whether to use the covering area grid to determine the number of patches. Only has an effect if
                `crop_to_patches` is set to `True`.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        crop_to_patches = crop_to_patches if crop_to_patches is not None else self.crop_to_patches
        min_patches = min_patches if min_patches is not None else self.min_patches
        max_patches = max_patches if max_patches is not None else self.max_patches
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=False)
        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        # PIL RGBA images are converted to RGB
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if crop_to_patches and max_patches > 1:
            images = [
                self.crop_image_to_patches(
                    image,
                    min_patches=min_patches,
                    max_patches=max_patches,
                    patch_size=size,
                    data_format=input_data_format,
                    use_covering_area_grid=use_covering_area_grid,
                )
                for image in images
            ]
            grids = [grid for _, grid in images]
            images = [image for images_list, _ in images for image in images_list]
        else:
            grids = [(1, 1)] * len(images)

        for i, image in enumerate(images):
            if do_resize:
                images[i] = self.resize(image, size=size, resample=resample, input_data_format=input_data_format)

            if do_rescale:
                images[i] = self.rescale(image=images[i], scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                images[i] = self.normalize(
                    image=images[i],
                    mean=image_mean,
                    std=image_std,
                    input_data_format=input_data_format,
                )

            images[i] = to_channel_dimension_format(images[i], data_format, input_channel_dim=input_data_format)

        encoded_outputs = BatchFeature(data={"pixel_values": images, "grids": grids}, tensor_type=return_tensors)

        return encoded_outputs

    def crop_image_to_patches(
        self,
        images: np.ndarray,
        min_patches: int,
        max_patches: int,
        use_covering_area_grid: bool = True,
        patch_size: Optional[Union[tuple, int, dict]] = None,
        data_format: ChannelDimension = None,
        covering_threshold: float = 0.9,
    ):
        """
        Crop the image to patches and return a list of cropped images.
        The number of patches and their grid arrangement are determined by the original image size,
        the target patch size and the minimum and maximum number of patches.
        The aspect ratio of the patches grid is chosen to be the closest to the original image aspect ratio.

        Args:
            images (`np.ndarray`):
                The image to be cropped.
            min_patches (`int`):
                The minimum number of patches to be extracted from the image.
            max_patches (`int`):
                The maximum number of patches to be extracted from the image.
            use_covering_area_grid (`bool`, *optional*, defaults to `True`):
                Whether to use the covering area grid to determine the number of patches.
            patch_size (`int`, `Tuple[int, int]`, `dict`, *optional*):
                The size of the output patches.
            data_format (`ChannelDimension`, *optional*):
                The format of the image data. If `None`, the format is inferred from the input image.
            covering_threshold (`float`, *optional*, defaults to `0.9`):
                The threshold for the covering area grid. If the covering area is less than this value, the grid is
                considered invalid.

        Returns:
            List[`PIL.Image.Image`] or List[np.ndarray]: The list of cropped images.
        """
        if data_format is None:
            data_format = infer_channel_dimension_format(images)
        images = to_channel_dimension_format(images, ChannelDimension.FIRST, data_format)
        patch_size_height, patch_size_width = patch_size["height"], patch_size["width"]
        original_height, original_width = images.shape[-2:]

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
                (original_height, original_width),
                (patch_size_height, patch_size_width),
                min_patches,
                max_patches,
            )

        # calculate the target width and height
        target_width = patch_size_width * num_columns
        target_height = patch_size_height * num_rows
        num_blocks = num_columns * num_rows

        # resize the image so that each patch is of patch_size
        resized_image = self.resize(
            images,
            {"height": target_height, "width": target_width},
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
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
            patch_image = to_channel_dimension_format(patch_image, data_format, ChannelDimension.FIRST)
            processed_images.append(patch_image)

        if len(processed_images) != 1:
            thumbnail_img = self.resize(
                images, patch_size, data_format=data_format, input_data_format=ChannelDimension.FIRST
            )
            processed_images.insert(0, thumbnail_img)

        return processed_images, (num_rows, num_columns)


__all__ = ["Ovis2ImageProcessor"]
