# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
import numpy as np
from functools import lru_cache
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
    PaddingMode,
    get_image_size,
)
from ...image_transforms import pad, resize
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_valid_image,
    to_numpy_array,
    validate_preprocess_arguments,
)
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


@lru_cache(maxsize=10)
def get_all_number_factors(number: int) -> List[int]:
    """
    Return a sorted list of all unique factors of a given number.

    This function calculates and returns all positive integers that evenly divide the input number,
    including 1 and the number itself. The factors are returned in ascending order.

    Args:
        number (`int`):
            The positive integer for which to find factors.

    Returns:
        `List[int]`:
            A sorted list of all unique factors of the input number.

    Examples:
        >>> get_all_number_factors(4)
        [1, 2, 4]
        >>> get_all_number_factors(6)
        [1, 2, 3, 6]
        >>> get_all_number_factors(12)
        [1, 2, 3, 4, 6, 12]

    Note:
        - The function uses an optimized algorithm that only checks up to the square root of the input number.
        - The result is cached using the @lru_cache decorator for improved performance on repeated calls.
    """
    factors = set()
    max_possible_factor = int(number**0.5) + 1
    for factor in range(1, max_possible_factor):
        if number % factor == 0:
            factors.add(factor)
            factors.add(number // factor)
    return sorted(factors)


@lru_cache(maxsize=10)
def find_supported_aspect_ratios(max_image_tiles: int) -> Dict[float, List[Tuple[int, int]]]:
    """
    Computes all allowed aspect ratios for a given maximum number of input tiles.

    This function calculates all possible arrangements of tiles that can be formed
    within the constraint of the maximum number of tiles. Each arrangement is
    represented by its aspect ratio (width/height) and the corresponding tile configuration.

    Args:
        max_image_tiles (int):
            The maximum number of tiles allowed.

    Returns:
        `Dict[float, List[Tuple[int, int]]]`: A dictionary where:
            - Keys are aspect ratios (`float`)
            - Values are lists of tuples, each tuple representing a valid (width, height)
              configuration in terms of number of tiles.

    Example:
        For max_image_tiles=5, the function returns:
        {
            0.2: [(1, 5)],   # 1 tile wide, 5 tiles high
            5.0: [(5, 1)],   # 5 tiles wide, 1 tile high
            0.25: [(1, 4)],  # 1 tile wide, 4 tiles high
            1.0: [(2, 2), (1, 1)],  # Square configurations
            4.0: [(4, 1)],   # 4 tiles wide, 1 tile high
            0.3333333333333333: [(1, 3)],  # 1 tile wide, 3 tiles high
            3.0: [(3, 1)],   # 3 tiles wide, 1 tile high
            0.5: [(1, 2)],   # 1 tile wide, 2 tiles high
            2.0: [(2, 1)]    # 2 tiles wide, 1 tile high
        }

    Note:
        - The aspect ratio is calculated as width/height.
        - Multiple configurations can have the same aspect ratio (e.g., 2x2 and 1x1).
        - The function considers all divisors of numbers up to max_image_tiles.
    """
    aspect_ratios_dict = defaultdict(list)
    for num_tiles in range(max_image_tiles, 0, -1):
        factors = get_all_number_factors(num_tiles)
        for num_tiles_width in factors:
            num_tiles_height = num_tiles // num_tiles_width
            ratio = num_tiles_width / num_tiles_height
            aspect_ratios_dict[ratio].append((num_tiles_width, num_tiles_height))
    return aspect_ratios_dict


def find_closest_aspect_ratio(max_image_tiles: int, image_width: int, image_height: int) -> Tuple:
    """
    Find the closest supported aspect ratio for an image given its dimensions and maximum number of tiles.

    This function determines the most suitable aspect ratio for an image, considering the constraints
    of a maximum number of tiles. It aims to find a tile configuration that closely matches the
    original image's aspect ratio while staying within the tile limit.

    Args:
        max_image_tiles (`int`):
            The maximum number of tiles allowed for the image.
        image_width (`int`):
            The width of the input image.
        image_height (`int`):
            The height of the input image.

    Returns:
        `Tuple[int, int]`: A tuple representing the number of tiles in width and height
        for the closest supported aspect ratio.

    Note:
        The function uses the `find_supported_aspect_ratios` to get all possible tile configurations,
        then selects the one that best matches the input image's aspect ratio.
    """
    target_aspect_ratio = image_width / image_height
    aspect_ratio_dict = find_supported_aspect_ratios(max_image_tiles)

    if target_aspect_ratio >= 1:
        # Search closest aspect ratio
        ratios = [ratio for ratio in aspect_ratio_dict if ratio <= target_aspect_ratio]
        closest_aspect_ratio = min(ratios, key=lambda ratio: abs(ratio - target_aspect_ratio))
        aspect_ratio_factors = aspect_ratio_dict[closest_aspect_ratio]
        # Find the aspect ratio factor with the maxium width
        widths = [num_tiles_width for num_tiles_width, num_tiles_height in aspect_ratio_factors]
        index = widths.index(max(widths))
        aspect_ratio = aspect_ratio_factors[index]
    else:
        # Search closest aspect ratio
        ratios = [ratio for ratio in aspect_ratio_dict if ratio > target_aspect_ratio]
        closest_aspect_ratio = min(ratios, key=lambda ratio: abs(1 / ratio - 1 / target_aspect_ratio))
        aspect_ratio_factors = aspect_ratio_dict[closest_aspect_ratio]
        # Find the aspect ratio factor with the maxium height
        heights = [num_tiles_height for num_tiles_width, num_tiles_height in aspect_ratio_factors]
        index = heights.index(max(heights))
        aspect_ratio = aspect_ratio_factors[index]
    return aspect_ratio


def get_size_for_image_fitted_to_tile_size(
    image_height: int,
    image_width: int,
    tile_size: int,
) -> Tuple[int, int]:
    """
    Calculate the size of an image when fitted to a tile of a specific size while maintaining aspect ratio.

    This function determines the new dimensions of an image when it's scaled to fit
    a tile of specified size. The scaling ensures that:
    1. The larger dimension of the image is at least as large as the tile size.
    2. The smaller dimension is scaled proportionally to maintain the original aspect ratio.

    Args:
        image_height (`int`):
            The height of the original image.
        image_width (`int`):
            The width of the original image.
        tile_size (`int`):
            The size of the tile to fit the image into.

    Returns:
        `Tuple[int, int]`:
            A tuple containing the new (height, width) of the fitted image.
    """
    scale = image_width / image_height

    if image_width > image_height:
        new_image_width = max(tile_size, image_width)
        new_image_height = math.floor(new_image_width / scale)
    else:
        new_image_height = max(tile_size, image_height)
        new_image_width = math.floor(new_image_height * scale)

    return new_image_height, new_image_width


def get_size_for_image_fitted_to_canvas(
    image_height: int,
    image_width: int,
    canvas_height: int,
    canvas_width: int,
) -> Tuple[int, int]:
    """
    Calculate the size of an image when fitted to a canvas while maintaining aspect ratio.

    This function determines the new dimensions of an image when it's scaled to fit
    a canvas of specified height and width. The scaling ensures that:
    1. The larger dimension of the image matches the corresponding dimension of the canvas.
    2. The smaller dimension is scaled proportionally to maintain the original aspect ratio.

    Args:
        image_height (`int`):
            The height of the original image.
        image_width (`int`):
            The width of the original image.
        canvas_height (`int`):
            The height of the target canvas.
        canvas_width (`int`): The width of the target canvas.

    Returns:
        `Tuple[int, int]`:
            A tuple containing the new (height, width) of the fitted image.
    """
    scale = image_width / image_height

    if image_width > image_height:
        new_image_width = canvas_width
        new_image_height = math.floor(new_image_width / scale)
    else:
        new_image_height = canvas_height
        new_image_width = math.floor(new_image_height * scale)

    return new_image_height, new_image_width


def get_aspect_ratio_of_optimal_canvas_larger_than_image(
    max_image_tiles: int, image_width: int, image_height: int, tile_size: int
) -> Optional[Tuple[int, int]]:
    """
    Determines the optimal canvas size for an image given a maximum number of tiles.

    This function attempts to fit an image without downsampling into various canvas sizes that can be constructed
    from a grid of tiles. It aims to find the best fit that minimizes unused space while
    maximizing the image's shorter edge.

    Args:
        max_image_tiles (`int`):
            The maximum number of tiles available to construct the canvas.
        image_width (`int`):
            The width of the input image.
        image_height (`int`):
            The height of the input image.
        tile_size (`int`):
            The size of each square tile (width and height are equal).

    Returns:
        `Optional[Tuple[int, int]]`:
            A tuple containing the number of tiles in width and height
            for the optimal canvas, or None if no suitable canvas is found.
    """
    # Initialize the optimal canvas to None. If no canvas is found where image fits, function returns None.
    optimal_canvas = None

    # Gather all potential supported canvas arrangements
    potential_arrangements = []
    aspect_ratios_dict = find_supported_aspect_ratios(max_image_tiles)
    for aspect_ratios in aspect_ratios_dict.values():
        potential_arrangements.extend(aspect_ratios)

    best_gap = float("inf")

    for num_tiles_width, num_tiles_height in potential_arrangements:
        # Compute the canvas size
        canvas_width = num_tiles_width * tile_size
        canvas_height = num_tiles_height * tile_size

        # Check if image can fit into the canvas without downsampling
        if canvas_width >= image_width and canvas_height >= image_height:
            # If we did not find a good canvas yet, we will use the current one
            if optimal_canvas is None:
                optimal_canvas = (num_tiles_width, num_tiles_height)

            # Compute the gap between the canvas and the image and update the best gap
            current_gap = (canvas_width - image_width) + (canvas_height - image_height)
            if current_gap < best_gap:
                optimal_canvas = (num_tiles_width, num_tiles_height)
                best_gap = current_gap

    return optimal_canvas


@lru_cache(maxsize=100)
def get_target_image_size_and_aspect_ratio(
    image_height: int,
    image_width: int,
    max_image_tiles: int,
    tile_size: int,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Get the target image size and aspect ratio for an image to fit optimally within a tiled canvas.

    This function determines the best size to resize an image and the optimal aspect ratio of the canvas
    it should be placed on, given constraints on the maximum number of tiles and tile size.

    The function follows these steps:
    1. Attempt to find an optimal canvas larger than the image.
    2. If a larger canvas is found, resize the image to better fit the tile size.
    3. If no larger canvas is found, find the closest possible aspect ratio and downscale the image.

    The result of the function is cached and will be not recomuted for the same parameters.

    Args:
        image_height (`int`):
            The height of the original image.
        image_width (`int`):
            The width of the original image.
        max_image_tiles (`int`):
            The maximum number of tiles allowed in the canvas.
        tile_size (`int`):
            The size of each tile (assumed to be square).

    Returns:
        `Tuple[Tuple[int, int], Tuple[int, int]]`:
            A tuple containing:
            - The new dimensions (height, width) for the resized image.
            - The aspect ratio of the canvas as (num_tiles_width, num_tiles_height)
    """

    # Get the aspect ratio of the optimal canvas larger than the image
    # if no canvas larger than image can be found with given parameters,
    # aspect_ratio will be None
    aspect_ratio = get_aspect_ratio_of_optimal_canvas_larger_than_image(
        max_image_tiles=max_image_tiles,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
    )

    # If we found a canvas, we get the optimal size for the image to better fit the tile size
    if aspect_ratio is not None:
        new_image_height, new_image_width = get_size_for_image_fitted_to_tile_size(
            image_height=image_height,
            image_width=image_width,
            tile_size=tile_size,
        )

    # If we did not find a canvas larger than the image,
    # we have to find the closest aspect ratio and downsample the image
    else:
        aspect_ratio = find_closest_aspect_ratio(
            max_image_tiles=max_image_tiles,
            image_width=image_width,
            image_height=image_height,
        )
        num_tiles_width, num_tiles_height = aspect_ratio
        canvas_width = num_tiles_width * tile_size
        canvas_height = num_tiles_height * tile_size
        new_image_height, new_image_width = get_size_for_image_fitted_to_canvas(
            image_height=image_height,
            image_width=image_width,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
        )

    return (new_image_height, new_image_width), aspect_ratio


def split_to_tiles(image: np.ndarray, num_tiles_width: int, num_tiles_height: int) -> np.ndarray:
    """
    Split an image into a specified number of tiles along its width and height dimensions.

    Args:
        image (`np.ndarray`):
            Input image with shape (num_channels, height, width).
        num_tiles_width (`int`):
            Number of tiles to split the image into along its width.
        num_tiles_height (`int`):
            Number of tiles to split the image into along its height.

    Returns:
        `np.ndarray`:
            Array of image tiles with shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width).
    """
    num_channels, height, width = image.shape
    tile_height = height // num_tiles_height
    tile_width = width // num_tiles_width

    image = image.reshape(num_channels, num_tiles_height, tile_height, num_tiles_width, tile_width)

    # Permute to (num_tiles_height, num_tiles_width, num_channels, tile_height, tile_width)
    image = image.transpose(1, 3, 0, 2, 4)
    # image = image.transpose(0, 2, 4, 1, 3)

    # Reshape into the desired output shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)
    image = image.reshape(num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)

    return np.ascontiguousarray(image)


def pack_images(
    batch_images: List[List[np.ndarray]],
    max_image_tiles: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Stack a list of lists of images with variable lengths into a numpy array, applying zero padding as needed.
    Each list in the input represents a batch sample, and each image within a list is expected to be
    pre-split into tiles. The resulting array will have a shape of
    (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width).

    Args:
        batch_images (`List[List[np.ndarray]]`):
            A list of lists of image tiles. Each inner list represents
            a batch sample containing multiple images, where each image is pre-split into tiles.
            The shape of each tile array is (num_tiles, channels, tile_height, tile_width).
        max_image_tiles (int):
            The maximum number of tiles any image was potantially split.

    Returns:
        `Tuple[np.ndarray, List[List[int]]]`: A tuple containing:
            - stacked_images (`np.ndarray`):
                A numpy array of stacked images with shape
                (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width).
            - all_num_tiles (`List[List[int]]`):
                A list of lists containing the number of tiles
                for each image in each batch sample.
    """

    # Determine output shape
    batch_size = len(batch_images)
    max_num_images = max([len(images) for images in batch_images])
    shapes = [image.shape for images in batch_images for image in images]
    _, channels, tile_height, tile_width = shapes[0]

    # Initialize the stacked images array with zeros
    stacked_images = np.zeros(
        (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width),
        dtype=np.float32,
    )

    # Fill the stacked images array with the tiled images from the batch
    all_num_tiles = []
    for i, images in enumerate(batch_images):
        num_sample_tiles = []
        for j, image in enumerate(images):
            num_tiles = image.shape[0]
            stacked_images[i, j, :num_tiles] = image
            num_sample_tiles.append(num_tiles)
        all_num_tiles.append(num_sample_tiles)

    return stacked_images, all_num_tiles


def pack_aspect_ratios(aspect_ratios: List[List[Tuple[int, int]]], pad_value: int = 1) -> np.ndarray:
    """
    Stack a list of aspect ratios into a numpy array.

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of aspect ratios.
        pad_value (`int`, *optional*, defaults to 1):
            The value to pad the aspect ratios with.

    Returns:
        `np.ndarray`:
            The aspect ratios stacked into a numpy array with shape (batch_size, max_num_images, 2).
    """
    batch_size = len(aspect_ratios)

    # TODO: in original code there is also max_images = max(max_images, 1)
    max_num_images = max([len(row) for row in aspect_ratios])

    aspect_ratios_stacked = np.full((batch_size, max_num_images, 2), pad_value, dtype=np.int64)
    for i, row in enumerate(aspect_ratios):
        if len(row) > 0:
            aspect_ratios_stacked[i, : len(row)] = np.array(row)
    return aspect_ratios_stacked


def convert_aspect_ratios_to_ids(aspect_ratios: List[List[Tuple[int, int]]], mux_num_tiles: int) -> np.ndarray:
    """
    Convert aspect ratio tuples to unique ids with the following encoding:

        id = (num_tiles_h - 1) * max_image_tiles + num_tiles_w

    For max_image_tiles = 4, we have the following encoding:

        - aspect ratio (1, 1) -> id = 1
        - aspect ratio (1, 2) -> id = 2
        - aspect ratio (1, 3) -> id = 3
        - aspect ratio (1, 4) -> id = 4
        - aspect ratio (2, 1) -> id = 5
        - aspect ratio (2, 2) -> id = 6
        - aspect ratio (3, 1) -> id = 9
        - aspect ratio (4, 1) -> id = 13

    For batch padding we use 0, because there might be different number of images in each batch.

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of aspect ratios.
        mux_num_tiles (`int`):
            The maximum number of tiles any image was potentially split into.

    Returns:
        `np.ndarray`:
            The aspect ratios ids as numpy array with shape (batch_size, max_num_images).
    """

    batch_size = len(aspect_ratios)
    max_num_images = max([len(row) for row in aspect_ratios])

    aspect_ratios_ids = np.zeros((batch_size, max_num_images), dtype=np.int64)
    for i, sample_aspect_ratios in enumerate(aspect_ratios):
        for j, (num_tiles_h, num_tiles_w) in enumerate(sample_aspect_ratios):
            aspect_ratios_ids[i, j] = (num_tiles_h - 1) * mux_num_tiles + num_tiles_w
    return aspect_ratios_ids


# Copied from transformers.models.idefics2.image_processing_idefics2.to_channel_dimension_format
def to_channel_dimension_format(
    image: np.ndarray,
    channel_dim: Union[ChannelDimension, str],
    input_channel_dim: Optional[Union[ChannelDimension, str]] = None,
) -> np.ndarray:
    """
    Converts `image` to the channel dimension format specified by `channel_dim`.

    Args:
        image (`numpy.ndarray`):
            The image to have its channel dimension set.
        channel_dim (`ChannelDimension`):
            The channel dimension format to use.
        input_channel_dim (`ChannelDimension`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred from the input image.

    Returns:
        `np.ndarray`:
            The image with the channel dimension set to `channel_dim`.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Input image must be of type np.ndarray, got {type(image)}")

    if input_channel_dim is None:
        input_channel_dim = infer_channel_dimension_format(image)

    target_channel_dim = ChannelDimension(channel_dim)
    if input_channel_dim == target_channel_dim:
        return image

    if target_channel_dim == ChannelDimension.FIRST:
        image = image.transpose((2, 0, 1))
    elif target_channel_dim == ChannelDimension.LAST:
        image = image.transpose((1, 2, 0))
    else:
        raise ValueError("Unsupported channel dimension format: {}".format(channel_dim))

    return image


# Modified from transformers.models.idefics2.image_processing_idefics2.make_list_of_images
def make_list_of_images(images: ImageInput) -> List[List[Optional[np.ndarray]]]:
    """
    Convert a single image or a list of images to a list of numpy arrays.

    Args:
        images (`ImageInput`):
            A single image or a list of images.

    Returns:
        A list of numpy arrays.
    """
    # If it's a single image, convert it to a list of lists
    if is_valid_image(images):
        output_images = [[images]]
    # If it's a list of images, it's a single batch, so convert it to a list of lists
    elif isinstance(images, (list, tuple)) and is_valid_list_of_images(images):
        output_images = [images]
    # If it's a list of batches, it's already in the right format
    elif (
        isinstance(images, (list, tuple))
        and all(isinstance(images_i, (list, tuple)) for images_i in images)
        and any([is_valid_list_of_images(images_i) for images_i in images])
    ):
        output_images = images
    else:
        raise ValueError(
            "Invalid input type. Must be a single image, a list of images, or a list of batches of images."
        )
    return output_images


def is_valid_list_of_images(images: List):
    return images and all([is_valid_image(image) for image in images])


def validate_size(size: Dict[str, int]) -> None:
    if not ("height" in size and "width" in size):
        raise ValueError(f"Argument `size` must be a dictionary with keys 'height' and 'width'. Got: {size}")
    if size["height"] != size["width"]:
        raise ValueError(f"Argument `size` must have the same height and width, got {size}")


def validate_mllama_preprocess_arguments(do_resize, size, do_pad, max_image_tiles):
    if not do_pad:
        raise ValueError("MllamaImageProcessor doesn't support `do_pad=False` mode.")
    if not do_resize:
        raise ValueError("MllamaImageProcessor doesn't support `do_resize=False` mode.")
    if max_image_tiles is None or max_image_tiles <= 0:
        raise ValueError(f"MllamaImageProcessor `max_image_tiles` must be a positive integer, got {max_image_tiles}.")
    validate_size(size)


class MllamaImageProcessor(BaseImageProcessor):
    """
    Constructs a Mllama image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*, defaults to `self.size`):
            Size of the image tile. Should be a dictionary containing 'height' and 'width' keys, both with integer values.
            The height and width values should be equal.
        resample (`int`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
            has an effect if `do_resize` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the image.
        rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
            Rescale factor to rescale the image by if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
            `True`.
        do_pad (`bool`, *optional*, defaults to `self.do_pad`):
            Whether or not to pad the images to the largest height and width in the batch.
        max_image_tiles (`int`, *optional*, defaults to `self.max_image_tiles`):
            The maximum number of tiles to split the image into.
    """

    model_input_names = ["pixel_values", "num_tiles", "aspect_ratios", "aspect_ratio_ids"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        max_image_tiles: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 224, "width": 224}
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.do_pad = do_pad
        self.max_image_tiles = max_image_tiles

        validate_mllama_preprocess_arguments(self.do_resize, self.size, self.do_pad, self.max_image_tiles)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        max_image_tiles: Optional[int] = None,
        input_data_format: Optional[ChannelDimension] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ):
        """
        Preprocess a batch of images.

        Args:
            images (`ImageInput`):
                A list of images to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image tile. Should be a dictionary containing 'height' and 'width' keys, both with integer values.
                The height and width values should be equal.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether or not to pad the images to the largest height and width in the batch.
            max_image_tiles (`int`, *optional*, defaults to `self.max_image_tiles`):
                The maximum number of tiles to split the image into.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.

        Returns:
            `BatchFeature` of the following structure:
                - **pixel_values** (`TensorType`): The preprocessed pixel values.
                - **aspect_ratios** (`TensorType`): The aspect ratios of the images.
                - **aspect_ratio_ids** (`TensorType`): The aspect ratio ids of the images.
                - **num_tiles** (`List[List[int]]`): The number of tiles for each image in the batch.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        max_image_tiles = max_image_tiles if max_image_tiles is not None else self.max_image_tiles

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

        # extra validation
        validate_mllama_preprocess_arguments(do_resize, size, do_pad, max_image_tiles)

        images_list = make_list_of_images(images)
        images_list = [[to_numpy_array(image) for image in images] for images in images_list]

        # convert all images to (num_channels, height, width) format, it's much faster for preprocessing
        data_format = ChannelDimension.FIRST
        images_list = [
            [
                to_channel_dimension_format(image, ChannelDimension.FIRST, input_channel_dim=input_data_format)
                for image in images
            ]
            for images in images_list
        ]

        batch_images = []
        batch_aspect_ratios = []

        # iterate over batch samples
        for images in images_list:
            sample_images = []
            sample_aspect_ratios = []

            # iterate over images in a batch sample
            for image in images:
                # do_resize=False is not supported, validated
                image, aspect_ratio = self.resize(
                    image=image,
                    size=size,
                    resample=resample,
                    max_image_tiles=max_image_tiles,
                    input_data_format=data_format,
                    data_format=data_format,
                )

                # do_pad=False is not supported, validated
                image = self.pad(
                    image=image,
                    size=size,
                    aspect_ratio=aspect_ratio,
                    input_data_format=data_format,
                    data_format=data_format,
                )

                if do_rescale:
                    image = self.rescale(
                        image=image,
                        scale=rescale_factor,
                        input_data_format=input_data_format,
                        data_format=data_format,
                    )

                if do_normalize:
                    image = self.normalize(
                        image=image,
                        mean=image_mean,
                        std=image_std,
                        input_data_format=input_data_format,
                        data_format=data_format,
                    )

                num_tiles_width, num_tiles_height = aspect_ratio
                image = split_to_tiles(image, num_tiles_width, num_tiles_height)

                sample_images.append(image)
                sample_aspect_ratios.append(aspect_ratio)

            batch_images.append(sample_images)
            batch_aspect_ratios.append(sample_aspect_ratios)

        images, num_tiles = pack_images(batch_images, max_image_tiles)

        # TODO: aspect ratios not be needed when ids are supported in modeling code
        aspect_ratios = pack_aspect_ratios(batch_aspect_ratios, pad_value=1)
        aspect_ratio_ids = convert_aspect_ratios_to_ids(batch_aspect_ratios, mux_num_tiles=max_image_tiles)

        # images (np.ndarray) with shape (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
        # aspect_ratios (np.ndarray) with shape (batch_size, max_num_images, 2) - aspect ratios for each image, padded to max_num_images with 1
        # aspect_ratio_ids (np.ndarray) with shape (batch_size, max_num_images) - aspect ratio ids for each image, padded to max_num_images with 0
        # num_tiles (List[List[int]]) with (batch_size, num_images_in_batch) - real number of tiles for each image, not padded
        encoded_inputs = BatchFeature(
            data=dict(pixel_values=images, aspect_ratios=aspect_ratios, aspect_ratio_ids=aspect_ratio_ids),
            tensor_type=return_tensors,
        )
        encoded_inputs["num_tiles"] = num_tiles

        return encoded_inputs

    def pad(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        aspect_ratio: Tuple[int, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad an image to the `size` x `aspect_ratio`. For example, if size is {height: 224, width: 224} and aspect ratio is
        (1, 2), the image will be padded to 224x448.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            aspect_ratio (`Tuple[int, int]`):
                The aspect ratio of the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            `np.ndarray`: The padded image.
        """

        validate_size(size)

        image_height, image_width = get_image_size(image, channel_dim=input_data_format)
        num_tiles_width, num_tiles_height = aspect_ratio
        padded_height = num_tiles_height * size["height"]
        padded_width = num_tiles_width * size["width"]
        pad_size = ((0, padded_height - image_height), (0, padded_width - image_width))

        image = pad(
            image,
            pad_size,
            mode=PaddingMode.CONSTANT,
            constant_values=0,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        return image

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        max_image_tiles: int,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Union[np.ndarray, Tuple[int, int]]:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """

        validate_size(size)

        image_height, image_width = get_image_size(image, channel_dim=input_data_format)

        (new_height, new_width), aspect_ratio = get_target_image_size_and_aspect_ratio(
            image_height=image_height,
            image_width=image_width,
            max_image_tiles=max_image_tiles,
            tile_size=size["height"],
        )

        image = resize(
            image,
            (new_height, new_width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        return image, aspect_ratio
