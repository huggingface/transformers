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

from typing import Any, Dict, List, Optional, Tuple, Union, Any, Tuple, Set, Dict, List

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


# TODO: update aspect ratio stack for different shapes
# TODO: update docs
# TODO: update copied from statements

logger = logging.get_logger(__name__)


def get_all_number_factors(n: int) -> Set[int]:
    """Return all factors of a number."""
    factors = set()
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)
    return factors


def find_supported_aspect_ratios(num_tiles: int) -> Dict[float, List[Tuple[int, int]]]:
    """
    This function computes all the allowed aspect ratios for a fixed
    number of input tiles.
    For example, with `num_tiles=5`, it will return:
    {
        0.2: [(1, 5)],
        5.0: [(5, 1)],
        0.25: [(1, 4)],
        1.0: [(2, 2), (1, 1)],
        4.0: [(4, 1)],
        0.3333333333333333: [(1, 3)],
        3.0: [(3, 1)],
        0.5: [(1, 2)],
        2.0: [(2, 1)]
    }
    """
    asp_dict = {}
    for tile_size in range(num_tiles, 0, -1):
        _factors = sorted(get_all_number_factors(tile_size))
        _asp_ratios = [(x, tile_size // x) for x in _factors]
        for ratio in _asp_ratios:
            k = ratio[0] / ratio[1]
            if k not in asp_dict:
                asp_dict[k] = [ratio]
            else:
                asp_dict[k].append(ratio)
    return asp_dict


def find_closest_aspect_ratio(num_tiles: int, img_width: int, img_height: int, tile_size: int) -> Tuple:
    """
    Given an image width, height and target number of tiles
    this function will find the closest supported aspect ratio.

    Args:
        tile_size: tile size

    """
    tgt_ar = img_width / img_height
    asp_dict = find_supported_aspect_ratios(num_tiles)
    cl_d, cl_p = 1e23, None
    if tgt_ar >= 1:
        cl_p = min(
            [k for k in asp_dict.keys() if k <= tgt_ar],
            key=lambda x: abs(x - tgt_ar),
        )
        v = asp_dict[cl_p]
        # select width
        widths = [(idx, tile_size * vv[0]) for idx, vv in enumerate(v)]
        tgt_idx = max(widths, key=lambda x: x[1])[0]
    else:
        cl_p = min(
            [k for k in asp_dict.keys() if k > tgt_ar],
            key=lambda x: abs(1 / x - 1 / tgt_ar),
        )
        v = asp_dict[cl_p]
        # select height
        heights = [(idx, tile_size * vv[1]) for idx, vv in enumerate(v)]
        tgt_idx = max(heights, key=lambda x: x[1])[0]
    out = v[tgt_idx]
    return out


def get_size_for_image_fitted_to_canvas(
    image_height: int,
    image_width: int,
    tile_size: int,
):
    scale = image_width / image_height

    if scale > 1.0:
        # width > height
        new_w = max(tile_size, image_width)
        new_h = math.floor(new_w / scale)
    else:
        # height >= width
        new_h = max(tile_size, image_height)
        new_w = math.floor(new_h * scale)

    return new_h, new_w


def get_size_for_image_not_fitted_to_canvas(
    image_height: int,
    image_width: int,
    canvas_height: int,
    canvas_width: int,
):
    scale = image_width / image_height

    if scale > 1.0:
        # width > height
        new_w = canvas_width
        new_h = math.floor(new_w / scale)
    else:
        # height >= width
        new_h = canvas_height
        new_w = math.floor(new_h * scale)

    return new_h, new_w


def get_target_image_size_and_aspect_ratio(
    image_height: int,
    image_width: int,
    max_image_tiles: int,
    tile_size: int,
):
    aspect_ratio = fit_image_to_canvas(
        num_tiles=max_image_tiles,
        img_width=image_width,
        img_height=image_height,
        tile_size=tile_size,
    )
    is_fit_to_canvas = aspect_ratio is not None

    if is_fit_to_canvas:
        size = get_size_for_image_fitted_to_canvas(
            image_height=image_height,
            image_width=image_width,
            tile_size=tile_size,
        )

    # If we did not find a canvas, we have to find the closest aspect ratio and downsample the image
    else:
        aspect_ratio = find_closest_aspect_ratio(
            num_tiles=max_image_tiles,
            img_width=image_width,
            img_height=image_height,
            tile_size=tile_size,
        )
        canvas_width = aspect_ratio[0] * tile_size
        canvas_height = aspect_ratio[1] * tile_size
        size = get_size_for_image_not_fitted_to_canvas(
            image_height=image_height,
            image_width=image_width,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
        )

    return size, aspect_ratio


# Copied from IDEFICS2
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
        `np.ndarray`: The image with the channel dimension set to `channel_dim`.
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
        raise ValueError(
            f"MllamaImageProcessor `max_image_tiles` must be a positive integer, got {max_image_tiles}."
        )
    validate_size(size)


def split_to_tiles(image: np.ndarray, ncw: int, nch: int) -> np.ndarray:
    # Split image into number of required tiles (width x height)
    num_channels, height, width = image.shape
    image = image.reshape(num_channels, nch, height // nch, ncw, width // ncw)
    # Permute dimensions to reorder the axes
    image = image.transpose(1, 3, 0, 2, 4)
    # Reshape into the desired output shape (batch_size * 4, num_channels, width/2, height/2)
    image = image.reshape(ncw * nch, num_channels, height // nch, width // ncw)
    # Make contiguous
    image = np.ascontiguousarray(image)
    return image


def fit_image_to_canvas(num_tiles: int, img_width: int, img_height: int, tile_size: int) -> Any:
    """
    Given an image width, height and target number of tiles this function will see if the image
    can be fit into any of the canvases that can be build from arranging the tiles in a grid.
    If the image can be fit onto several canvases, it will return the canvas where the shorter edge
    of the image will be largest.
    """
    # Initialize the optimal canvas to None. If no canvas is found where image fits, function returns None.
    optimal_canvas = None

    # Gather all potential supported image resolutions and iterate through them to find best match
    potential_arrangements = [
        item for sublist in find_supported_aspect_ratios(num_tiles).values() for item in sublist
    ]

    current_gap = 1e23
    for n_w, n_h in potential_arrangements:
        # Compute the canvas size
        canvas_width, canvas_height = n_w * tile_size, n_h * tile_size

        # Check if image can fit into the canvas without downsampling
        if canvas_width >= img_width and canvas_height >= img_height:
            # If we did not find a good canvas yet, we will use the current one
            if optimal_canvas is None:
                # Set optimal canvas and determine the actual image height and width in the canvas with aspect ratio preserving resampling
                optimal_canvas = (n_w, n_h)
            else:
                # Find closest fit based on gap
                image_width_height = (n_w * tile_size, n_h * tile_size)
                gap = abs(img_width - image_width_height[0]) + abs(img_height - image_width_height[1])
                if gap < current_gap:
                    # If the gap is smaller than the previous one, we will update our optimal canvas and image width height
                    optimal_canvas = (n_w, n_h)
                    current_gap = gap
    return optimal_canvas


def stack_images(
    batch_images: List[List[np.ndarray]],
    max_image_tiles: int,
) -> Tuple[np.ndarray, List[List[int]]]:
    # for each sample in a batch we have a list of images, and
    # each image is split into num_tiles tiles. So, the image is represented as array
    # of shape (num_tiles, channels, tile_height, tile_width), while the whole batch is
    # of shape (batch_size, num_images, num_tiles, channels, tile_height, tile_width)

    # TODO: in original code there is also max_images = max(max_images, 1)
    max_num_images = max([len(images) for images in batch_images])

    # collect shapes
    shapes = [image.shape for images in batch_images for image in images]
    _, channels, tile_height, tile_width = np.array(shapes).max(axis=0)

    out_images, out_num_tiles = [], []
    for images in batch_images:
        out_images_i = np.zeros(
            shape=(
                max_num_images,
                max_image_tiles,
                channels,
                tile_height,
                tile_width,
            ),
            dtype=np.float32,
        )
        num_tiles_i = []
        for j, image in enumerate(images):
            num_tiles = image.shape[0]
            out_images_i[j, :num_tiles] = image
            num_tiles_i.append(num_tiles)
        out_images.append(out_images_i)
        out_num_tiles.append(num_tiles_i)

    return np.stack(out_images), out_num_tiles


def stack_aspect_ratios(aspect_ratios: List[List[Tuple[int, int]]], pad_value: int = 1) -> np.ndarray:
    """
    Stack a list of aspect ratios into a numpy array.

    Args:
        aspect_ratios (`List[List[Tuple[int, int]]]`):
            A list of aspect ratios.
        pad_value (`int`, *optional*, defaults to 1):
            The value to pad the aspect ratios with.

    Returns:
        `np.ndarray`: The aspect ratios stacked into a numpy array with shape (batch_size, max_num_images, 2).
    """
    batch_size = len(aspect_ratios)

    # TODO: in original code there is also max_images = max(max_images, 1)
    max_num_images = max([len(row) for row in aspect_ratios])

    aspect_ratios_stacked = np.full((batch_size, max_num_images, 2), pad_value, dtype=np.int64)
    for i, row in enumerate(aspect_ratios):
        if len(row) > 0:
            aspect_ratios_stacked[i, : len(row)] = np.array(row)
    return aspect_ratios_stacked

def is_valid_list_of_images(images: List):
    return images and all([is_valid_image(image) for image in images])

# Inspired by IDEFICS2
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
    elif (
        isinstance(images, (list, tuple))
        and is_valid_list_of_images(images)
    ):
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


class MllamaImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values", "num_tiles", "aspect_ratios"]

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

        validate_mllama_preprocess_arguments(
            self.do_resize, self.size, self.do_pad, self.max_image_tiles
        )

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
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[ChannelDimension] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
    ):
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

        # do_resize=False is not supported, validated
        resized_images_and_aspect_ratios = [
            [
                self.resize(
                    image,
                    size,
                    resample=resample,
                    data_format=data_format,
                    input_data_format=input_data_format,
                    max_image_tiles=max_image_tiles,
                )
                for image in images
            ]
            for images in images_list
        ]
        images_list = [[image for image, ratio in images] for images in resized_images_and_aspect_ratios]
        aspect_ratio_list = [[ratio for image, ratio in images] for images in resized_images_and_aspect_ratios]

        # do_pad=False is not supported, validated
        images_list = [
            [
                self.pad(
                    image,
                    size,
                    aspect_ratio,
                    data_format=data_format,
                    input_data_format=input_data_format,
                )
                for image, aspect_ratio in zip(images, aspect_ratios)
            ]
            for images, aspect_ratios in zip(images_list, aspect_ratio_list)
        ]

        if do_rescale:
            images_list = [
                [
                    self.rescale(
                        image=image,
                        scale=rescale_factor,
                        input_data_format=input_data_format,
                    )
                    for image in images
                ]
                for images in images_list
            ]

        if do_normalize:
            images_list = [
                [self.normalize(image, mean=image_mean, std=image_std) for image in images] for images in images_list
            ]

        # Split each image to tiles
        images_list = [
            [split_to_tiles(image, aspect_ratio[0], aspect_ratio[1]) for image, aspect_ratio in zip(images, aspect_ratios)]
            for images, aspect_ratios in zip(images_list, aspect_ratio_list)
        ]

        images, num_tiles = stack_images(images_list, max_image_tiles)
        aspect_ratios = stack_aspect_ratios(aspect_ratio_list, pad_value=1)

        # images: (batch_size, num_images, MAX_num_tiles, channels, tile_height, tile_width) - padded to max num tiles
        # aspect_ratios: (batch_size, num_images, 2) - aspect ratios for each image, padded to max num images
        # num_tiles: (batch_size, num_images)  - real number of tiles for each image

        encoded_inputs = BatchFeature(
            data=dict(pixel_values=images, aspect_ratios=aspect_ratios),
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

        image_height, image_width = get_image_size(image)
        padded_height = aspect_ratio[1] * size["height"]
        padded_width = aspect_ratio[0] * size["width"]
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

        image_height, image_width = get_image_size(image)

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
