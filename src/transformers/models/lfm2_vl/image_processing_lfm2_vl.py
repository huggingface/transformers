# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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

import numpy as np
from PIL import Image

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
    convert_to_rgb,
    to_channel_dimension_format,
    to_pil_image,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    make_nested_list_of_images,
    to_numpy_array,
    validate_preprocess_arguments,
)
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
)
from ...utils import TensorType, filter_out_non_signature_kwargs, logging


logger = logging.get_logger(__name__)


# resize adapted from qwen2.5
# https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
def round_by_factor(number: float, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


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


def convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Convert 3D array image of shape (image_height, image_width, num_channels) into 2D array of patches of shape
    (num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
    """
    image_height, image_width, num_channels = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(num_patches_height, patch_size, num_patches_width, patch_size, num_channels)
    patched_image = patched_image.transpose(0, 2, 1, 3, 4)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def pad_along_first_dim(array: np.ndarray, target_length: int, pad_value: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad the array along the first dimension.
    """
    current_length = array.shape[0]
    padding_length = target_length - current_length
    mask = np.ones((target_length,), dtype=np.int32)
    if padding_length > 0:
        paddings = [(0, padding_length)] + [(0, 0)] * (array.ndim - 1)
        array = np.pad(array, paddings, mode="constant", constant_values=pad_value)
        mask[-padding_length:] = 0
    return array, mask


class Lfm2VlImagesKwargs(ImagesKwargs, total=False):
    return_row_col_info: Optional[bool]
    max_image_size: Optional[dict[str, int]]


class Lfm2VlProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Lfm2VlImagesKwargs

    _defaults = {
        "text_kwargs": {
            "add_special_tokens": False,
            "padding": False,
            "is_split_into_words": False,
        },
    }


class Lfm2VlImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Lfm2VlImageProcessor.

    [`Lfm2VlImageProcessor`] offers all the functionalities of [`Siglip2ImageProcessor`], and extends it with smart resize and image splitting.

    Args:
            downsample_factor (`int`, *optional*, defaults to 2): <fill_docstring>
            do_image_splitting (`bool`, *optional*, defaults to `True`): <fill_docstring>
            min_tiles (`int`, *optional*, defaults to 2): <fill_docstring>
            max_tiles (`int`, *optional*, defaults to 10): <fill_docstring>
            use_thumbnail (`bool`, *optional*, defaults to `True`): <fill_docstring>
            min_image_tokens (`int`, *optional*, defaults to 64): <fill_docstring>
            max_image_tokens (`int`, *optional*, defaults to 256): <fill_docstring>
            encoder_patch_size (`int`, *optional*, defaults to 16): <fill_docstring>
            tile_size (`int`, *optional*, defaults to 512): <fill_docstring>
            max_pixels_tolerance (`float`, *optional*, defaults to 2.0): <fill_docstring>
            resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`): <fill_docstring>
            do_rescale (`bool`, *optional*, defaults to `True`): <fill_docstring>
            rescale_factor (`float`, *optional*, defaults to 0.0): <fill_docstring>
            do_normalize (`bool`, *optional*, defaults to `True`): <fill_docstring>
            image_mean (`Union`, *optional*): <fill_docstring>
            image_std (`Union`, *optional*): <fill_docstring>
            do_convert_rgb (`Optional`, *optional*): <fill_docstring>
    """

    def __init__(
        self,
        downsample_factor: int = 2,
        do_image_splitting: bool = True,
        min_tiles: int = 2,
        max_tiles: int = 10,
        use_thumbnail: bool = True,
        min_image_tokens: int = 64,
        max_image_tokens: int = 256,
        encoder_patch_size: int = 16,
        tile_size: int = 512,
        max_pixels_tolerance: float = 2.0,
        resample: "PILImageResampling" = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        **kwargs,
    ):
        self.downsample_factor = downsample_factor
        self.do_image_splitting = do_image_splitting
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.use_thumbnail = use_thumbnail
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.encoder_patch_size = encoder_patch_size
        self.tile_size = tile_size
        self.max_pixels_tolerance = max_pixels_tolerance
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb

        # copied from Siglip2ImageProcessor
        image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]

        super().__init__(
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            **kwargs,
        )

        max_thumbnail_image_patches = max_image_tokens * downsample_factor**2
        tile_size_patches = (tile_size // encoder_patch_size) ** 2 if self.do_image_splitting else 0
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

    def _get_grid_layout(self, image: Image.Image, min_tiles: int, max_tiles: int, tile_size: int) -> tuple[int, int]:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = self._target_ratios(min_tiles, max_tiles)

        # find best matching grid configuration
        grid_width, grid_height = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, tile_size
        )

        target_width = tile_size * grid_width
        target_height = tile_size * grid_height
        total_patches = grid_width * grid_height

        return grid_width, grid_height, target_width, target_height, total_patches

    def _high_res_preprocessor(
        self,
        image: Image.Image,
        min_tiles,
        max_tiles,
        tile_size,
        resample: "PILImageResampling",
    ) -> list[Image.Image]:
        """Process a high resolution image into patches.
        This method splits a high resolution image into a grid of smaller patches while trying to maintain
        the original aspect ratio. It finds the optimal grid configuration within the specified tile constraints.
        """
        grid_width, _, target_width, target_height, total_patches = self._get_grid_layout(
            image, min_tiles, max_tiles, tile_size
        )
        # resize and split image into patches
        resized_img = image.resize((target_width, target_height), resample=resample)
        patches = []

        for i in range(total_patches):
            # calculate patch coordinates
            col = i % grid_width
            row = i // grid_width
            box = (
                col * tile_size,
                row * tile_size,
                (col + 1) * tile_size,
                (row + 1) * tile_size,
            )
            patch = resized_img.crop(box)
            patches.append(patch)

        return patches

    def _smart_resize(
        self,
        image: Image.Image,
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
        width, height = image.size

        total_factor = encoder_patch_size * downsample_factor
        smart_resize_min_pixels = min_image_tokens * encoder_patch_size**2 * downsample_factor**2
        smart_resize_max_pixels = max_image_tokens * encoder_patch_size**2 * downsample_factor**2

        h_bar = max(total_factor, round_by_factor(height, total_factor))
        w_bar = max(total_factor, round_by_factor(width, total_factor))

        if h_bar * w_bar > smart_resize_max_pixels:
            beta = math.sqrt((height * width) / smart_resize_max_pixels)
            h_bar = max(total_factor, floor_by_factor(height / beta, total_factor))
            w_bar = max(total_factor, floor_by_factor(width / beta, total_factor))
        elif h_bar * w_bar < smart_resize_min_pixels:
            beta = math.sqrt(smart_resize_min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, total_factor)
            w_bar = ceil_by_factor(width * beta, total_factor)

        return w_bar, h_bar

    def _get_tokens_num(
        self, image_width: int, image_height: int, downsample_factor: int, encoder_patch_size: int
    ) -> int:
        num_patches_height = image_height // encoder_patch_size
        num_patches_width = image_width // encoder_patch_size

        dwn_num_patches_height = math.ceil(num_patches_height / downsample_factor)
        dwn_num_patches_width = math.ceil(num_patches_width / downsample_factor)

        return dwn_num_patches_height * dwn_num_patches_width

    def _is_img_too_large(
        self,
        image: Image.Image,
        max_image_tokens: int,
        encoder_patch_size: int,
        downsample_factor: int,
        max_pixels_tolerance: float,
    ) -> bool:
        """Check if the image is too large to be processed as one tile."""
        width, height = image.size

        total_factor = encoder_patch_size * downsample_factor

        h_bar = max(encoder_patch_size, round_by_factor(height, total_factor))
        w_bar = max(encoder_patch_size, round_by_factor(width, total_factor))
        return (
            h_bar * w_bar > max_image_tokens * encoder_patch_size**2 * self.downsample_factor**2 * max_pixels_tolerance
        )

    def _resize_and_maybe_split(
        self,
        image: ImageInput,
        downsample_factor: int,
        min_tiles: int,
        max_tiles: int,
        use_thumbnail: bool,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
        tile_size: int,
        max_pixels_tolerance: float,
        resample: "PILImageResampling",
    ) -> list[Image.Image]:
        """Apply smart resize and maybe split the image into tiles if image too large.
        Return:
            image_tiles: ImageInput
        """
        image = to_pil_image(image)
        do_image_splitting = not min_tiles == max_tiles == 1
        if (
            self._is_img_too_large(
                image,
                max_image_tokens,
                encoder_patch_size,
                downsample_factor,
                max_pixels_tolerance,
            )
            and do_image_splitting
        ):
            image_tiles = self._high_res_preprocessor(image, min_tiles, max_tiles, tile_size, resample)
            if len(image_tiles) > 1:
                if use_thumbnail:
                    thumbnail_width, thumbnail_height = self._smart_resize(
                        image,
                        downsample_factor,
                        min_image_tokens,
                        max_image_tokens,
                        encoder_patch_size,
                    )
                    thumbnail_image = image.resize((thumbnail_width, thumbnail_height), resample=resample)
                    image_tiles.append(thumbnail_image)

                return image_tiles
        else:
            new_width, new_height = self._smart_resize(
                image,
                downsample_factor,
                min_image_tokens,
                max_image_tokens,
                encoder_patch_size,
            )
            image = image.resize((new_width, new_height), resample=resample)
            return [image]

    def _preprocess(
        self,
        images: list[list[ImageInput]],
        downsample_factor: int,
        min_tiles: int,
        max_tiles: int,
        use_thumbnail: bool,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
        tile_size: int,
        max_pixels_tolerance: float,
        resample: "PILImageResampling",
    ) -> list[list[Image.Image]]:
        image_inputs = []

        for sample_images in images:
            sample_tiles = []
            for image in sample_images:
                image_tiles = self._resize_and_maybe_split(
                    image,
                    downsample_factor,
                    min_tiles,
                    max_tiles,
                    use_thumbnail,
                    min_image_tokens,
                    max_image_tokens,
                    encoder_patch_size,
                    tile_size,
                    max_pixels_tolerance,
                    resample,
                )
                sample_tiles.extend(image_tiles)
            image_inputs.append(sample_tiles)

        return image_inputs

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        downsample_factor: Optional[int] = None,
        do_image_splitting: Optional[bool] = None,
        min_tiles: Optional[int] = None,
        max_tiles: Optional[int] = None,
        use_thumbnail: Optional[bool] = None,
        min_image_tokens: Optional[int] = None,
        max_image_tokens: Optional[int] = None,
        encoder_patch_size: Optional[int] = None,
        tile_size: Optional[int] = None,
        max_pixels_tolerance: Optional[float] = None,
        resample: Optional["PILImageResampling"] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        do_convert_rgb: Optional[bool] = None,
        patch_size: Optional[int] = None,
    ) -> "Image.Image":
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            downsample_factor (`int`, *optional*, defaults to 2):
                The downsample_factor factor of the vision backbone.
            min_image_tokens (`int`, *optional*, defaults to 64):
                The minimum number of image tokens for smart resize.
            max_image_tokens (`int`, *optional*, defaults to 256):
                The maximum number of image tokens for smart resize.
            encoder_patch_size (`int`, *optional*, defaults to 16):
                The patch size of the encoder.
            do_image_splitting (`bool`, *optional*, defaults to `True`):
                Whether to split large images into tiles.
            min_tiles (`int`, *optional*, defaults to 2):
                The minimum number of tiles to split the image into.
            max_tiles (`int`, *optional*, defaults to 10):
                The maximum number of tiles to split the image into.
            tile_size (`int`, *optional*, defaults to 512):
                The size of the tile to split the image into.
            max_pixels_tolerance (`float`, *optional*, defaults to 2.0):
                The maximum tolerance for the number of pixels in the image before splitting.
            use_thumbnail (`bool`, *optional*, defaults to `True`):
                Whether to append the thumbnail of the image when splitting.
            resample (`int`, *optional*, defaults to `self.resample`):
                Siglip2ImageProcessor's `resample` parameter. Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Siglip2ImageProcessor's `do_rescale` parameter. Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Siglip2ImageProcessor's `rescale_factor` parameter. Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Siglip2ImageProcessor's `do_normalize` parameter. Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Siglip2ImageProcessor's `image_mean` parameter. Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Siglip2ImageProcessor's `image_std` parameter. Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                Siglip2ImageProcessor's `return_tensors` parameter. The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                Siglip2ImageProcessor's `input_data_format` parameter. The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Siglip2ImageProcessor's `do_convert_rgb` parameter. Whether to convert the image to RGB.
            max_num_patches (`int`, *optional*, defaults to `self.max_num_patches`):
                Siglip2ImageProcessor's `max_num_patches` parameter. Maximum number of patches per image, the image will be resized to have at most this number of patches.
        """
        downsample_factor = downsample_factor if downsample_factor is not None else self.downsample_factor
        do_image_splitting = do_image_splitting if do_image_splitting is not None else self.do_image_splitting

        min_tiles = min_tiles if min_tiles is not None else self.min_tiles
        max_tiles = max_tiles if max_tiles is not None else self.max_tiles

        if not do_image_splitting:
            min_tiles = 1
            max_tiles = 1
            logger.debug(
                "Image splitting is disabled, setting min_tiles and max_tiles to 1. Set do_image_splitting=True to enable splitting."
            )

        if do_image_splitting and min_tiles > max_tiles:
            raise ValueError("min_tiles must be less than or equal to max_tiles")

        use_thumbnail = use_thumbnail if use_thumbnail is not None else self.use_thumbnail
        min_image_tokens = min_image_tokens if min_image_tokens is not None else self.min_image_tokens
        max_image_tokens = max_image_tokens if max_image_tokens is not None else self.max_image_tokens
        encoder_patch_size = encoder_patch_size if encoder_patch_size is not None else self.encoder_patch_size
        tile_size = tile_size if tile_size is not None else self.tile_size
        max_pixels_tolerance = max_pixels_tolerance if max_pixels_tolerance is not None else self.max_pixels_tolerance

        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        max_thumbnail_image_patches = max_image_tokens * downsample_factor**2
        tile_size_patches = (tile_size // encoder_patch_size) ** 2 if do_image_splitting else 0
        max_num_patches = max(
            max_thumbnail_image_patches,
            tile_size_patches,
        )

        images = make_nested_list_of_images(images)
        images = self._preprocess(
            images,
            downsample_factor=downsample_factor,
            min_tiles=min_tiles,
            max_tiles=max_tiles,
            use_thumbnail=use_thumbnail,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
            encoder_patch_size=encoder_patch_size,
            tile_size=tile_size,
            max_pixels_tolerance=max_pixels_tolerance,
            resample=resample,
        )

        # copied from Siglip2ImageProcessor
        data_format = ChannelDimension.LAST

        images = make_flat_list_of_images(images)

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )
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

        pixel_masks = []
        pixel_values = []
        spatial_shapes = []

        for image in images:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=data_format)

            if do_normalize:
                image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=data_format)

            patches = convert_image_to_patches(image, encoder_patch_size)
            patches, mask = pad_along_first_dim(patches, max_num_patches)
            num_patches_height = image.shape[0] // encoder_patch_size
            num_patches_width = image.shape[1] // encoder_patch_size

            spatial_shapes.append((num_patches_height, num_patches_width))
            pixel_values.append(patches)
            pixel_masks.append(mask)

        batch_feature = BatchFeature(
            data={
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_masks,
                "spatial_shapes": spatial_shapes,
            },
            tensor_type=return_tensors,
        )

        return batch_feature

    def get_tile_grid_and_sizes(
        self,
        image: Image.Image,
        kwargs: Optional[dict] = None,
    ) -> tuple[int, int, int, int]:
        """Get the tile grid and sizes for a given image."""
        downsample_factor = kwargs.get("downsample_factor", self.downsample_factor)
        do_image_splitting = kwargs.get("do_image_splitting", self.do_image_splitting)
        min_tiles = kwargs.get("min_tiles", self.min_tiles)
        max_tiles = kwargs.get("max_tiles", self.max_tiles)

        if not do_image_splitting:
            min_tiles = 1
            max_tiles = 1
            logger.debug(
                "Image splitting is disabled, setting min_tiles and max_tiles to 1. Set do_image_splitting=True to enable splitting."
            )

        if do_image_splitting and min_tiles > max_tiles:
            raise ValueError("min_tiles must be less than or equal to max_tiles")

        use_thumbnail = kwargs.get("use_thumbnail", self.use_thumbnail)
        min_image_tokens = kwargs.get("min_image_tokens", self.min_image_tokens)
        max_image_tokens = kwargs.get("max_image_tokens", self.max_image_tokens)
        encoder_patch_size = kwargs.get("encoder_patch_size", self.encoder_patch_size)
        tile_size = kwargs.get("tile_size", self.tile_size)
        max_pixels_tolerance = kwargs.get("max_pixels_tolerance", self.max_pixels_tolerance)

        do_image_splitting = not min_tiles == max_tiles == 1
        if (
            self._is_img_too_large(
                image,
                max_image_tokens,
                encoder_patch_size,
                downsample_factor,
                max_pixels_tolerance,
            )
            and do_image_splitting
        ):
            num_rows, num_cols, _, _, _ = self._get_grid_layout(image, min_tiles, max_tiles, tile_size)
            num_thumbnail_tokens = 0
            if use_thumbnail:
                thumbnail_width, thumbnail_height = self._smart_resize(
                    image,
                    downsample_factor,
                    min_image_tokens,
                    max_image_tokens,
                    encoder_patch_size,
                )
                num_thumbnail_tokens = self._get_tokens_num(
                    thumbnail_width, thumbnail_height, downsample_factor, encoder_patch_size
                )
            return (
                self._get_tokens_num(tile_size, tile_size, downsample_factor, encoder_patch_size),
                num_rows,
                num_cols,
                num_thumbnail_tokens,
            )

        else:
            new_width, new_height = self._smart_resize(
                image,
                downsample_factor,
                min_image_tokens,
                max_image_tokens,
                encoder_patch_size,
            )
            return self._get_tokens_num(new_width, new_height, downsample_factor, encoder_patch_size), 1, 1, 0


__all__ = ["Lfm2VlImageProcessor"]
