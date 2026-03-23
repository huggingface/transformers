# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Image processor class for DeepSeek-OCR-2."""

from functools import lru_cache

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...processing_utils import ImagesKwargs
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


class DeepseekOcr2ImageProcessorKwargs(ImagesKwargs, total=False):
    """
    crop_to_patches (`bool`, *optional*, defaults to `True`):
        Whether to crop the image into local patches. When `False`, only the global view is produced.
        Can be overridden by the `crop_to_patches` parameter in the `preprocess` method.
    min_patches (`int`, *optional*, defaults to `2`):
        The minimum number of patches to extract from the image for the local view.
        Only has an effect if `crop_to_patches` is set to `True`.
        Can be overridden by the `min_patches` parameter in the `preprocess` method.
    max_patches (`int`, *optional*, defaults to `6`):
        The maximum number of patches to extract from the image for the local view.
        Only has an effect if `crop_to_patches` is set to `True`.
        Can be overridden by the `max_patches` parameter in the `preprocess` method.
    """

    crop_to_patches: bool
    min_patches: int
    max_patches: int


class DeepseekOcr2ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a DeepSeek-OCR-2 image processor.

    This processor handles dual-view image processing:
    - **Global view**: Pads the image to a square of `size` x `size`.
    - **Local view**: Crops the image into a grid of 768 x 768 tiles (fixed tile size),
      with the number of tiles determined by the image's aspect ratio.

    When `crop_to_patches=True` and the image is larger than 768px, both views are produced.
    When `crop_to_patches=False` or the image is small, only the global view is produced at 768x768.

    Args:
        crop_to_patches (`bool`, *optional*, defaults to `True`):
            Whether to crop the image into local patches. When `False`, only the global view is produced.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict[str, int]`, *optional*, defaults to `{"height": 1024, "width": 1024}`):
            Size of the global view image. When cropping, the image is padded to this size.
            When not cropping, this is overridden to `tile_size` x `tile_size`.
            Can be overridden by the `size` parameter in the `preprocess` method.
        tile_size (`int`, *optional*, defaults to `768`):
            The size of each local tile. Must match the model's query embedding size (e.g. 768 for query_768).
        min_patches (`int`, *optional*, defaults to `2`):
            The minimum number of patches to extract from the image for the local view.
            Only has an effect if `crop_to_patches` is set to `True`.
        max_patches (`int`, *optional*, defaults to `6`):
            The maximum number of patches to extract from the image for the local view.
            Only has an effect if `crop_to_patches` is set to `True`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.LANCZOS`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Mean to use if normalizing the image. Can be overridden by the `image_mean` parameter in the `preprocess`
            method.
        image_std (`float` or `list[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Standard deviation to use if normalizing the image. Can be overridden by the `image_std` parameter in
            the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values", "pixel_values_local"]
    valid_kwargs = DeepseekOcr2ImageProcessorKwargs

    def __init__(
        self,
        crop_to_patches: bool = True,
        do_resize: bool = True,
        size: dict[str, int] | None = None,
        tile_size: int = 768,
        min_patches: int = 2,
        max_patches: int = 6,
        resample: PILImageResampling = PILImageResampling.LANCZOS,
        do_rescale: bool = True,
        rescale_factor: int | float = 1 / 255,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 1024, "width": 1024}
        size = get_size_dict(size, default_to_square=True)

        self.crop_to_patches = crop_to_patches
        self.do_resize = do_resize
        self.size = size
        self.tile_size = tile_size
        self.min_patches = min_patches
        self.max_patches = max_patches
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb

    def crop_image_to_patches(
        self,
        images: np.ndarray,
        min_patches: int,
        max_patches: int,
        tile_size: tuple | int | dict | None = None,
        data_format: ChannelDimension | None = None,
    ):
        """
        Crop the image to patches and return a list of cropped images.
        The number of patches and their grid arrangement are determined by the original image size,
        the target tile size and the minimum and maximum number of patches.
        """
        if data_format is None:
            data_format = infer_channel_dimension_format(images)
        images = to_channel_dimension_format(images, ChannelDimension.FIRST, data_format)
        tile_size_height, tile_size_width = tile_size["height"], tile_size["width"]
        original_height, original_width = images.shape[-2:]

        num_columns, num_rows = get_optimal_tiled_canvas(
            (original_height, original_width), (tile_size_height, tile_size_width), min_patches, max_patches
        )

        target_width = tile_size_width * num_columns
        target_height = tile_size_height * num_rows
        num_blocks = num_columns * num_rows

        resized_image = self.resize(
            images,
            {"height": target_height, "width": target_width},
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

        processed_images = []
        for i in range(num_blocks):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * tile_size_width,
                row * tile_size_height,
                (column + 1) * tile_size_width,
                (row + 1) * tile_size_height,
            )
            patch_image = resized_image[..., box[1] : box[3], box[0] : box[2]]
            patch_image = to_channel_dimension_format(patch_image, data_format, ChannelDimension.FIRST)
            processed_images.append(patch_image)

        return processed_images, (num_columns, num_rows)

    # Same as deepseek_vl's pad_to_square
    def pad_to_square(
        self,
        image: np.ndarray,
        background_color: int | tuple[int, int, int] = 0,
        data_format: str | ChannelDimension | None = None,
        input_data_format: str | ChannelDimension | None = None,
    ) -> np.ndarray:
        """
        Pads an image to a square based on the longest edge.

        Args:
            image (`np.ndarray`):
                The image to pad.
            background_color (`int` or `tuple[int, int, int]`, *optional*, defaults to 0):
                The color to use for the padding.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image.

        Returns:
            `np.ndarray`: The padded image.
        """
        height, width = get_image_size(image, input_data_format)
        num_channels = image.shape[0] if input_data_format == ChannelDimension.FIRST else image.shape[-1]

        if height == width:
            image = (
                to_channel_dimension_format(image, data_format, input_data_format)
                if data_format is not None
                else image
            )
            return image

        max_dim = max(height, width)

        if isinstance(background_color, int):
            background_color = [background_color]
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )

        if input_data_format == ChannelDimension.FIRST:
            result = np.zeros((num_channels, max_dim, max_dim), dtype=image.dtype)
            for i, color in enumerate(background_color):
                result[i, :, :] = color
            if width > height:
                start = (max_dim - height) // 2
                result[:, start : start + height, :] = image
            else:
                start = (max_dim - width) // 2
                result[:, :, start : start + width] = image
        else:
            result = np.zeros((max_dim, max_dim, num_channels), dtype=image.dtype)
            for i, color in enumerate(background_color):
                result[:, :, i] = color
            if width > height:
                start = (max_dim - height) // 2
                result[start : start + height, :, :] = image
            else:
                start = (max_dim - width) // 2
                result[:, start : start + width, :] = image

        return result

    def resize(
        self,
        image: np.ndarray,
        size: dict[str, int],
        resample: PILImageResampling = PILImageResampling.LANCZOS,
        data_format: str | ChannelDimension | None = None,
        input_data_format: str | ChannelDimension | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.LANCZOS`):
                `PILImageResampling` filter to use when resizing the image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image.

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
        crop_to_patches: bool | None = None,
        do_resize: bool | None = None,
        size: dict[str, int] | None = None,
        min_patches: int | None = None,
        max_patches: int | None = None,
        resample: PILImageResampling | None = None,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        return_tensors: str | TensorType | None = None,
        do_convert_rgb: bool | None = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: str | ChannelDimension | None = None,
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images for DeepSeek-OCR-2.

        For each image, produces:
        - A global view padded to `size` x `size` (1024 when cropping, 768 when not)
        - Local tiles of 768 x 768 (only when `crop_to_patches=True` and image > 768px)

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255.
                If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            crop_to_patches (`bool`, *optional*, defaults to `self.crop_to_patches`):
                Whether to crop the image into local patches.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the global view image.
            min_patches (`int`, *optional*, defaults to `self.min_patches`):
                Minimum number of local patches.
            max_patches (`int`, *optional*, defaults to `self.max_patches`):
                Maximum number of local patches.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image.
        """
        crop_to_patches = crop_to_patches if crop_to_patches is not None else self.crop_to_patches
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=True)
        min_patches = min_patches if min_patches is not None else self.min_patches
        max_patches = max_patches if max_patches is not None else self.max_patches
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = self.fetch_images(images)
        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, or torch.Tensor")

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

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        all_pixel_values_local = []  # flat list of all local patches
        all_pixel_values_global = []  # global view per image
        num_local_patches = []  # number of local patches per image

        for image in images:
            image_np = to_numpy_array(image)
            if input_data_format is None:
                img_format = infer_channel_dimension_format(image_np)
            else:
                img_format = input_data_format

            if do_rescale and is_scaled_image(image_np):
                logger.warning_once(
                    "It looks like you are trying to rescale already rescaled images. If the input"
                    " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
                )

            original_height, original_width = get_image_size(image_np, channel_dim=img_format)

            # --- Local patches ---
            if crop_to_patches and max(original_width, original_height) > self.tile_size:
                tile_size_dict = {"height": self.tile_size, "width": self.tile_size}
                local_patches, (num_cols, num_rows) = self.crop_image_to_patches(
                    image_np,
                    min_patches=min_patches,
                    max_patches=max_patches,
                    tile_size=tile_size_dict,
                    data_format=img_format,
                )

                for patch_np in local_patches:
                    patch_fmt = infer_channel_dimension_format(patch_np)
                    if do_rescale:
                        patch_np = self.rescale(image=patch_np, scale=rescale_factor, input_data_format=patch_fmt)
                    if do_normalize:
                        patch_np = self.normalize(image=patch_np, mean=image_mean, std=image_std, input_data_format=patch_fmt)
                    patch_np = to_channel_dimension_format(patch_np, data_format, input_channel_dim=patch_fmt)
                    all_pixel_values_local.append(patch_np)

                num_local_patches.append(len(local_patches))
            else:
                num_local_patches.append(0)

            # Global view size: crop_to_patches=True → base size, False → tile_size
            global_target_size = size["height"] if crop_to_patches else self.tile_size

            # --- Global view ---
            scale = global_target_size / max(original_width, original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            global_np = resize(image_np, (new_height, new_width), resample=resample, input_data_format=img_format)

            global_fmt = infer_channel_dimension_format(global_np)
            global_np = self.pad_to_square(global_np, background_color=(127, 127, 127), input_data_format=global_fmt)

            if do_rescale:
                global_np = self.rescale(image=global_np, scale=rescale_factor, input_data_format=global_fmt)
            if do_normalize:
                global_np = self.normalize(image=global_np, mean=image_mean, std=image_std, input_data_format=global_fmt)
            global_np = to_channel_dimension_format(global_np, data_format, input_channel_dim=global_fmt)

            all_pixel_values_global.append(global_np)

        data = {
            "pixel_values": all_pixel_values_global,
            "num_local_patches": num_local_patches,
        }
        if all_pixel_values_local:
            data["pixel_values_local"] = all_pixel_values_local

        encoded_outputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_outputs


__all__ = ["DeepseekOcr2ImageProcessor"]
