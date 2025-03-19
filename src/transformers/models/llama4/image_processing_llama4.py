# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

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
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import (
    TensorType,
    filter_out_non_signature_kwargs,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)


if is_vision_available():
    import PIL
if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

logger = logging.get_logger(__name__)


def split_to_tiles_tensor(images: "torch.Tensor", num_tiles_height: int, num_tiles_width: int) -> "torch.Tensor":
    # Split image into number of required tiles (width x height)
    num_channels, height, width = images.size()
    images = images.view(
        num_channels, num_tiles_height, height // num_tiles_height, num_tiles_width, width // num_tiles_width
    )
    # Permute dimensions to reorder the axes
    image = images.permute(1, 3, 0, 2, 4).contiguous()
    # Reshape into the desired output shape (batch_size * 4, num_channels, width/2, height/2)
    image = image.view(
        num_tiles_width * num_tiles_height, num_channels, height // num_tiles_height, width // num_tiles_width
    )
    return image


def split_to_tiles(image: np.ndarray, num_tiles_height: int, num_tiles_width: int) -> np.ndarray:
    # Split image into number of required tiles (width x height)
    num_channels, height, width = image.shape
    image = image.reshape(
        num_channels, num_tiles_height, height // num_tiles_height, num_tiles_width, width // num_tiles_width
    )
    # Permute dimensions to reorder the axes
    image = image.transpose(1, 3, 0, 2, 4)
    # Reshape into the desired output shape (batch_size * 4, num_channels, width/2, height/2)
    image = image.reshape(
        num_tiles_width * num_tiles_height, num_channels, height // num_tiles_height, width // num_tiles_width
    )
    return image


def get_factors(dividend: int) -> Set[int]:
    """
    Calculate all factors of a given number, i.e. a dividor that leaves
    no remainder. For example, if dividend=12, it will return {1, 2, 3, 4, 6, 12}.

    Args:
        dividend (int): The number to find factors for.

    Returns:
        set: A set containing all factors of the number.
    """
    factors_set = set()

    for i in range(1, int(dividend**0.5) + 1):
        if dividend % i == 0:
            factors_set.add(i)
            factors_set.add(dividend // i)
    return factors_set


def find_supported_resolutions(max_num_chunks: int, patch_size: dict) -> np.ndarray:
    """
    Computes all of the allowed resolutions for a fixed number of chunks
    and patch_size. Useful for when dividing an image into chunks.

    Args:
        max_num_chunks (int): Maximum number of chunks for processing.
        patch_size (int): Size of the side of the patch.

    Returns:
        torch.Tensor: List of possible resolutions as tuples (height, width).

    Example:
        >>> max_num_chunks = 5
        >>> patch_size = 224
        >>> find_supported_resolutions(max_num_chunks, patch_size)
        tensor([(224, 896), (448, 448), (224, 224), (896, 224), (224, 672),
        (672, 224), (224, 448), (448, 224)])

        Given max_num_chunks=4, patch_size=224, it will create a dictionary:
        {
        0.25: [(1, 4)],
        1.0: [(2, 2), (1, 1)],
        4.0: [(4, 1)],
        0.33: [(1, 3)],
        3.0: [(3, 1)],
        0.5: [(1, 2)],
        2.0: [(2, 1)]
        }

        and return the resolutions multiplied by the patch_size:
        [(1*224, 4*224), (2*224, 2*224), ..., (2*224, 1*224)]
    """
    height, width = patch_size["height"], patch_size["width"]
    if height != width:
        raise ValueError("`size` must be square.")

    patch_size = height

    asp_dict = defaultdict(list)
    for chunk_size in range(max_num_chunks, 0, -1):
        _factors = sorted(get_factors(chunk_size))
        _asp_ratios = [(factor, chunk_size // factor) for factor in _factors]
        for height, width in _asp_ratios:
            ratio_float = height / width
            asp_dict[ratio_float].append((height, width))

    # get the resolutions multiplied by the patch_size
    possible_resolutions = []
    for key, value in asp_dict.items():
        for height, depth in value:
            possible_resolutions.append((height * patch_size, depth * patch_size))

    return possible_resolutions


def pad_to_best_fit(image: np.ndarray, target_size) -> np.ndarray:
    new_height, new_width = target_size
    bottom_padding = new_height - image.shape[1]
    right_padding = new_width - image.shape[2]
    image = np.pad(
        image,
        ((0, 0), (0, bottom_padding), (0, right_padding)),
        mode="constant",
        constant_values=0,
    )
    return image


def get_max_res_without_distortion(
    image_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Determines the maximum resolution to which an image can be resized to without distorting its
    aspect ratio, based on the target resolution.

    Args:
        image_size (Tuple[int, int]): The original resolution of the image (height, width).
        target_resolution (Tuple[int, int]): The desired resolution to fit the image into (height, width).
    Returns:
        Tuple[int, int]: The optimal dimensions (height, width) to which the image should be resized.
    Example:
        >>> _get_max_res_without_distortion([200, 300], target_size = [450, 200])
        (134, 200)
        >>> _get_max_res_without_distortion([800, 600], target_size = [450, 1300])
        (450, 338)
    """

    original_height, original_width = image_size
    target_height, target_width = target_size

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.floor(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.floor(original_width * scale_h), target_width)

    return new_height, new_width


def get_best_fit(
    image_size: Tuple[int, int],
    possible_resolutions: np.ndarray,
    resize_to_max_canvas: bool = False,
) -> Tuple[int, int]:
    """
    Determines the best canvas possible from a list of possible resolutions to, without distortion,
    resize an image to.

    For each possible resolution, calculates the scaling factors for
    width and height, and selects the smallest one, which is the limiting side.
    E.g. to match the canvas you can upscale height by 2x, and width by 1.5x,
    therefore, the maximum upscaling you can do is min(2, 1.5) = 1.5.

    If upscaling is possible (any of the scaling factors is greater than 1),
    then picks the smallest upscaling factor > 1, unless resize_to_max_canvas is True.

    If upscaling is not possible, then picks the largest scaling factor <= 1, i.e.
    reduce downscaling as much as possible.

    If there are multiple resolutions with the same max scale, we pick the one with the lowest area,
    to minimize padding. E.g., the same image can be upscaled to 224x224 and 224x448, but the latter
    has more padding.

    Args:
        image_size (Tuple[int, int]): A tuple containing the height and width of the image.
        possible_resolutions (np.ndarray): A tensor of shape (N, 2) where each
            row represents a possible resolution (height, width).
        resize_to_max_canvas (bool): If True, will return the largest upscaling resolution.

    Returns:
        List[int]: The best resolution [height, width] for the given image.

    Example:
        >>> image_size = (200, 300)
        >>> possible_resolutions = torch.tensor([[224, 672],
        ...                                     [672, 224],
        ...                                     [224, 448],
        ...                                     [448, 224],
        ...                                     [224, 224]])
        >>> get_best_fit(image_size, possible_resolutions)
        [224, 448]

        We have:
            scale_w = tensor([2.2400, 0.7467, 1.4933, 0.7467, 0.7467])
            scale_h = tensor([1.1200, 3.3600, 1.1200, 2.2400, 1.1200])
            scales = tensor([1.1200, 0.7467, 1.1200, 0.7467, 0.7467])
        Only one of the scales > 1:
            upscaling_possible = tensor([1.1200, 1.1200])
            smallest_rescale = tensor(1.1200)
        So we pick the resolution with the smallest smallest area:
            areas = tensor([150528, 100352]) # [672, 224], [224, 448]
            optimal_canvas = tensor([224, 448])
    """

    original_height, original_width = image_size

    # get all possible resolutions heights/widths
    target_heights, target_widths = (
        possible_resolutions[:, 0],
        possible_resolutions[:, 1],
    )

    # get scaling factors to resize the image without distortion
    scale_w = target_widths / original_width
    scale_h = target_heights / original_height

    # get the min scale between width and height (limiting side -> no distortion)
    scales = np.where(scale_h > scale_w, scale_w, scale_h)

    # filter only scales that allow upscaling
    upscaling_options = scales[scales >= 1]
    if len(upscaling_options) > 0:
        if resize_to_max_canvas:
            selected_scale = np.max(upscaling_options)
        else:
            selected_scale = np.min(upscaling_options)
    else:
        # no upscaling possible,
        # get the minimum downscaling (max scale for scales<1)
        downscaling_options = scales[scales < 1]
        selected_scale = np.max(downscaling_options)

    # get all resolutions that support this scaling factor,
    # e.g. you can upscale to 224x224, 224x448, 224x672 without distortion
    chosen_canvas = possible_resolutions[scales == selected_scale]

    # if there are multiple resolutions,
    # get the one with minimum area to reduce padding
    if len(chosen_canvas) > 1:
        areas = chosen_canvas[:, 0] * chosen_canvas[:, 1]
        optimal_idx = np.argmin(areas)
        optimal_canvas = chosen_canvas[optimal_idx]
    else:
        optimal_canvas = chosen_canvas[0]

    return tuple(optimal_canvas.tolist())


class Llama4ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Llama4 image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 336, "width": 336}`):
            Size of the image tiles. Can be overridden by the `size` parameter in the `preprocess`
            method.
        max_patches (`int`, *optional*, defaults to 16):
            The maximum number of patches to be extracted from the image.
            Can be overridden by the `max_patches` parameter in the `preprocess` method.
        resize_to_max_canvas (`bool`, *optional*, defaults to False):
            Whether to resize the image to the maximum canvas size.
            If True, picks the canvas the allows the largest resizing without distortion.
            If False, downsample as little as possible, including no resizing at all,
            but never upsample, unless the image is smaller than the patch size.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
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
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        max_patches: int = 16,
        resize_to_max_canvas: bool = False,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 336, "width": 336}
        size = get_size_dict(size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.max_patches = max_patches
        self.resize_to_max_canvas = resize_to_max_canvas
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
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
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
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
        size: Optional[Dict[str, int]] = None,
        max_patches: Optional[int] = None,
        resize_to_max_canvas: bool = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_convert_rgb: bool = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            max_patches (`int`, *optional*, defaults to 16):
                The maximum number of patches to be extracted from the image.
                Can be overridden by the `max_patches` parameter in the `preprocess` method.
            resize_to_max_canvas (`bool`, *optional*, defaults to False):
                Whether to resize the image to the maximum canvas size.
                If True, picks the canvas the allows the largest resizing without distortion.
                If False, downsample as little as possible, including no resizing at all,
                but never upsample, unless the image is smaller than the patch size.
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
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        max_patches = max_patches if max_patches is not None else self.max_patches
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        resize_to_max_canvas = resize_to_max_canvas if resize_to_max_canvas is not None else self.resize_to_max_canvas

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

        if input_data_format == ChannelDimension.LAST:
            # We force the channel dimension to be first for torch tensors as this is what torchvision expects.
            images = [image.transpose(2, 0, 1) for image in images]
            input_data_format = ChannelDimension.FIRST

        possible_resolutions = find_supported_resolutions(max_num_chunks=max_patches, patch_size=size)
        possible_resolutions = np.array(possible_resolutions)
        aspect_ratios = []
        processed_images = []
        for image in images:
            image_size = image.shape[-2:]
            target_size = get_best_fit(image_size, possible_resolutions, resize_to_max_canvas=resize_to_max_canvas)
            # If target_size requires upscaling, we might want to limit the upscaling to max_upscaling_size
            max_upscaling_size = None if resize_to_max_canvas else size["height"]
            if max_upscaling_size is not None:
                new_target_height = min(max(image_size[0], max_upscaling_size), target_size[0])
                new_target_width = min(max(image_size[1], max_upscaling_size), target_size[1])
                target_size_without_distortion = (new_target_height, new_target_width)

            # resize to target_size while preserving aspect ratio
            new_size_without_distortion = get_max_res_without_distortion(image_size, target_size_without_distortion)
            new_size_without_distortion = {
                "height": max(new_size_without_distortion[0], 1),
                "width": max(new_size_without_distortion[1], 1),
            }
            processed_image = self.resize(
                image, new_size_without_distortion, resample=resample, input_data_format=input_data_format
            )

            # pad to target_size to be able to split into tiles
            processed_image = pad_to_best_fit(processed_image, target_size)
            if is_torch_available() and is_torchvision_available():
                processed_image = torch.tensor(processed_image, dtype=torch.bfloat16)
                if do_rescale:
                    processed_image = processed_image * rescale_factor
                if do_normalize:
                    processed_image = F.normalize(processed_image, image_mean, image_std)
            else:
                if do_rescale:
                    processed_image = self.rescale(
                        image=processed_image, scale=rescale_factor, input_data_format=input_data_format
                    )
                if do_normalize:
                    processed_image = self.normalize(
                        image=processed_image, mean=image_mean, std=image_std, input_data_format=input_data_format
                    )

            ratio_h, ratio_w = (
                target_size[0] // size["height"],
                target_size[1] // size["height"],
            )
            # split into tiles
            if is_torch_available() and is_torchvision_available():
                processed_image = split_to_tiles_tensor(processed_image, ratio_h, ratio_w)
            else:
                processed_image = split_to_tiles(processed_image, ratio_h, ratio_w)

            aspect_ratios.append([ratio_h, ratio_w])

            # add a global tile to the processed tile if there are more than one tile
            if ratio_h * ratio_w > 1:
                global_tile = self.resize(image, size, resample=resample, input_data_format=input_data_format)
                if is_torch_available() and is_torchvision_available():
                    global_tile = torch.tensor(global_tile, dtype=torch.bfloat16)
                    if do_rescale:
                        global_tile = global_tile.div(255)
                    if do_normalize:
                        global_tile = F.normalize(global_tile, image_mean, image_std)
                    processed_image = torch.cat([processed_image, global_tile.unsqueeze(0)], dim=0)
                else:
                    if do_rescale:
                        global_tile = self.rescale(
                            image=global_tile, scale=rescale_factor, input_data_format=input_data_format
                        )
                    if do_normalize:
                        global_tile = self.normalize(
                            image=global_tile, mean=image_mean, std=image_std, input_data_format=input_data_format
                        )
                    processed_image = np.concatenate([processed_image, np.expand_dims(global_tile, axis=0)], axis=0)

            processed_images.append(processed_image)
        # flatten processed_images
        if is_torch_available() and is_torchvision_available():
            images = processed_images
        else:
            images = processed_images
            images = [
                [to_channel_dimension_format(chunk, data_format, input_channel_dim=input_data_format) for chunk in image] for image in images
            ]

        return BatchFeature(data={"pixel_values": images, "aspect_ratios": aspect_ratios}, tensor_type=return_tensors)


__all__ = ["Llama4ImageProcessor"]
