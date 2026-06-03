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

import math
from collections import defaultdict
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
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)


def get_factors(dividend: int) -> set[int]:
    """
    Calculate all factors of a given number, i.e. a divisor that leaves
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


def get_max_res_without_distortion(
    image_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int]:
    """
    Determines the maximum resolution to which an image can be resized to without distorting its
    aspect ratio, based on the target resolution.

    Args:
        image_size (tuple[int, int]): The original resolution of the image (height, width).
        target_resolution (tuple[int, int]): The desired resolution to fit the image into (height, width).
    Returns:
        tuple[int, int]: The optimal dimensions (height, width) to which the image should be resized.
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


def split_to_tiles(images: torch.Tensor, num_tiles_height: int, num_tiles_width: int) -> torch.Tensor:
    # Split image into number of required tiles (width x height)
    batch_size, num_channels, height, width = images.size()
    images = images.view(
        batch_size,
        num_channels,
        num_tiles_height,
        height // num_tiles_height,
        num_tiles_width,
        width // num_tiles_width,
    )
    # Permute dimensions to reorder the axes
    image = images.permute(0, 2, 4, 1, 3, 5).contiguous()
    # Reshape into the desired output shape (batch_size * 4, num_channels, width/2, height/2)
    image = image.view(
        batch_size,
        num_tiles_width * num_tiles_height,
        num_channels,
        height // num_tiles_height,
        width // num_tiles_width,
    )
    return image


@lru_cache(maxsize=1)
def find_supported_resolutions(max_num_chunks: int, patch_size: SizeDict) -> torch.Tensor:
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
    height, width = patch_size.height, patch_size.width
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
    for value in asp_dict.values():
        for height, depth in value:
            possible_resolutions.append((height * patch_size, depth * patch_size))

    return possible_resolutions


def pad_to_best_fit(
    images: "torch.Tensor",
    target_size: tuple[int, int],
    background_color: Union[int, tuple[int, int, int]] = 0,
) -> "torch.Tensor":
    """
    Pads an image to fit the target size.

    Args:
        images (`np.ndarray`):
            The images to pad.
        background_color (`int` or `tuple[int, int, int]`, *optional*, defaults to 0):
            The color to use for the padding. Can be an integer for single channel or a
            tuple of integers representing for multi-channel images. If passed as integer
            in multi-channel mode, it will default to `0` in subsequent channels.
    Returns:
        `torch.Tensor`: The padded images.
    """

    num_channels = images.shape[1] if len(images.shape) == 4 else images.shape[0]
    if isinstance(background_color, int):
        background_color = [background_color] + [0] * (num_channels - 1)
    elif len(background_color) != num_channels:
        raise ValueError(
            f"background_color must have no more than {num_channels} elements to match the number of channels"
        )

    height, width = images.shape[-2:]
    target_height, target_width = target_size
    paste_x_right = target_width - width
    paste_y_right = target_height - height
    padded_images = F.pad(images, padding=[0, 0, paste_x_right, paste_y_right], fill=background_color)

    return padded_images


def get_best_fit(
    image_size: tuple[int, int],
    possible_resolutions: torch.Tensor,
    resize_to_max_canvas: bool = False,
) -> tuple[int, int]:
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
        image_size (tuple[int, int]): A tuple containing the height and width of the image.
        possible_resolutions (torch.Tensor): A tensor of shape (N, 2) where each
            row represents a possible resolution (height, width).
        resize_to_max_canvas (bool): If True, will return the largest upscaling resolution.

    Returns:
        list[int]: The best resolution [height, width] for the given image.

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
    scales = torch.where(scale_h > scale_w, scale_w, scale_h)

    # filter only scales that allow upscaling
    upscaling_options = scales[scales >= 1]
    if len(upscaling_options) > 0:
        if resize_to_max_canvas:
            selected_scale = torch.max(upscaling_options)
        else:
            selected_scale = torch.min(upscaling_options)
    else:
        # no upscaling possible,
        # get the minimum downscaling (max scale for scales<1)
        downscaling_options = scales[scales < 1]
        selected_scale = torch.max(downscaling_options)

    # get all resolutions that support this scaling factor,
    # e.g. you can upscale to 224x224, 224x448, 224x672 without distortion
    chosen_canvas = possible_resolutions[scales == selected_scale]

    # if there are multiple resolutions,
    # get the one with minimum area to reduce padding
    if len(chosen_canvas) > 1:
        areas = chosen_canvas[:, 0] * chosen_canvas[:, 1]
        optimal_idx = torch.argmin(areas)
        optimal_canvas = chosen_canvas[optimal_idx]
    else:
        optimal_canvas = chosen_canvas[0]

    return optimal_canvas


class Llama4ImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    max_patches (`int`, *optional*, defaults to 16):
        The maximum number of patches to be extracted from the image.
        Can be overridden by the `max_patches` parameter in the `preprocess` method.
    resize_to_max_canvas (`bool`, *optional*, defaults to False):
        Whether to resize the image to the maximum canvas size.
        If True, picks the canvas the allows the largest resizing without distortion.
        If False, downsample as little as possible, including no resizing at all,
        but never upsample, unless the image is smaller than the patch size.
    """

    max_patches: Optional[int]
    resize_to_max_canvas: Optional[bool]


@auto_docstring
class Llama4ImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    size = {"height": 336, "width": 336}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    max_patches = 16
    resize_to_max_canvas = False
    valid_kwargs = Llama4ImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[Llama4ImageProcessorKwargs]):
        super().__init__(**kwargs)

    # Disable compilation here as conversion to bfloat16 causes differences in the output of the compiled and non-compiled versions
    @torch.compiler.disable
    def rescale_and_normalize(
        self,
        images: "torch.Tensor",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, list[float]],
        image_std: Union[float, list[float]],
    ) -> "torch.Tensor":
        """
        Rescale and normalize images.
        Override to rescale and normalize the images in torch.bfloat16 as in the original implementation
        """
        if do_rescale and do_normalize:
            images = images.to(dtype=torch.bfloat16) * rescale_factor
            images = self.normalize(images, image_mean, image_std)
        elif do_rescale:
            images = images * rescale_factor
        elif do_normalize:
            images = self.normalize(images, image_mean, image_std)

        return images

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Llama4ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        size: SizeDict,
        max_patches: int,
        resize_to_max_canvas: bool,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        possible_resolutions = find_supported_resolutions(max_num_chunks=max_patches, patch_size=size)
        possible_resolutions = torch.tensor(possible_resolutions, device=images[0].device)
        # process images by batch, grouped by shape
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        grouped_processed_images = {}
        grouped_aspect_ratios = {}
        for shape, stacked_images in grouped_images.items():
            image_size = stacked_images.shape[-2:]
            target_size = get_best_fit(image_size, possible_resolutions, resize_to_max_canvas=resize_to_max_canvas)
            # If target_size requires upscaling, we might want to limit the upscaling to max_upscaling_size
            max_upscaling_size = None if resize_to_max_canvas else size.height
            if max_upscaling_size is not None:
                new_target_height = min(max(image_size[0], max_upscaling_size), target_size[0])
                new_target_width = min(max(image_size[1], max_upscaling_size), target_size[1])
                target_size_without_distortion = (new_target_height, new_target_width)

            # resize to target_size while preserving aspect ratio
            new_size_without_distortion = get_max_res_without_distortion(image_size, target_size_without_distortion)
            new_size_without_distortion = SizeDict(
                height=max(new_size_without_distortion[0], 1), width=max(new_size_without_distortion[1], 1)
            )
            processed_images = self.resize(
                stacked_images,
                new_size_without_distortion,
                interpolation=interpolation,
            )

            # pad to target_size to be able to split into tiles
            processed_images = pad_to_best_fit(processed_images, target_size)
            processed_images = self.rescale_and_normalize(
                processed_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            ratio_h, ratio_w = (
                target_size[0] // size.height,
                target_size[1] // size.width,
            )
            # split into tiles
            processed_images = split_to_tiles(processed_images, ratio_h, ratio_w)
            grouped_processed_images[shape] = processed_images
            grouped_aspect_ratios[shape] = torch.tensor(
                [[ratio_h, ratio_w]] * stacked_images.shape[0], device=images[0].device
            )

            # add a global tile to the processed tile if there are more than one tile
            if ratio_h * ratio_w > 1:
                global_tiles = self.resize(
                    stacked_images,
                    size,
                    interpolation=interpolation,
                )
                global_tiles = self.rescale_and_normalize(
                    global_tiles, do_rescale, rescale_factor, do_normalize, image_mean, image_std
                )
                grouped_processed_images[shape] = torch.cat([processed_images, global_tiles.unsqueeze(1)], dim=1)
        processed_images = reorder_images(grouped_processed_images, grouped_images_index)
        aspect_ratios_list = reorder_images(grouped_aspect_ratios, grouped_images_index)

        processed_images = torch.cat(processed_images, dim=0) if return_tensors else processed_images
        aspect_ratios = torch.stack(aspect_ratios_list, dim=0) if return_tensors else aspect_ratios_list
        return BatchFeature(
            data={"pixel_values": processed_images, "aspect_ratios": aspect_ratios}, tensor_type=return_tensors
        )


__all__ = ["Llama4ImageProcessorFast"]
