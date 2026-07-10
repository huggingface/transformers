# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights reserved.
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
"""Image processor for Cosmos3 Edge."""

import math
from collections.abc import Iterable

import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


class Cosmos3EdgeImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    min_pixels (`int`, *optional*, defaults to `65536`):
        Minimum number of pixels in a resized image.
    max_pixels (`int`, *optional*, defaults to `16777216`):
        Maximum number of pixels in a resized image.
    patch_size (`int`, *optional*, defaults to `16`):
        Spatial patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to `2`):
        Number of adjacent patches merged along each spatial axis by the projector.
    per_image_kwargs (`list[dict]`, *optional*):
        Per-image overrides for `min_pixels` and `max_pixels`.
    """

    min_pixels: int
    max_pixels: int
    patch_size: int
    merge_size: int
    per_image_kwargs: list[dict | None]


def round_by_factor(number: int, factor: int) -> int:
    """Return the closest multiple of ``factor`` to ``number``."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Return the smallest multiple of ``factor`` greater than or equal to ``number``."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Return the largest multiple of ``factor`` less than or equal to ``number``."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 256 * 256,
    max_pixels: int = 4096 * 4096,
) -> tuple[int, int]:
    """Resize dimensions while preserving aspect ratio and the model's patch alignment.

    Cosmos3 Edge receives unpadded SigLIP2 patches. Therefore, both output dimensions
    must be divisible by the spatial projector factor (``patch_size * merge_size``).
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got height={height}, width={width}.")
    if min_pixels <= 0 or max_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError(
            "`min_pixels` and `max_pixels` must be positive and `max_pixels` must be greater than or equal to "
            f"`min_pixels`, got min_pixels={min_pixels}, max_pixels={max_pixels}."
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    resized_height = max(factor, round_by_factor(height, factor))
    resized_width = max(factor, round_by_factor(width, factor))

    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, floor_by_factor(height / beta, factor))
        resized_width = max(factor, floor_by_factor(width / beta, factor))
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = ceil_by_factor(height * beta, factor)
        resized_width = ceil_by_factor(width * beta, factor)

    return resized_height, resized_width


@auto_docstring
class Cosmos3EdgeImageProcessor(TorchvisionBackend):
    """Dynamically resize images and return packed, unpadded SigLIP2 patches."""

    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 256 * 256, "longest_edge": 4096 * 4096}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_convert_rgb = True
    patch_size = 16
    merge_size = 2
    valid_kwargs = Cosmos3EdgeImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[Cosmos3EdgeImageProcessorKwargs]):
        size = kwargs.pop("size", None)
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)

        size = dict(self.size) if size is None else dict(size)
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("`size` must contain `shortest_edge` and `longest_edge` keys.")

        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(
        self,
        size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None or max_pixels is not None:
            current_size = self.size if size is None else size
            if isinstance(current_size, SizeDict):
                current_min_pixels = current_size.shortest_edge
                current_max_pixels = current_size.longest_edge
            else:
                current_min_pixels = current_size["shortest_edge"]
                current_max_pixels = current_size["longest_edge"]
            size = SizeDict(
                shortest_edge=current_min_pixels if min_pixels is None else min_pixels,
                longest_edge=current_max_pixels if max_pixels is None else max_pixels,
            )

        kwargs = super()._standardize_kwargs(size=size, **kwargs)
        size = kwargs.get("size", self.size)
        if size.shortest_edge is None or size.longest_edge is None:
            raise ValueError("`size` must contain `shortest_edge` and `longest_edge` keys.")
        return kwargs

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Cosmos3EdgeImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        merge_size: int,
        return_tensors: str | TensorType | None,
        per_image_kwargs: list[dict | None] | None = None,
        **kwargs,
    ) -> BatchFeature:
        pixel_values = []
        image_grids = []

        for image_index, image in enumerate(images):
            image_kwargs = {}
            if per_image_kwargs is not None and image_index < len(per_image_kwargs):
                image_kwargs = per_image_kwargs[image_index] or {}

            height, width = image.shape[-2:]
            if do_resize:
                min_pixels = image_kwargs.get("min_pixels", size.shortest_edge)
                max_pixels = image_kwargs.get("max_pixels", size.longest_edge)
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                image = self.resize(
                    image=image,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            else:
                resized_height, resized_width = height, width
                patch_group_size = patch_size * merge_size
                if resized_height % patch_group_size or resized_width % patch_group_size:
                    raise ValueError(
                        "Images must have dimensions divisible by `patch_size * merge_size` when `do_resize=False`, "
                        f"got height={resized_height}, width={resized_width}, patch_size={patch_size}, "
                        f"merge_size={merge_size}."
                    )

            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            channels = image.shape[0]
            grid_height, grid_width = resized_height // patch_size, resized_width // patch_size
            patches = image.reshape(
                channels,
                grid_height // merge_size,
                merge_size,
                patch_size,
                grid_width // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(1, 4, 2, 5, 0, 3, 6).reshape(grid_height * grid_width, -1)

            pixel_values.append(patches)
            image_grids.append((1, grid_height, grid_width))

        return BatchFeature(
            data={
                "pixel_values": torch.cat(pixel_values, dim=0),
                "image_grid_thw": torch.tensor(image_grids, dtype=torch.long),
            },
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """Return the number of pre-projector vision patches for an image size."""
        images_kwargs = images_kwargs or {}
        size = images_kwargs.get("size", self.size)
        if isinstance(size, SizeDict):
            default_min_pixels = size.shortest_edge
            default_max_pixels = size.longest_edge
        else:
            default_min_pixels = size["shortest_edge"]
            default_max_pixels = size["longest_edge"]

        min_pixels = images_kwargs.get("min_pixels", default_min_pixels)
        max_pixels = images_kwargs.get("max_pixels", default_max_pixels)
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return (resized_height // patch_size) * (resized_width // patch_size)


__all__ = ["Cosmos3EdgeImageProcessor"]
