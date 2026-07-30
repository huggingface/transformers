# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""PIL image processor class for Ovis2.5."""

import math
from collections.abc import Iterable

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


class Ovis2_5ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    min_pixels (`int`, *optional*, defaults to `448 * 448`):
        The minimum number of pixels in a resized image.
    max_pixels (`int`, *optional*, defaults to `1344 * 1792`):
        The maximum number of pixels in a resized image.
    patch_size (`int`, *optional*, defaults to 16):
        The spatial patch size used by the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 1):
        The temporal patch size used by the vision encoder. The released Ovis2.5 checkpoints require a value of 1.
    merge_size (`int`, *optional*, defaults to 2):
        The spatial merge size between the vision encoder and language model.
    """

    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int


# Adapted from transformers.models.ovis2_5.image_processing_ovis2_5.smart_resize
def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 448 * 448,
    max_pixels: int = 1344 * 1792,
) -> tuple[int, int]:
    """Resize while preserving Ovis2.5's native-resolution constraints."""
    if height <= 0 or width <= 0:
        raise ValueError(f"`height` and `width` must be positive, got height={height} and width={width}.")
    if factor <= 0:
        raise ValueError(f"`factor` must be positive, got {factor}.")
    if min_pixels <= 0 or max_pixels <= 0:
        raise ValueError(
            f"`min_pixels` and `max_pixels` must be positive, got min_pixels={min_pixels} and max_pixels={max_pixels}."
        )
    if min_pixels > max_pixels:
        raise ValueError(
            f"`min_pixels` must be less than or equal to `max_pixels`, got {min_pixels} and {max_pixels}."
        )

    if height < factor or width < factor:
        if height < width:
            width = round(factor / height * width)
            height = factor
        else:
            height = round(factor / width * height)
            width = factor
    elif max(height, width) / min(height, width) > 200:
        if height > width:
            height = 200 * width
        else:
            width = 200 * height

    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = math.floor(height / beta / factor) * factor
        resized_width = math.floor(width / beta / factor) * factor
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor

    if resized_height <= 0 or resized_width <= 0:
        raise ValueError(
            "Ovis2.5 smart resizing produced a non-positive output size. "
            f"Got ({resized_height}, {resized_width}) from ({height}, {width}); use a less extreme aspect ratio."
        )
    return resized_height, resized_width


# Adapted from transformers.models.ovis2_5.image_processing_ovis2_5._resolve_size
def _resolve_size(
    size: int | Iterable[int] | dict[str, int] | SizeDict | None,
    default_size: dict[str, int] | SizeDict,
    min_pixels: int | None,
    max_pixels: int | None,
) -> SizeDict:
    if size is None:
        size = default_size
    if isinstance(size, SizeDict):
        shortest_edge = size.shortest_edge
        longest_edge = size.longest_edge
    elif isinstance(size, dict):
        shortest_edge = size.get("shortest_edge")
        longest_edge = size.get("longest_edge")
    else:
        raise ValueError("`size` must be a dictionary or `SizeDict` with `shortest_edge` and `longest_edge`.")

    shortest_edge = min_pixels if min_pixels is not None else shortest_edge
    longest_edge = max_pixels if max_pixels is not None else longest_edge
    if shortest_edge is None or longest_edge is None:
        raise ValueError("`size` must contain `shortest_edge` and `longest_edge`.")
    return SizeDict(shortest_edge=shortest_edge, longest_edge=longest_edge)


# Adapted from transformers.models.ovis2_5.image_processing_ovis2_5._validate_patch_grid
def _validate_patch_grid(height: int, width: int, patch_size: int, merge_size: int) -> None:
    factor = patch_size * merge_size
    if height % factor != 0 or width % factor != 0:
        raise ValueError(
            "Ovis2.5 images must have height and width divisible by "
            f"`patch_size * merge_size` ({factor}), got ({height}, {width})."
        )


# Adapted from transformers.models.ovis2_5.image_processing_ovis2_5._validate_temporal_patch_size
def _validate_temporal_patch_size(temporal_patch_size: int) -> None:
    if temporal_patch_size != 1:
        raise ValueError(
            f"The released Ovis2.5 checkpoints require `temporal_patch_size=1`, got {temporal_patch_size}."
        )


@auto_docstring
class Ovis2_5ImageProcessorPil(PilBackend):
    do_resize = True
    resample = PILImageResampling.BILINEAR
    size = {"shortest_edge": 448 * 448, "longest_edge": 1344 * 1792}
    default_to_square = False
    do_rescale = True
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    valid_kwargs = Ovis2_5ImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[Ovis2_5ImageProcessorKwargs]):
        size = _resolve_size(
            kwargs.pop("size", None),
            self.size,
            kwargs.pop("min_pixels", None),
            kwargs.pop("max_pixels", None),
        )
        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(
        self,
        size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ) -> dict:
        size = _resolve_size(size, self.size, min_pixels, max_pixels)
        kwargs = super()._standardize_kwargs(size=size, **kwargs)
        return kwargs

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Ovis2_5ImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        if not images:
            raise ValueError("Ovis2.5 requires at least one image.")
        _validate_temporal_patch_size(temporal_patch_size)

        all_patches = []
        all_grids = []
        for image in images:
            height, width = image.shape[-2:]
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                image = self.resize(
                    image,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            else:
                resized_height, resized_width = height, width
            _validate_patch_grid(resized_height, resized_width, patch_size, merge_size)

            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            # The first two dimensions are image batch and time. An image has
            # one temporal position, which is repeated when a custom temporal
            # patch size greater than one is requested.
            patches = np.expand_dims(np.expand_dims(image, axis=0), axis=1)
            if pad := -patches.shape[1] % temporal_patch_size:
                repeats = np.repeat(patches[:, -1:], pad, axis=1)
                patches = np.concatenate([patches, repeats], axis=1)

            batch_size = patches.shape[0]
            grid_t = patches.shape[1] // temporal_patch_size
            channel = patches.shape[2]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
            patches = patches.reshape(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size * grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )
            all_patches.append(flatten_patches)
            all_grids.append([grid_t, grid_h, grid_w])

        pixel_values = np.concatenate(all_patches, axis=0)
        image_grid_thw = np.asarray(all_grids, dtype=np.int64)
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        """Return the number of pre-merge vision patches for one image."""
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        temporal_patch_size = images_kwargs.get("temporal_patch_size", self.temporal_patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        _validate_temporal_patch_size(temporal_patch_size)
        do_resize = images_kwargs.get("do_resize", self.do_resize)
        if do_resize:
            min_pixels = images_kwargs.get("min_pixels", self.size["shortest_edge"])
            max_pixels = images_kwargs.get("max_pixels", self.size["longest_edge"])
            height, width = smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        _validate_patch_grid(height, width, patch_size, merge_size)
        return (height // patch_size) * (width // patch_size)


__all__ = ["Ovis2_5ImageProcessorPil"]
