# Copyright 2026 the MiniMax AI Team and HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fast image processor for MiniMax M3 VL."""

import math

import torch
from torchvision.transforms import InterpolationMode

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType


MAX_RATIO = 200


def _round(n: int, f: int) -> int:
    return round(n / f) * f


def _ceil(n: int, f: int) -> int:
    return math.ceil(n / f) * f


def _floor(n: int, f: int) -> int:
    return math.floor(n / f) * f


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 451584,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, _round(height, factor))
    w_bar = max(factor, _round(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor(height / beta, factor)
        w_bar = _floor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil(height * beta, factor)
        w_bar = _ceil(width * beta, factor)
    return h_bar, w_bar


class MiniMaxM3VLImageProcessorKwargs(ImagesKwargs, total=False):
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    max_pixels: int


class MiniMaxM3VLImageProcessorFast(BaseImageProcessorFast):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"height": 672, "width": 672}
    default_to_square = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    max_pixels = 451584
    valid_kwargs = MiniMaxM3VLImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[MiniMaxM3VLImageProcessorKwargs]):
        super().__init__(**kwargs)

    def preprocess(self, images, **kwargs: Unpack[MiniMaxM3VLImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        max_pixels: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_grouped = {}
        factor = patch_size * merge_size
        for shape, stacked in grouped_images.items():
            height, width = stacked.shape[-2:]
            if do_resize:
                rh, rw = smart_resize(height, width, factor=factor, max_pixels=max_pixels)
                stacked = self.resize(stacked, size=SizeDict(height=rh, width=rw), resample=resample)
            resized_grouped[shape] = stacked
        resized = reorder_images(resized_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized, disable_grouping=disable_grouping)
        processed_grouped = {}
        grids = {}
        for shape, stacked in grouped_images.items():
            rh, rw = stacked.shape[-2:]
            patches = self.rescale_and_normalize(
                stacked, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:
                patches = patches.unsqueeze(1)
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(
                    1, temporal_patch_size - (patches.shape[1] % temporal_patch_size), 1, 1, 1
                )
                patches = torch.cat([patches, repeats], dim=1)

            bs, grid_t, c = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = rh // patch_size, rw // patch_size
            patches = patches.view(
                bs,
                grid_t,
                temporal_patch_size,
                c,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flat = patches.reshape(bs, grid_t * grid_h * grid_w, c * temporal_patch_size * patch_size * patch_size)
            processed_grouped[shape] = flat
            grids[shape] = [[grid_t, grid_h, grid_w]] * bs

        processed = reorder_images(processed_grouped, grouped_images_index)
        grids = reorder_images(grids, grouped_images_index)
        pixel_values = torch.cat(processed, dim=0)
        image_grid_thw = torch.tensor(grids, dtype=torch.long)
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        max_pixels = images_kwargs.get("max_pixels", self.max_pixels)
        rh, rw = smart_resize(height, width, factor=patch_size * merge_size, max_pixels=max_pixels)
        grid_h, grid_w = rh // patch_size, rw // patch_size
        return grid_h * grid_w


__all__ = ["MiniMaxM3VLImageProcessorFast"]
