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
"""Fast Image processor class for DeepSeek-OCR-2."""

from typing import Optional

import torch
import torchvision.transforms.v2.functional as tvF

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import ChannelDimension, PILImageResampling, SizeDict, pil_torch_interpolation_mapping
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring, logging
from .image_processing_deepseek_ocr2 import DeepseekOcr2ImageProcessorKwargs, get_optimal_tiled_canvas


logger = logging.get_logger(__name__)


@auto_docstring
class DeepseekOcr2ImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.LANCZOS
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.5, 0.5, 0.5)
    size = {"height": 1024, "width": 1024}
    tile_size = 768
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    crop_to_patches = True
    min_patches = 2
    max_patches = 6
    model_input_names = ["pixel_values", "num_local_patches"]
    valid_kwargs = DeepseekOcr2ImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[DeepseekOcr2ImageProcessorKwargs]):
        super().__init__(**kwargs)
        self.background_color = kwargs.get("background_color", [127, 127, 127])

    @auto_docstring
    def preprocess(self, images, **kwargs: Unpack[DeepseekOcr2ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def pad_to_square(
        self,
        images: "torch.Tensor",
        background_color: int | tuple[int, int, int] = 0,
    ) -> "torch.Tensor":
        """
        Pads images to a square based on the longest edge.

        Args:
            images (`torch.Tensor`):
                The images to pad, shape `(batch, channels, height, width)`.
            background_color (`int` or `tuple[int, int, int]`, *optional*, defaults to 0):
                The color to use for the padding.

        Returns:
            `torch.Tensor`: The padded images.
        """
        height, width = images.shape[-2:]
        num_channels = images.shape[1]
        batch_size = images.shape[0]

        if height == width:
            return images

        max_dim = max(height, width)

        if isinstance(background_color, int):
            background_color = [background_color]
        elif len(background_color) != num_channels:
            raise ValueError(
                f"background_color must have no more than {num_channels} elements to match the number of channels"
            )

        padded_images = torch.zeros(
            (batch_size, num_channels, max_dim, max_dim), dtype=images.dtype, device=images.device
        )
        for i, color in enumerate(background_color):
            padded_images[:, i, :, :] = color
        if width > height:
            start = (max_dim - height) // 2
            padded_images[:, :, start : start + height, :] = images
        else:
            start = (max_dim - width) // 2
            padded_images[:, :, :, start : start + width] = images

        return padded_images

    def crop_image_to_patches(
        self,
        images: "torch.Tensor",
        min_patches: int,
        max_patches: int,
        tile_size: int,
        interpolation: Optional["tvF.InterpolationMode"] = None,
    ) -> tuple["torch.Tensor", int]:
        """
        Crop batched images to patches based on optimal tiling.

        Same-shape images share the same optimal grid, so the entire batch is processed together.

        Args:
            images (`torch.Tensor`):
                The images to crop, shape `(batch, channels, height, width)`.
            min_patches (`int`):
                Minimum number of patches.
            max_patches (`int`):
                Maximum number of patches.
            tile_size (`int`):
                The size of each tile.
            interpolation (`tvF.InterpolationMode`, *optional*):
                Interpolation mode for resizing.

        Returns:
            `tuple[torch.Tensor, int]`: Stacked patches `(batch, num_patches, channels, tile_size, tile_size)`
            and number of patches per image.
        """
        original_height, original_width = images.shape[-2:]

        num_columns, num_rows = get_optimal_tiled_canvas(
            (original_height, original_width), (tile_size, tile_size), min_patches, max_patches
        )

        target_width = tile_size * num_columns
        target_height = tile_size * num_rows
        num_blocks = num_columns * num_rows

        resized = self.resize(
            images, SizeDict(height=target_height, width=target_width), interpolation=interpolation
        )

        # Extract patches: (batch, C, grid_H, grid_W) → (batch, num_patches, C, tile, tile)
        patches = []
        for i in range(num_blocks):
            col = i % num_columns
            row = i // num_columns
            patch = resized[
                ...,
                row * tile_size : (row + 1) * tile_size,
                col * tile_size : (col + 1) * tile_size,
            ]
            patches.append(patch)

        # Stack: list of (batch, C, tile, tile) → (batch, num_patches, C, tile, tile)
        stacked_patches = torch.stack(patches, dim=1)

        return stacked_patches, num_blocks

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        crop_to_patches: bool,
        min_patches: int,
        max_patches: int,
        tile_size: int,
        interpolation: Optional["tvF.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        if interpolation == tvF.InterpolationMode.LANCZOS:
            logger.warning_once(
                "You have used fast image processor with LANCZOS resample which is not yet supported for torch.Tensor. "
                "BICUBIC resample will be used as an alternative. Please fall back to slow image processor if you "
                "want full consistency with the original model."
            )
            interpolation = tvF.InterpolationMode.BICUBIC

        # --- Local patches (batched by shape group) ---
        # Same shape = same aspect ratio = same grid, so batch crop is possible.
        num_local_patches = {}
        local_patches_grouped = {}

        if crop_to_patches:
            grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)

            for shape, stacked_images in grouped_images.items():
                h, w = shape[-2:]
                if max(h, w) > tile_size:
                    # (batch, C, H, W) → (batch, n_patches, C, tile, tile)
                    stacked_patches, n_patches = self.crop_image_to_patches(
                        stacked_images,
                        min_patches=min_patches,
                        max_patches=max_patches,
                        tile_size=tile_size,
                        interpolation=interpolation,
                    )
                    # Rescale + normalize as (batch*n_patches, C, tile, tile)
                    flat_patches = stacked_patches.reshape(-1, *stacked_patches.shape[2:])
                    flat_patches = self.rescale_and_normalize(
                        flat_patches, do_rescale, rescale_factor, do_normalize, image_mean, image_std
                    )
                    # Split back to per-image list of (n_patches, C, tile, tile)
                    local_patches_grouped[shape] = flat_patches.reshape(stacked_patches.shape)
                    num_local_patches[shape] = [n_patches] * stacked_images.shape[0]
                else:
                    local_patches_grouped[shape] = [None] * stacked_images.shape[0]
                    num_local_patches[shape] = [0] * stacked_images.shape[0]

            num_local_patches = reorder_images(num_local_patches, grouped_images_index)
            ordered_local = reorder_images(local_patches_grouped, grouped_images_index)
        else:
            num_local_patches = [0] * len(images)
            ordered_local = []

        # Flatten to list of (C, tile, tile) in original image order
        flat_local_list = [patch for item in ordered_local if item is not None for patch in item]

        # --- Global view (batched by shape group) ---
        # Same shape = same aspect ratio = same (new_h, new_w), so batch resize is possible.
        global_target_size = size.height if crop_to_patches else tile_size

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_global_grouped = {}
        for shape, stacked in grouped_images.items():
            h, w = shape[-2:]
            scale = global_target_size / max(h, w)
            new_h = round(h * scale)
            new_w = round(w * scale)
            stacked = self.resize(stacked, SizeDict(height=new_h, width=new_w), interpolation=interpolation)
            stacked = self.pad_to_square(stacked, background_color=self.background_color)
            stacked = self.rescale_and_normalize(
                stacked, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_global_grouped[shape] = stacked
        all_pixel_values_global = reorder_images(processed_global_grouped, grouped_images_index)

        data = {
            "pixel_values": all_pixel_values_global,
            "num_local_patches": num_local_patches,
        }
        if flat_local_list:
            data["pixel_values_local"] = flat_local_list

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _further_process_kwargs(
        self,
        size: SizeDict | None = None,
        default_to_square: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        data_format=None,
        **kwargs,
    ) -> dict:
        if kwargs is None:
            kwargs = {}
        if size is not None:
            size = SizeDict(**{"height": size["height"], "width": size["width"]} if isinstance(size, dict) else size)
        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)
        if data_format is None:
            data_format = ChannelDimension.FIRST

        resample = kwargs.pop("resample", None)
        if resample is not None:
            kwargs["interpolation"] = (
                pil_torch_interpolation_mapping[resample]
                if isinstance(resample, (int, PILImageResampling))
                else resample
            )

        kwargs["size"] = size
        kwargs["image_mean"] = image_mean
        kwargs["image_std"] = image_std
        kwargs["data_format"] = data_format

        return kwargs

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None) -> int:
        """
        Returns the number of local patches for a given image size.

        Args:
            height (`int`): Height of the input image.
            width (`int`): Width of the input image.
            images_kwargs (`dict`, *optional*): Kwargs to override processor defaults.

        Returns:
            `int`: Number of local patches.
        """
        if images_kwargs is None:
            images_kwargs = {}
        min_patches = images_kwargs.get("min_patches", self.min_patches)
        max_patches = images_kwargs.get("max_patches", self.max_patches)
        tile_size = images_kwargs.get("tile_size", self.tile_size)
        crop_to_patches = images_kwargs.get("crop_to_patches", self.crop_to_patches)

        if not crop_to_patches or max(height, width) <= tile_size:
            return 0

        num_columns, num_rows = get_optimal_tiled_canvas(
            (height, width), (tile_size, tile_size), min_patches, max_patches
        )
        return num_columns * num_rows


__all__ = ["DeepseekOcr2ImageProcessorFast"]
