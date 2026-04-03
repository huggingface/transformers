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
"""PIL-based Image processor class for DeepSeek-OCR-2."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
    infer_channel_dimension_format,
)
from ...processing_utils import Unpack
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires
from .image_processing_deepseek_ocr2 import DeepseekOcr2ImageProcessorKwargs, get_optimal_tiled_canvas


@requires(backends=("vision",))
@auto_docstring
class DeepseekOcr2ImageProcessorPil(PilBackend):
    valid_kwargs = DeepseekOcr2ImageProcessorKwargs
    resample = PILImageResampling.BICUBIC
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    size = {"height": 1024, "width": 1024}
    tile_size = 768
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    crop_to_patches = True
    min_patches = 2
    max_patches = 6
    background_color = [127, 127, 127]
    model_input_names = ["pixel_values", "num_local_patches"]

    def __init__(self, **kwargs: Unpack[DeepseekOcr2ImageProcessorKwargs]):
        super().__init__(**kwargs)

    def crop_image_to_patches(
        self,
        image: np.ndarray,
        min_patches: int,
        max_patches: int,
        tile_size: int,
        resample: "PILImageResampling | int | None" = None,
    ):
        """
        Crop the image to patches and return a list of cropped images.
        """
        input_data_format = infer_channel_dimension_format(image)
        image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

        original_height, original_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)

        num_columns, num_rows = get_optimal_tiled_canvas(
            (original_height, original_width), (tile_size, tile_size), min_patches, max_patches
        )

        target_width = tile_size * num_columns
        target_height = tile_size * num_rows
        num_blocks = num_columns * num_rows

        resized_image = self.resize(image, SizeDict(height=target_height, width=target_width), resample=resample)

        processed_images = []
        for i in range(num_blocks):
            column = i % num_columns
            row = i // num_columns
            box = (
                column * tile_size,
                row * tile_size,
                (column + 1) * tile_size,
                (row + 1) * tile_size,
            )
            patch_image = resized_image[..., box[1] : box[3], box[0] : box[2]]
            patch_image = to_channel_dimension_format(patch_image, input_data_format, ChannelDimension.FIRST)
            processed_images.append(patch_image)

        return processed_images

    def pad_to_square(
        self,
        image: np.ndarray,
        background_color: list[int] | int = 0,
    ) -> np.ndarray:
        """
        Pads an image to a square based on the longest edge.
        """
        input_data_format = infer_channel_dimension_format(image)
        height, width = get_image_size(image, input_data_format)
        num_channels = image.shape[0] if input_data_format == ChannelDimension.FIRST else image.shape[-1]

        if height == width:
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

    def _preprocess(
        self,
        images: list[np.ndarray],
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        crop_to_patches: bool = True,
        min_patches: int = 2,
        max_patches: int = 6,
        tile_size: int = 768,
        background_color: list[int] | None = None,
        **kwargs,
    ) -> BatchFeature:
        if background_color is None:
            background_color = self.background_color

        all_pixel_values_local = []
        all_pixel_values_global = []
        num_local_patches = []

        for image in images:
            original_height, original_width = get_image_size(image)

            # --- Local patches ---
            if crop_to_patches and max(original_width, original_height) > tile_size:
                local_patches = self.crop_image_to_patches(
                    image,
                    min_patches=min_patches,
                    max_patches=max_patches,
                    tile_size=tile_size,
                    resample=resample,
                )
                for patch in local_patches:
                    if do_rescale:
                        patch = self.rescale(patch, rescale_factor)
                    if do_normalize:
                        patch = self.normalize(patch, image_mean, image_std)
                    all_pixel_values_local.append(patch)
                num_local_patches.append(len(local_patches))
            else:
                num_local_patches.append(0)

            # --- Global view ---
            global_target_size = size.height if crop_to_patches else tile_size
            scale = global_target_size / max(original_width, original_height)
            new_width = round(original_width * scale)
            new_height = round(original_height * scale)

            global_img = self.resize(image, SizeDict(height=new_height, width=new_width), resample=resample)
            global_img = self.pad_to_square(global_img, background_color=background_color)
            if do_rescale:
                global_img = self.rescale(global_img, rescale_factor)
            if do_normalize:
                global_img = self.normalize(global_img, image_mean, image_std)
            all_pixel_values_global.append(global_img)

        data = {
            "pixel_values": all_pixel_values_global,
            "num_local_patches": num_local_patches,
        }
        if all_pixel_values_local:
            data["pixel_values_local"] = all_pixel_values_local

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["DeepseekOcr2ImageProcessorPil"]
