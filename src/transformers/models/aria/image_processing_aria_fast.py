# coding=utf-8
# Copyright 2025 The Rhymes-AI Teams Authors and The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union

from ...image_processing_utils import (
    BatchFeature,
    get_patch_output_size,
    select_best_resolution,
)
from ...image_processing_utils_fast import BaseImageProcessorFast, divide_to_patches
from ...image_utils import ChannelDimension, PILImageResampling
from ...utils import auto_docstring, is_torch_available, is_torchvision_available


if is_torch_available():
    import torch
    import torch.nn.functional as F

if is_torchvision_available():
    import torchvision.transforms.functional as F_tv


@auto_docstring
class AriaImageProcessorFast(BaseImageProcessorFast):
    model_input_names = ["pixel_values", "pixel_mask", "num_crops"]

    def __init__(
        self,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
        max_image_size: int = 980,
        min_image_size: int = 336,
        split_resolutions: Optional[list[tuple[int, int]]] = None,
        split_image: Optional[bool] = False,
        do_convert_rgb: Optional[bool] = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: Optional[bool] = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.split_image = split_image
        if split_resolutions is None:
            split_resolutions = [
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (1, 6),
                (1, 7),
                (1, 8),
                (2, 4),
                (2, 3),
                (2, 2),
                (2, 1),
                (3, 1),
                (3, 2),
                (4, 1),
                (4, 2),
                (5, 1),
                (6, 1),
                (7, 1),
                (8, 1),
            ]
            split_resolutions = [(el[0] * 490, el[1] * 490) for el in split_resolutions]
        self.split_resolutions = split_resolutions
        self.do_convert_rgb = do_convert_rgb
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.resample = resample

    def _preprocess(
        self,
        images: Union[torch.Tensor, list[torch.Tensor]],
        max_image_size: Optional[int] = None,
        min_image_size: Optional[int] = None,
        split_image: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        **kwargs,
    ) -> BatchFeature:
        max_image_size = max_image_size or self.max_image_size
        min_image_size = min_image_size or self.min_image_size
        split_image = split_image if split_image is not None else self.split_image
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor or self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize

        if max_image_size not in [490, 980]:
            raise ValueError("max_image_size must be either 490 or 980")

        if not isinstance(images, list):
            images = [images]

        pixel_values = []
        pixel_masks = []
        num_crops = None

        for image in images:
            if split_image:
                crop_images = self._get_image_patches_fast(
                    image, self.split_resolutions, max_image_size
                )
            else:
                crop_images = [image]

            if num_crops is None or len(crop_images) > num_crops:
                num_crops = len(crop_images)

            for crop_image in crop_images:
                h, w = crop_image.shape[-2:]
                scale = max_image_size / max(h, w)

                if w >= h:
                    new_size = (max(int(h * scale), min_image_size), max_image_size)
                else:
                    new_size = (max_image_size, max(int(w * scale), min_image_size))

                crop_image_resized = F_tv.resize(
                    crop_image, new_size, interpolation=F_tv.InterpolationMode.BICUBIC
                )

                padding_bottom = max_image_size - new_size[0]
                padding_right = max_image_size - new_size[1]
                crop_image_padded = F.pad(
                    crop_image_resized, (0, padding_right, 0, padding_bottom)
                )

                pixel_mask = torch.zeros(
                    (max_image_size, max_image_size), dtype=torch.bool
                )
                pixel_mask[: new_size[0], : new_size[1]] = True
                pixel_masks.append(pixel_mask)

                if do_rescale:
                    crop_image_padded = crop_image_padded * rescale_factor

                if do_normalize:
                    mean = torch.tensor(
                        self.image_mean,
                        device=crop_image_padded.device,
                        dtype=crop_image_padded.dtype,
                    ).view(-1, 1, 1)
                    std = torch.tensor(
                        self.image_std,
                        device=crop_image_padded.device,
                        dtype=crop_image_padded.dtype,
                    ).view(-1, 1, 1)
                    crop_image_padded = (crop_image_padded - mean) / std

                pixel_values.append(crop_image_padded)

        return BatchFeature(
            data={
                "pixel_values": torch.stack(pixel_values, dim=0),
                "pixel_mask": torch.stack(pixel_masks, dim=0),
                "num_crops": num_crops,
            }
        )

    def _get_image_patches_fast(
        self,
        image: torch.Tensor,
        grid_pinpoints: list[tuple[int, int]],
        patch_size: int,
    ) -> list[torch.Tensor]:
        h, w = image.shape[-2:]
        best_resolution = select_best_resolution((h, w), grid_pinpoints)

        new_height, new_width = get_patch_output_size(
            image, best_resolution, ChannelDimension.FIRST
        )
        resized_image = F_tv.resize(
            image, (new_height, new_width), interpolation=F_tv.InterpolationMode.BICUBIC
        )

        target_height, target_width = best_resolution
        paste_x, r_x = divmod(target_width - new_width, 2)
        paste_y, r_y = divmod(target_height - new_height, 2)

        padded_image = F.pad(
            resized_image, (paste_x, paste_x + r_x, paste_y, paste_y + r_y)
        )

        return divide_to_patches(padded_image, patch_size)


__all__ = ["AriaImageProcessorFast"]
