# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Image processor for LocateAnything (native-resolution MoonViT patches)."""

import math

import torch

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature, SizeDict
from ...image_utils import PILImageResampling
from ...processing_utils import ImagesKwargs
from ...utils import TensorType, auto_docstring


class LocateAnythingImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to 14):
        Side length in pixels of a single vision patch.
    merge_kernel_size (`tuple[int, int]`, *optional*, defaults to `(2, 2)`):
        Patch-merge window applied in the projector; the placeholder token count is `h*w // prod(merge_kernel_size)`.
    in_token_limit (`int`, *optional*, defaults to 25600):
        Maximum number of patches per image; larger images are downscaled to fit.
    """

    patch_size: int
    merge_kernel_size: tuple[int, int]
    in_token_limit: int


@auto_docstring
class LocateAnythingImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    patch_size = 14
    merge_kernel_size = [2, 2]
    in_token_limit = 25600
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    model_input_names = ["pixel_values", "image_grid_thw"]

    valid_kwargs = LocateAnythingImageProcessorKwargs

    def _target_size(
        self, height: int, width: int, patch_size: int, merge_kernel_size: tuple[int, int], in_token_limit: int
    ) -> tuple[int, int]:
        if (width // patch_size) * (height // patch_size) > in_token_limit:
            scale = math.sqrt(in_token_limit / ((width // patch_size) * (height // patch_size)))
            height, width = int(height * scale), int(width * scale)
        pad_h = merge_kernel_size[0] * patch_size
        pad_w = merge_kernel_size[1] * patch_size
        target_h = math.ceil(height / pad_h) * pad_h
        target_w = math.ceil(width / pad_w) * pad_w
        return target_h, target_w

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        merge_kernel_size: tuple[int, int],
        in_token_limit: int,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        all_patches, all_grids = [], []
        for image in images:
            _, height, width = image.shape
            target_h, target_w = self._target_size(height, width, patch_size, merge_kernel_size, in_token_limit)
            if (target_h, target_w) != (height, width):
                image = self.resize(image, SizeDict(height=target_h, width=target_w), resample=resample)
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

            channels, height, width = image.shape
            grid_h, grid_w = height // patch_size, width // patch_size
            patches = image.reshape(channels, grid_h, patch_size, grid_w, patch_size)
            patches = patches.permute(1, 3, 0, 2, 4).reshape(-1, channels, patch_size, patch_size)
            all_patches.append(patches)
            all_grids.append([1, grid_h, grid_w])

        data = {
            "pixel_values": torch.cat(all_patches, dim=0),
            "image_grid_thw": torch.tensor(all_grids, dtype=torch.long),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["LocateAnythingImageProcessor"]
