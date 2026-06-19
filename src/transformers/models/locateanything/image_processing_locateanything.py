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
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, make_list_of_images
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
    image_mean = (0.5, 0.5, 0.5)
    image_std = (0.5, 0.5, 0.5)
    patch_size = 14
    merge_kernel_size = (2, 2)
    in_token_limit = 25600
    do_rescale = True
    do_normalize = True

    valid_kwargs = LocateAnythingImageProcessorKwargs

    def _target_size(self, height: int, width: int) -> tuple[int, int]:
        patch = self.patch_size
        if (width // patch) * (height // patch) > self.in_token_limit:
            scale = math.sqrt(self.in_token_limit / ((width // patch) * (height // patch)))
            height, width = int(height * scale), int(width * scale)
        pad_h = self.merge_kernel_size[0] * patch
        pad_w = self.merge_kernel_size[1] * patch
        target_h = math.ceil(height / pad_h) * pad_h
        target_w = math.ceil(width / pad_w) * pad_w
        return target_h, target_w

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        return_tensors: str | TensorType | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        images = make_list_of_images(images)
        patch = self.patch_size
        all_patches, all_grids = [], []
        for image in images:
            image = tvF.pil_to_tensor(image.convert("RGB")) if not isinstance(image, torch.Tensor) else image
            _, height, width = image.shape
            target_h, target_w = self._target_size(height, width)
            if (target_h, target_w) != (height, width):
                image = tvF.resize(
                    image, [target_h, target_w], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True
                )
            image = tvF.to_dtype(image, torch.float32, scale=True)
            image = tvF.normalize(image, list(self.image_mean), list(self.image_std))

            channels, height, width = image.shape
            grid_h, grid_w = height // patch, width // patch
            patches = image.reshape(channels, grid_h, patch, grid_w, patch)
            patches = patches.permute(1, 3, 0, 2, 4).reshape(-1, channels, patch, patch)
            all_patches.append(patches)
            all_grids.append([grid_h, grid_w])

        data = {
            "pixel_values": torch.cat(all_patches, dim=0),
            "image_grid_hws": torch.tensor(all_grids, dtype=torch.long),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["LocateAnythingImageProcessor"]
