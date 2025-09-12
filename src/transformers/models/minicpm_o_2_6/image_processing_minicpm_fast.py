# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
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

import math
from typing import Optional, Union

import numpy as np
from numpy.lib.stride_tricks import as_strided

from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_transforms import to_pil_image
from ...image_utils import valid_images
from ...utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, TensorType
from ...utils.import_utils import (
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
)
from .processing_minicpm_o_2_6 import MiniCPMOBatchFeature


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)


def to_tensor(x):
    if is_torchvision_v2_available():
        img = F.to_image(x)
        return F.to_dtype(img, dtype=torch.float32, scale=True)
    return F.to_tensor(x)


class MiniCPMVImageProcessorFast(BaseImageProcessorFast):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        max_slice_nums=9,
        scale_resolution=448,
        patch_size=14,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
        self.patch_size = patch_size

        self.use_image_id = kwargs.pop("use_image_id", False)
        self.image_feature_size = kwargs.pop("image_feature_size", 64)

        self.slice_mode = kwargs.pop("slice_mode", True)

        self.image_mean = np.array(image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN)
        self.image_std = np.array(image_std if image_std is not None else IMAGENET_STANDARD_STD)

    def ensure_divide(self, length, patch_size):
        return max(round(length / patch_size) * patch_size, patch_size)

    def find_best_resize(self, original_size, scale_resolution, patch_size, allow_upscale=False):
        width, height = original_size
        if (width * height > scale_resolution * scale_resolution) or allow_upscale:
            r = width / height
            height = int(scale_resolution / math.sqrt(r))
            width = int(height * r)
        best_width = self.ensure_divide(width, patch_size)
        best_height = self.ensure_divide(height, patch_size)
        return (best_width, best_height)

    def get_refine_size(self, original_size, grid, scale_resolution, patch_size, allow_upscale=False):
        width, height = original_size
        grid_x, grid_y = grid

        refine_width = self.ensure_divide(width, grid_x)
        refine_height = self.ensure_divide(height, grid_y)

        grid_width = refine_width / grid_x
        grid_height = refine_height / grid_y

        best_grid_size = self.find_best_resize(
            (grid_width, grid_height), scale_resolution, patch_size, allow_upscale=allow_upscale
        )
        refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)
        return refine_size

    def split_to_patches(self, image, grid):
        patches = []
        width, height = image.size
        grid_x = int(width / grid[0])
        grid_y = int(height / grid[1])
        for i in range(0, height, grid_y):
            images = []
            for j in range(0, width, grid_x):
                box = (j, i, j + grid_x, i + grid_y)
                patch = image.crop(box)
                images.append(patch)
            patches.append(images)
        return patches

    def slice_image(self, image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False):
        original_size = image.size
        source_image = None
        best_grid = self.get_sliced_grid(original_size, max_slice_nums, never_split)
        patches = []

        if best_grid is None:
            # dont need to slice, upsample
            best_size = self.find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=True)
            source_image = image.resize(best_size, resample=Image.Resampling.BICUBIC)
        else:
            # source image, down-sampling and ensure divided by patch_size
            best_resize = self.find_best_resize(original_size, scale_resolution, patch_size)
            source_image = image.copy().resize(best_resize, resample=Image.Resampling.BICUBIC)
            refine_size = self.get_refine_size(
                original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
            )
            refine_image = image.resize(refine_size, resample=Image.Resampling.BICUBIC)
            patches = self.split_to_patches(refine_image, best_grid)

        return source_image, patches, best_grid

    def get_sliced_images(self, image, max_slice_nums=None):
        slice_images = []

        if not self.slice_mode:
            return [image]

        max_slice_nums = self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        if max_slice_nums <= 0:
            raise ValueError(f"max_slice_nums must be greater than 0, got {max_slice_nums}")
        source_image, patches, sliced_grid = self.slice_image(
            # default: 9  # default: 448  # default: 14
            image,
            max_slice_nums,
            self.scale_resolution,
            self.patch_size,
        )

        slice_images.append(source_image)
        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    slice_images.append(patches[i][j])
        return slice_images

    def get_sliced_grid(self, image_size, max_slice_nums, nerver_split=False):
        original_width, original_height = image_size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (self.scale_resolution * self.scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1 or nerver_split:
            return None
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        candidate_grids = []
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        return best_grid

    def reshape_by_patch(self, image):
        """
        :param image: shape [3, H, W]
        :param patch_size:
        :return: [3, patch_size, HW/patch_size]
        """
        C, H, W = image.shape
        patch_size = self.patch_size
        out_h, out_w = H // patch_size, W // patch_size
        s_c, s_h, s_w = image.strides
        patches = as_strided(
            image,
            shape=(C, out_h, out_w, patch_size, patch_size),
            strides=(s_c, s_h * patch_size, s_w * patch_size, s_h, s_w),
            writeable=False,
        )
        patches_t = patches.transpose(0, 3, 1, 2, 4)
        return patches_t.reshape(C, patch_size, -1)

    def preprocess(
        self,
        images: Union[Image.Image, list[Image.Image], list[list[Image.Image]]],
        max_slice_nums: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_normalize: bool = True,
        **kwargs,
    ) -> MiniCPMOBatchFeature:
        # in batch inference, it may be [[]], so we can't use `make_nested_list_of_images`
        if isinstance(images, Image.Image):
            images_list = [[images]]
        elif isinstance(images[0], Image.Image):
            images_list = [images]
        else:
            images_list = images

        new_images_list = []
        image_sizes_list = []
        tgt_sizes_list = []

        for _images in images_list:
            if not valid_images(_images):
                raise ValueError(
                    "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                    "torch.Tensor, tf.Tensor or jax.ndarray."
                )

            # Convert to PIL and RGB using torchvision
            _images = [to_pil_image(image).convert("RGB") for image in _images]

            new_images = []
            image_sizes = [image.size for image in _images]
            tgt_sizes = []
            for image in _images:
                image_patches = self.get_sliced_images(image, max_slice_nums)

                # Convert PIL images to torch tensors and normalize using torchvision
                image_patches_tensors = []
                for patch in image_patches:
                    # Convert PIL to tensor (0-1 range) and normalize
                    # Shape: [C, H, W], range [0, 1]
                    tensor_patch = to_tensor(patch)
                    if do_normalize:
                        normalized_patch = F.normalize(
                            tensor_patch, mean=self.image_mean.tolist(), std=self.image_std.tolist()
                        )  # Apply normalization
                    image_patches_tensors.append(normalized_patch)

                # Convert back to numpy for compatibility with existing code
                image_patches = [patch.numpy() for patch in image_patches_tensors]

                for slice_image in image_patches:
                    new_images.append(self.reshape_by_patch(slice_image))
                    tgt_sizes.append(
                        np.array((slice_image.shape[1] // self.patch_size, slice_image.shape[2] // self.patch_size))
                    )

            # in batch inference, it may be []
            if tgt_sizes:
                tgt_sizes = np.vstack(tgt_sizes)

            new_images_list.append(new_images)
            image_sizes_list.append(image_sizes)
            tgt_sizes_list.append(tgt_sizes)

        return MiniCPMOBatchFeature(
            data={"pixel_values": new_images_list, "image_sizes": image_sizes_list, "tgt_sizes": tgt_sizes_list},
            tensor_type=return_tensors,
        )


__all__ = ["MiniCPMVImageProcessorFast"]
