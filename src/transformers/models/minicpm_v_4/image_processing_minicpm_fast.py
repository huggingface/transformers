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
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode

from transformers.utils import TensorType

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import (
    is_torch_tensor,
    make_nested_list_of_images,
    valid_images,
)
from ...utils import auto_docstring


@auto_docstring
class MiniCPMVImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = None
    image_mean = None
    image_std = None
    size = None
    default_to_square = None
    crop_size = None
    do_resize = None
    do_center_crop = None
    do_rescale = None
    do_normalize = None
    do_convert_rgb = None
    model_input_names = ["pixel_values"]

    def __init__(
            self,
            max_slice_nums=9,
            scale_resolution=448,
            patch_size=14,
            **kwargs):
        super().__init__(**kwargs)
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
        self.patch_size = patch_size
        self.use_image_id = kwargs.pop("use_image_id", False)
        self.image_feature_size = kwargs.pop("image_feature_size", 64)
        self.slice_mode = kwargs.pop("slice_mode", True)
        self.mean = np.array(kwargs.pop("norm_mean", [0.5, 0.5, 0.5]))
        self.std = np.array(kwargs.pop("norm_std", [0.5, 0.5, 0.5]))
        self.version = kwargs.pop("version", 2.0)

    def ensure_divide(self, length, patch_size):
        return max(round(length / patch_size) * patch_size, patch_size)

    def find_best_resize(self,
                         original_size,
                         scale_resolution,
                         patch_size,
                         allow_upscale=False):
        width, height = original_size
        if (width * height >
                scale_resolution * scale_resolution) or allow_upscale:
            r = width / height
            height = int(scale_resolution / math.sqrt(r))
            width = int(height * r)
        best_width = self.ensure_divide(width, patch_size)
        best_height = self.ensure_divide(height, patch_size)
        return (best_width, best_height)

    def get_refine_size(self,
                        original_size,
                        grid,
                        scale_resolution,
                        patch_size,
                        allow_upscale=False):
        width, height = original_size
        grid_x, grid_y = grid

        refine_width = self.ensure_divide(width, grid_x)
        refine_height = self.ensure_divide(height, grid_y)

        grid_width = refine_width / grid_x
        grid_height = refine_height / grid_y

        best_grid_size = self.find_best_resize((grid_width, grid_height),
                                               scale_resolution,
                                               patch_size,
                                               allow_upscale=allow_upscale)
        refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)
        return refine_size

    def slice_image_tensor(
        self, image_tensor: torch.Tensor, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
    ):
        """
        Slices and resizes an image tensor based on the model's slicing logic.

        Note on numerical precision:
        This function utilizes `torchvision.transforms.v2.functional.resize` for image scaling, which has
        a different underlying implementation of the BICUBIC interpolation algorithm compared to the
        `Pillow` library's `resize` method used in the slow processor.

        This implementation difference leads to minor, unavoidable numerical discrepancies in the
        output pixel values when processing images. Consequently, while this fast processor
        offers performance gains, it is **not recommended for model evaluation or any task
        that requires bit-for-bit reproducibility** with the original slow processor. For exact
        numerical consistency, please use the corresponding slow image processor.
        """
        original_size = (image_tensor.shape[2], image_tensor.shape[1]) # (W, H)
        best_grid = self.get_sliced_grid(original_size, max_slice_nums, never_split)

        patches = []

        if best_grid is None:
            best_size = self.find_best_resize(
                original_size, scale_resolution, patch_size, allow_upscale=True
            )
            source_image = F.resize(image_tensor, (best_size[1], best_size[0]), interpolation=InterpolationMode.BICUBIC, antialias=True)
            return source_image, patches, best_grid
        else:
            best_resize = self.find_best_resize(original_size, scale_resolution, patch_size)
            source_image = F.resize(image_tensor, (best_resize[1], best_resize[0]), interpolation=InterpolationMode.BICUBIC, antialias=True)

            refine_size = self.get_refine_size(
                original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
            )
            refine_image = F.resize(image_tensor, (refine_size[1], refine_size[0]), interpolation=InterpolationMode.BICUBIC, antialias=True)

            grid_x, grid_y = best_grid
            patch_h = refine_image.shape[1] // grid_y
            patch_w = refine_image.shape[2] // grid_x

            for i in range(grid_y):
                images_row = []
                for j in range(grid_x):
                    patch = refine_image[:, i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
                    images_row.append(patch)
                patches.append(images_row)

        return source_image, patches, best_grid

    def get_sliced_grid(self, image_size, max_slice_nums, nerver_split=False):
        original_width, original_height = image_size
        log_ratio = math.log(original_width / original_height) if original_width > 0 and original_height > 0 else 0
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
            error = abs(log_ratio - math.log(grid[0] / grid[1])) if grid[0] > 0 and grid[1] > 0 else float("inf")
            if error < min_error:
                best_grid = grid
                min_error = error

        return best_grid

    def reshape_by_patch(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param image_tensor: shape [3, H, W]
        :return: [3, patch_size, HW/patch_size]
        """
        patch_size = self.patch_size
        patches = torch.nn.functional.unfold(
            image_tensor.unsqueeze(0),
            (patch_size, patch_size),
            stride=(patch_size, patch_size)
        ).squeeze(0)

        patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(image_tensor.size(0), patch_size, -1)
        return patches

    def _preprocess(
        self,
        images: Union[Image.Image, list[Image.Image], list[list[Image.Image]]],
        do_pad: Optional[bool] = True,  # Maintained for API consistency
        max_slice_nums: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> BatchFeature:
        images_list = make_nested_list_of_images(images)
        max_slice_nums = self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)

        global_max_h, global_max_w = 0, 0
        all_unpadded_patches_in_batch = []
        all_original_shapes_in_batch = []
        all_original_tgt_sizes_in_batch = []
        image_sizes_list = []
        num_patches_per_image = []

        device = "cpu"
        if images_list and images_list[0] and is_torch_tensor(images_list[0][0]):
            device = images_list[0][0].device

        for _images in images_list:
            if not valid_images(_images):
                raise ValueError(
                    "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                    "or torch.Tensor."
                )

            current_image_sizes = []
            for img in _images:
                if isinstance(img, Image.Image):
                    current_image_sizes.append(img.size)
                elif is_torch_tensor(img):
                    current_image_sizes.append((img.shape[-1], img.shape[-2]))
                else: # numpy array
                    current_image_sizes.append((img.shape[1], img.shape[0]))
            image_sizes_list.append(current_image_sizes)

            image_tensors = [F.to_image(img).to(device) for img in _images]

            patches_for_sample, shapes_for_sample, tgt_sizes_for_sample = [], [], []
            mean_tensor = torch.tensor(self.mean, device=device, dtype=torch.float32).view(3, 1, 1)
            std_tensor = torch.tensor(self.std, device=device, dtype=torch.float32).view(3, 1, 1)

            for image_tensor in image_tensors:
                source_image_u8, patches_u8, _ = self.slice_image_tensor(
                    image_tensor, max_slice_nums, self.scale_resolution, self.patch_size
                )

                slice_images_tensors_u8 = [source_image_u8]
                if patches_u8:
                    for row in patches_u8:
                        slice_images_tensors_u8.extend(row)

                num_patches_per_image.append(len(slice_images_tensors_u8))

                for slice_tensor_u8 in slice_images_tensors_u8:
                    slice_tensor_f32 = F.convert_image_dtype(slice_tensor_u8, dtype=torch.float32)
                    normalized_tensor = F.normalize(slice_tensor_f32, mean=mean_tensor, std=std_tensor)

                    _, h, w = normalized_tensor.shape
                    if h > global_max_h:
                        global_max_h = h
                    if w > global_max_w:
                        global_max_w = w

                    patches_for_sample.append(normalized_tensor)
                    shapes_for_sample.append(normalized_tensor.shape) # ori torch.Size

                    # ori tgt_size (torch)
                    tgt_size = torch.tensor(
                        (h // self.patch_size, w // self.patch_size), device=device, dtype=torch.long
                    )
                    tgt_sizes_for_sample.append(tgt_size)

            all_unpadded_patches_in_batch.append(patches_for_sample)
            all_original_shapes_in_batch.append(shapes_for_sample)
            if tgt_sizes_for_sample:
                all_original_tgt_sizes_in_batch.append(torch.stack(tgt_sizes_for_sample))
            else:
                all_original_tgt_sizes_in_batch.append(torch.empty((0, 2), dtype=torch.long, device=device))

        patch_size = self.patch_size

        padded_h = (global_max_h + patch_size - 1) // patch_size * patch_size
        padded_w = (global_max_w + patch_size - 1) // patch_size * patch_size

        padded_patches_list = []
        for patches_for_sample in all_unpadded_patches_in_batch:
            sample_padded_and_reshaped = []
            for unpadded_patch in patches_for_sample:
                _, h, w = unpadded_patch.shape
                pad_h = padded_h - h
                pad_w = padded_w - w

                padded_patch = F.pad(unpadded_patch, padding=(0, 0, pad_w, pad_h), padding_mode="constant", fill=0)

                reshaped_patch = self.reshape_by_patch(padded_patch)
                sample_padded_and_reshaped.append(reshaped_patch)

            padded_patches_list.append(sample_padded_and_reshaped)

        max_patches_in_batch = max(len(p) for p in padded_patches_list) if padded_patches_list else 0

        if max_patches_in_batch > 0:
            first_valid_patch = next((p[0] for p in padded_patches_list if p), None)
            if first_valid_patch is not None:
                patch_shape = first_valid_patch.shape
                patch_dtype = first_valid_patch.dtype
                dummy_patch = torch.zeros(patch_shape, dtype=patch_dtype, device=device)
                dummy_shape = (-1, -1, -1)
                dummy_tgt_size = torch.tensor([-1, -1], dtype=torch.long, device=device)

                for i in range(len(padded_patches_list)):
                    num_to_pad = max_patches_in_batch - len(padded_patches_list[i])
                    if num_to_pad > 0:
                        padded_patches_list[i].extend([dummy_patch] * num_to_pad)

                        all_original_shapes_in_batch[i].extend([dummy_shape] * num_to_pad)

                        dummy_tgt_sizes_to_add = dummy_tgt_size.unsqueeze(0).expand(num_to_pad, -1)
                        all_original_tgt_sizes_in_batch[i] = torch.cat(
                            [all_original_tgt_sizes_in_batch[i], dummy_tgt_sizes_to_add], dim=0
                        )

        data = {
            "pixel_values": padded_patches_list,
            "image_sizes": image_sizes_list,
            "num_patches_per_image": num_patches_per_image,
            "original_patch_shapes": all_original_shapes_in_batch,
            "original_tgt_sizes": all_original_tgt_sizes_in_batch,
            "padded_image_shape": (padded_h, padded_w),
        }

        return BatchFeature(data=data)


__all__ = ["MiniCPMVImageProcessorFast"]
