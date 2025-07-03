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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import PIL
import PIL.Image
import PIL.ImageSequence
import torch
from PIL import Image
from transformers import AutoImageProcessor
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_utils import ChannelDimension
from transformers.image_utils import infer_channel_dimension_format
from transformers.image_utils import is_torch_tensor
from transformers.image_utils import to_numpy_array
from transformers.image_utils import valid_images
from transformers.utils import is_torch_device
from transformers.utils import is_torch_dtype
from transformers.utils import requires_backends
from transformers.utils import TensorType
from .processing_minicpm_o_2_6 import MiniCPMOBatchFeature


def recursive_converter(converter, value):
    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value += [recursive_converter(converter, v)]
        return new_value
    else:
        return converter(value)


class MiniCPMVImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(self, max_slice_nums=9, scale_resolution=448, patch_size=14, **kwargs):
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

    def get_grid_placeholder(self, grid):
        if grid is None:
            return ""
        slice_image_placeholder = (
            self.tokenizer.slice_start + self.tokenizer.unk_token * self.image_feature_size + self.tokenizer.slice_end
        )

        cols = grid[0]
        rows = grid[1]
        slices = []
        for i in range(rows):
            lines = []
            for j in range(cols):
                lines.append(slice_image_placeholder)
            slices.append("".join(lines))

        slice_placeholder = "\n".join(slices)
        return slice_placeholder

    def get_image_id_placeholder(self, idx=0):
        return f"{self.tokenizer.im_id_start}{idx}{self.tokenizer.im_id_end}"

    def get_sliced_images(self, image, max_slice_nums=None):
        slice_images = []

        if not self.slice_mode:
            return [image]

        max_slice_nums = self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        assert max_slice_nums > 0
        source_image, patches, sliced_grid = self.slice_image(
            image, max_slice_nums, self.scale_resolution, self.patch_size  # default: 9  # default: 448  # default: 14
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

    def get_slice_image_placeholder(self, image_size, image_idx=0, max_slice_nums=None, use_image_id=None):
        max_slice_nums = self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        assert max_slice_nums > 0
        grid = self.get_sliced_grid(image_size=image_size, max_slice_nums=max_slice_nums)

        image_placeholder = self.tokenizer.im_start + self.tokenizer.unk_token * self.image_feature_size + self.tokenizer.im_end
        use_image_id = self.use_image_id if use_image_id is None else bool(use_image_id)
        if use_image_id:
            final_placeholder = self.get_image_id_placeholder(image_idx) + image_placeholder
        else:
            final_placeholder = image_placeholder

        if self.slice_mode:
            final_placeholder = final_placeholder + self.get_grid_placeholder(grid=grid)
        return final_placeholder

    def to_pil_image(self, image, rescale=None) -> PIL.Image.Image:
        """
        Converts `image` to a PIL Image. Optionally rescales it and puts the channel dimension back as the last axis if
        needed.

        Args:
            image (`PIL.Image.Image` or `numpy.ndarray` or `torch.Tensor`):
                The image to convert to the PIL Image format.
            rescale (`bool`, *optional*):
                Whether or not to apply the scaling factor (to make pixel values integers between 0 and 255). Will
                default to `True` if the image type is a floating type, `False` otherwise.
        """
        if isinstance(image, PIL.Image.Image):
            return image
        if is_torch_tensor(image):
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if rescale is None:
                # rescale default to the array being of floating type.
                rescale = isinstance(image.flat[0], np.floating)
            # If the channel as been moved to first dim, we put it back at the end.
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            if rescale:
                image = image * 255
            image = image.astype(np.uint8)
            return PIL.Image.fromarray(image)
        return image

    def reshape_by_patch(self, image):
        """
        :param image: shape [3, H, W]
        :param patch_size:
        :return: [3, patch_size, HW/patch_size]
        """
        image = torch.from_numpy(image)
        patch_size = self.patch_size
        patches = torch.nn.functional.unfold(image, (patch_size, patch_size), stride=(patch_size, patch_size))

        patches = patches.reshape(image.size(0), patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(image.size(0), patch_size, -1)
        return patches.numpy()

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]],
        do_pad: Optional[bool] = True,
        max_slice_nums: int = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> MiniCPMOBatchFeature:
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
            if _images is None or len(_images) == 0:
                new_images_list.append([])
                image_sizes_list.append([])
                tgt_sizes_list.append([])
                continue
            if not valid_images(_images):
                raise ValueError(
                    "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                    "torch.Tensor, tf.Tensor or jax.ndarray."
                )

            _images = [self.to_pil_image(image).convert("RGB") for image in _images]
            input_data_format = infer_channel_dimension_format(np.array(_images[0]))

            new_images = []
            image_sizes = [image.size for image in _images]
            tgt_sizes = []
            for image in _images:
                image_patches = self.get_sliced_images(image, max_slice_nums)
                image_patches = [to_numpy_array(image).astype(np.float32) / 255 for image in image_patches]
                image_patches = [
                    self.normalize(image=image, mean=self.mean, std=self.std, input_data_format=input_data_format)
                    for image in image_patches
                ]
                image_patches = [
                    to_channel_dimension_format(image, ChannelDimension.FIRST, input_channel_dim=input_data_format)
                    for image in image_patches
                ]
                for slice_image in image_patches:
                    new_images.append(self.reshape_by_patch(slice_image))
                    tgt_sizes.append(
                        np.array((slice_image.shape[1] // self.patch_size, slice_image.shape[2] // self.patch_size))
                    )

            if tgt_sizes:
                tgt_sizes = np.vstack(tgt_sizes)

            new_images_list.append(new_images)
            image_sizes_list.append(image_sizes)
            tgt_sizes_list.append(tgt_sizes)
        return MiniCPMOBatchFeature(
            data={"pixel_values": new_images_list, "image_sizes": image_sizes_list, "tgt_sizes": tgt_sizes_list},
            tensor_type=return_tensors,
        )


AutoImageProcessor.register("MiniCPMVImageProcessor", MiniCPMVImageProcessor)


__all__ = ["MiniCPMVImageProcessor"]
