# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
"""PIL Image processor class for Qwen2-VL."""

import math

import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


class Kimi2_6ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    max_patches (`int`, *optional*, defaults to `16384`):
        The max limit to resize resize the image.
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    merge_kernel_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    """

    max_patches: int
    patch_size: int
    merge_size: int


def navit_resize(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    max_patches: int,
    max_size_per_side: int,
):
    num_patches_w = max(1.0, width // patch_size)
    num_patches_h = max(1.0, height // patch_size)
    current_patch_count = num_patches_w * num_patches_h

    # Scale to satisfy total patch budget (affects both dims, hence sqrt)
    scale_for_total_patches = math.sqrt(max_patches / current_patch_count)

    # Scale to satisfy per-side patch budget
    scale_for_width_patches = (max_size_per_side * patch_size) / width
    scale_for_height_patches = (max_size_per_side * patch_size) / height

    # Use the most restrictive scale, never upscale
    scale = min(1.0, scale_for_total_patches, scale_for_width_patches, scale_for_height_patches)

    # Make sure the resized size doesn't go beyond predefined `max`
    new_width, new_height = max(1, int(width * scale)), max(1, int(height * scale))
    new_width = min(new_width, max_size_per_side * patch_size)
    new_height = min(new_height, max_size_per_side * patch_size)

    # Calculate the padding to make the height and width divisible by the merge kernel size and patch size.
    factor = merge_kernel_size * patch_size
    pad_height = (factor - new_height % factor) % factor + new_height
    pad_width = (factor - new_width % factor) % factor + new_width

    return (new_height, new_width), (pad_height, pad_width)


@auto_docstring
class Kimi2_6ImageProcessor(TorchvisionBackend):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"max_height": 512, "max_width": 512}
    max_patches = 16384
    default_to_square = False
    do_rescale = True
    do_normalize = True
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_convert_rgb = True
    patch_size = 14
    merge_size = 2
    valid_kwargs = Kimi2_6ImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[Kimi2_6ImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(
        self,
        size: SizeDict | None = None,
        **kwargs,
    ) -> dict:
        if size is not None:
            if size.max_height is None or size.max_width is None or (size.max_height != size.max_width):
                raise ValueError(f"size must contain 'max_height' and 'max_width' keys with identical values but got {size}.")
        super()._validate_preprocess_kwargs(size=size, **kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Kimi2_6ImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        merge_size: int,
        max_patches: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                (resized_height, resized_width), (pad_height, pad_width) = navit_resize(
                    height,
                    width,
                    patch_size=patch_size,
                    merge_kernel_size=merge_size,
                    max_patches=max_patches,
                    max_size_per_side=size.max_height,
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_images = self.pad(stacked_images, pad_size=SizeDict(height=pad_height, width=pad_width))
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            # Patchify in NaViT style, TODO maybe same as Siglip2 - needs to check with model
            batch_size, channels, height, width = stacked_images.shape
            grid_h, grid_w = height // patch_size, width // patch_size
            patches = stacked_images.reshape(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
            patches = patches.permute(0, 2, 4, 1, 3, 5)

            processed_images_grouped[shape] = patches.reshape(batch_size, -1, channels, patch_size, patch_size)
            processed_grids[shape] = [[1, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids_ordered = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.long)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )


__all__ = ["Kimi2_6ImageProcessor"]
