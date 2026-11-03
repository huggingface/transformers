# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Image processor for the NemotronH Omni model."""

import math

import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


class NemotronH_Omni_Reasoning_V3ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to 16):
        Side length, in pixels, of one vision-tower patch.
    downsample_ratio (`float`, *optional*, defaults to 0.5):
        Pixel-shuffle spatial downsample ratio applied by the model; each side of the patch grid is a multiple
        of its reciprocal.
    min_num_patches (`int`, *optional*, defaults to 1024):
        Minimum number of `patch_size` patches an image is resized to.
    max_num_patches (`int`, *optional*, defaults to 13312):
        Maximum number of `patch_size` patches an image is resized to; `0` disables the cap.
    max_model_len (`int`, *optional*, defaults to 16384):
        Context length of the language model, which caps the patch budget of the images in one call.
    """

    patch_size: int
    downsample_ratio: float
    min_num_patches: int
    max_num_patches: int
    max_model_len: int


def get_patch_grid_size(
    height: int, width: int, patch_size: int, merge_size: int, min_num_patches: int, max_num_patches: int
) -> tuple[int, int]:
    """
    Picks the `(grid_height, grid_width)` patch grid an image is resized to: close to the image's own grid,
    scaled into `[min_num_patches, max_num_patches]` and rounded so both sides are multiples of `merge_size`.
    """
    # `round(x + 0.5)` is `floor(x) + 1` for non-integer `x` and `x` otherwise
    grid_height = round(height / patch_size + 0.5)
    grid_width = round(width / patch_size + 0.5)

    factor = min(math.sqrt(max_num_patches / (grid_height * grid_width)), 1.0)
    grid_height = math.floor(factor * grid_height)
    grid_width = math.floor(factor * grid_width)

    if max_num_patches > min_num_patches and grid_height * grid_width < min_num_patches:
        factor = math.sqrt(min_num_patches / (grid_height * grid_width))
        grid_height = math.ceil(factor * grid_height)
        grid_width = math.ceil(factor * grid_width)

    if remainder := grid_height % merge_size:
        if (grid_height + merge_size - remainder) * grid_width <= max_num_patches:
            grid_height += merge_size - remainder
        else:
            grid_height = max(merge_size, grid_height - remainder)
    if remainder := grid_width % merge_size:
        if grid_height * (grid_width + merge_size - remainder) <= max_num_patches:
            grid_width += merge_size - remainder
        else:
            grid_width = max(merge_size, grid_width - remainder)

    return grid_height, grid_width


def convert_image_to_patches(images: "torch.Tensor", patch_size: int) -> "torch.Tensor":
    """
    Splits `(batch_size, num_channels, height, width)` images into `(batch_size, num_patches, patch_dim)` patches in
    row-major patch order, each patch laid out channel-major as `(num_channels, patch_size, patch_size)`.
    """
    batch_size, num_channels, height, width = images.shape
    grid_height, grid_width = height // patch_size, width // patch_size
    patches = images.reshape(batch_size, num_channels, grid_height, patch_size, grid_width, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(batch_size, grid_height * grid_width, num_channels * patch_size**2)


@auto_docstring
class NemotronH_Omni_Reasoning_V3ImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    do_rescale = True
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 16
    downsample_ratio = 0.5
    min_num_patches = 1024
    max_num_patches = 13312
    max_model_len = 16384
    valid_kwargs = NemotronH_Omni_Reasoning_V3ImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_hw"]

    def __init__(self, **kwargs: Unpack[NemotronH_Omni_Reasoning_V3ImageProcessorKwargs]):
        # backward compatibility: released checkpoints store the normalization statistics as `norm_mean` / `norm_std`
        if (norm_mean := kwargs.pop("norm_mean", None)) is not None:
            kwargs.setdefault("image_mean", norm_mean)
        if (norm_std := kwargs.pop("norm_std", None)) is not None:
            kwargs.setdefault("image_std", norm_std)
        kwargs.pop("_downsample_factor", None)
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images, **kwargs: Unpack[NemotronH_Omni_Reasoning_V3ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,
        downsample_ratio: float,
        min_num_patches: int,
        max_num_patches: int,
        max_model_len: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        merge_size = round(1 / downsample_ratio)
        # every image of the call gets the same budget: the context length (in pre-pixel-shuffle patches),
        # clamped to `[min_num_patches, max_num_patches]`
        num_patches = max((max_model_len - 4) * merge_size**2, min_num_patches * len(images))
        if max_num_patches:
            num_patches = min(num_patches, max_num_patches)
        num_patches = max(num_patches, min_num_patches)

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        image_grids_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            grid_height, grid_width = get_patch_grid_size(
                height, width, patch_size, merge_size, min_num_patches, num_patches
            )
            # resized in float32 so the bicubic overshoot is kept rather than clamped back to uint8
            stacked_images = stacked_images.to(torch.float32)
            if (height, width) != (grid_height * patch_size, grid_width * patch_size):
                stacked_images = self.resize(
                    stacked_images,
                    SizeDict(height=grid_height * patch_size, width=grid_width * patch_size),
                    resample=resample,
                )
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = convert_image_to_patches(stacked_images, patch_size)
            image_grids_grouped[shape] = torch.tensor([[grid_height, grid_width]] * len(stacked_images))

        pixel_values = reorder_images(processed_images_grouped, grouped_images_index)
        image_grid_hw = reorder_images(image_grids_grouped, grouped_images_index)
        return BatchFeature(
            data={
                "pixel_values": torch.cat(pixel_values),  # (total_patches, num_channels * patch_size**2)
                "image_grid_hw": torch.stack(image_grid_hw),  # (num_images, 2)
            },
            tensor_type=return_tensors,
        )


__all__ = ["NemotronH_Omni_Reasoning_V3ImageProcessor"]
