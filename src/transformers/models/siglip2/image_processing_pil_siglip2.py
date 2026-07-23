# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for SigLIP2."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)


def convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Convert 3D array image of shape (num_channels, image_height, image_width) into 2D array of patches of shape
    (num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
    """
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.transpose(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def pad_along_first_dim(array: np.ndarray, target_length: int, pad_value: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad the array along the first dimension.
    """
    current_length = array.shape[0]
    padding_length = target_length - current_length
    mask = np.ones((target_length,), dtype=np.int32)
    if padding_length > 0:
        paddings = [(0, padding_length)] + [(0, 0)] * (array.ndim - 1)
        array = np.pad(array, paddings, mode="constant", constant_values=pad_value)
        mask[-padding_length:] = 0
    return array, mask


# Adapted from transformers.models.siglip2.image_processing_siglip2.Siglip2ImageProcessorKwargs
class Siglip2ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to `self.patch_size`):
        The size (resolution) of each patch the image will be split to.
    max_num_patches (`int`, *optional*, defaults to `self.max_num_patches`):
        The image will be resized to have at most this number of patches,
        and then padded in "patch" dimension to match this number exactly.
    """

    patch_size: int
    max_num_patches: int


# Adapted from transformers.models.siglip2.image_processing_siglip2.get_image_size_for_max_num_patches
def get_image_size_for_max_num_patches(
    image_height: int, image_width: int, patch_size: int, max_num_patches: int, eps: float = 1e-5
) -> tuple[int, int]:
    """
    Determine image size based on max number of patches, ensure dimensions are divisible by patch size and image is at least 1 patch.

    Args:
        image_height (`int`):
            Original image height.
        image_width (`int`):
            Original image width.
        patch_size (`int`):
            Patch size for processing.
        max_num_patches (`int`):
            Maximum number of patches.
        eps (`float`):
            Small threshold for binary search.

    Returns:
        Tuple: (target_height, target_width)
    """
    import math

    def get_scaled_image_size(scale: float, size: int, patch_size: int) -> int:
        scaled_size = size * scale
        scaled_size = math.ceil(scaled_size / patch_size) * patch_size  # make divisible by patch_size
        scaled_size = max(patch_size, scaled_size)  # ensure at least 1 patch
        return int(scaled_size)

    # Binary search for optimal scale
    scale_min, scale_max = eps / 10, 100.0
    while (scale_max - scale_min) >= eps:
        scale = (scale_min + scale_max) / 2
        target_height = get_scaled_image_size(scale, image_height, patch_size)
        target_width = get_scaled_image_size(scale, image_width, patch_size)
        num_patches = (target_height / patch_size) * (target_width / patch_size)

        if num_patches <= max_num_patches:
            scale_min = scale
        else:
            scale_max = scale

    scale = scale_min
    target_height = get_scaled_image_size(scale, image_height, patch_size)
    target_width = get_scaled_image_size(scale, image_width, patch_size)
    return target_height, target_width


@auto_docstring
class Siglip2ImageProcessorPil(PilBackend):
    valid_kwargs = Siglip2ImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    patch_size = 16
    max_num_patches = 256
    model_input_names = ["pixel_values", "pixel_attention_mask", "spatial_shapes"]

    def __init__(self, **kwargs: Unpack[Siglip2ImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Siglip2ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _validate_preprocess_kwargs(self, **kwargs) -> tuple:
        # Remove do_resize from kwargs to not raise an error as size is None (computed dynamically)
        kwargs.pop("do_resize", None)
        return super()._validate_preprocess_kwargs(**kwargs)

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        patch_size: int,
        max_num_patches: int,
        resample: "PILImageResampling | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        pixel_masks = []
        pixel_values = []
        spatial_shapes = []

        for image in images:
            if do_resize:
                height, width = get_image_size_for_max_num_patches(
                    image_height=image.shape[-2],
                    image_width=image.shape[-1],
                    patch_size=patch_size,
                    max_num_patches=max_num_patches,
                )
                size_dict = SizeDict(height=height, width=width)
                image = self.resize(image=image, size=size_dict, resample=resample)

            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)

            # (num_channels, height, width) -> (num_patches, patch_size * patch_size * num_channels)
            patches = convert_image_to_patches(image, patch_size)
            patches, mask = pad_along_first_dim(patches, max_num_patches)

            num_patches_height = image.shape[-2] // patch_size
            num_patches_width = image.shape[-1] // patch_size

            spatial_shapes.append((num_patches_height, num_patches_width))
            pixel_values.append(patches)
            pixel_masks.append(mask)

        batch_feature = BatchFeature(
            data={
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_masks,
                "spatial_shapes": spatial_shapes,
            },
            tensor_type=return_tensors,
        )
        return batch_feature


__all__ = ["Siglip2ImageProcessorPil"]
