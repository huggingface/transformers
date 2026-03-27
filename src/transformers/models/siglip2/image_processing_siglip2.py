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

import torch

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torchvision_available,
)


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


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


def convert_image_to_patches(image: "torch.Tensor", patch_size: int) -> "torch.Tensor":
    """
    Convert 3D tensor image of shape (num_channels, image_height, image_width) into 2D tensor of patches of shape
    (num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
    """
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.permute(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def pad_along_first_dim(
    tensor: "torch.Tensor", target_length: int, pad_value: int = 0
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Pad the tensor along the first dimension.
    """
    current_length = tensor.shape[0]
    padding_length = target_length - current_length
    mask = torch.ones((target_length,), dtype=torch.int32)
    if padding_length > 0:
        padding = [0, 0] * (tensor.ndim - 1) + [0, padding_length]
        tensor = torch.nn.functional.pad(tensor, padding, mode="constant", value=pad_value)
        mask[-padding_length:] = 0
    return tensor, mask


@auto_docstring
class Siglip2ImageProcessor(TorchvisionBackend):
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
        images: list["torch.Tensor"],
        do_resize: bool,
        patch_size: int,
        max_num_patches: int,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
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

            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

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


__all__ = ["Siglip2ImageProcessor"]
