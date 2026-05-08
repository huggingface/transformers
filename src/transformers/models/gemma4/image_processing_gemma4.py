# Copyright 2026 the HuggingFace Team. All rights reserved.
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


import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, PILImageResampling
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, logging
from .image_processing_pil_gemma4 import _SUPPORTED_SOFT_TOKENS, get_aspect_ratio_preserving_size


logger = logging.get_logger(__name__)


# Copied from transformers.models.siglip2.image_processing_siglip2.convert_image_to_patches
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


# Adopted from Siglip2 (mask -> position ids)
def pad_along_first_dim(
    image: "torch.Tensor", positions: "torch.Tensor", target_length: int
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Pad the tensor along the first dimension.
    """
    current_length = image.shape[0]
    padding_length = target_length - current_length
    if padding_length > 0:
        padding = [0, 0] * (image.ndim - 1) + [0, padding_length]
        pos_padding = (0, 0, 0, padding_length)
        image = torch.nn.functional.pad(image, padding, mode="constant", value=0)
        positions = torch.nn.functional.pad(positions, pos_padding, mode="constant", value=-1)
    return image, positions


class Gemma4ImageProcessorKwargs(ImagesKwargs, total=False):
    """
    patch_size (`int`, *optional*):
        Size of each image patch in pixels.
    max_soft_tokens (`int`, *optional*):
        Maximum number of soft (vision) tokens per image.
        Must be one of {70, 140, 280, 560, 1120}.
    pooling_kernel_size (`int`, *optional*):
        Spatial pooling kernel size applied after patchification.
    """

    patch_size: int
    max_soft_tokens: int
    pooling_kernel_size: int


@auto_docstring(custom_intro="Constructs a Gemma4 image processor.")
class Gemma4ImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]
    size = None
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = False
    patch_size = 16
    max_soft_tokens = 280
    pooling_kernel_size = 3
    valid_kwargs = Gemma4ImageProcessorKwargs
    model_input_names = ["pixel_values", "image_position_ids", "num_soft_tokens_per_image"]

    def __init__(self, **kwargs: Unpack[Gemma4ImageProcessorKwargs]):
        super().__init__(**kwargs)

        if self.max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {self.max_soft_tokens}.")

    def _validate_preprocess_kwargs(self, **kwargs):
        # Gemma4 uses aspect_ratio_preserving_resize driven by patch_size,
        # max_soft_tokens, and pooling_kernel_size — not the standard `size`
        # parameter. Temporarily disable do_resize so the base validation
        # doesn't require `size` to be set.
        kwargs["do_resize"] = False
        super()._validate_preprocess_kwargs(**kwargs)

    def aspect_ratio_preserving_resize(
        self,
        image: torch.Tensor,
        patch_size: int,
        max_patches: int,
        pooling_kernel_size: int,
        resample: F.InterpolationMode,
    ) -> torch.Tensor:
        height, width = image.shape[-2], image.shape[-1]
        target_height, target_width = get_aspect_ratio_preserving_size(
            height=height,
            width=width,
            patch_size=patch_size,
            max_patches=max_patches,
            pooling_kernel_size=pooling_kernel_size,
        )

        if target_height == height and target_width == width:
            return image

        return F.resize(
            image,
            size=[target_height, target_width],
            interpolation=resample,
            antialias=True,
        )

    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Gemma4ImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        resample: "PILImageResampling | F.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        patch_size: int | None = None,
        max_soft_tokens: int | None = None,
        pooling_kernel_size: int | None = None,
        **kwargs,
    ) -> BatchFeature:
        if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
            raise ValueError(f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}.")

        # Compute max_patches from max_soft_tokens and pooling_kernel_size
        max_patches = max_soft_tokens * pooling_kernel_size**2

        # Process each image individually: resize, rescale/normalize, patchify, pad.
        # Images have different aspect ratios and thus different resized dimensions,
        # so patchification and padding must happen per-image before stacking.
        pixel_values = []
        position_ids = []
        num_soft_tokens_per_image = []

        for image in images:
            # Step 1: Aspect-ratio-preserving resize
            if do_resize:
                image = self.aspect_ratio_preserving_resize(
                    image=image,
                    patch_size=patch_size,
                    max_patches=max_patches,
                    pooling_kernel_size=pooling_kernel_size,
                    resample=resample,
                )

            # Step 2: Rescale pixel values (typically to [0, 1]) and optionally identity normalize
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

            # Step 3: Patchify the image
            # (num_channels, height, width) -> (num_patches, patch_size * patch_size * num_channels)
            patch_height = image.shape[-2] // patch_size
            patch_width = image.shape[-1] // patch_size
            patches = convert_image_to_patches(image, patch_size)
            num_soft_tokens_per_image.append(patches.shape[0] // pooling_kernel_size**2)

            # Step 5: Compute position IDs
            device = image.device
            patch_grid = torch.meshgrid(
                torch.arange(patch_width, device=device),
                torch.arange(patch_height, device=device),
                indexing="xy",
            )
            stacked_grid = torch.stack(patch_grid, dim=-1)
            real_positions = stacked_grid.reshape(patches.shape[0], 2)

            # Step 6. Pad pacthes and positions to `max_patches`
            patches, positions = pad_along_first_dim(patches, real_positions, max_patches)
            pixel_values.append(patches)
            position_ids.append(positions)

        # Stack into batch tensors
        pixel_values = torch.stack(pixel_values, dim=0)  # (batch, max_patches, patch_pixels)
        position_ids = torch.stack(position_ids, dim=0)  # (batch, max_patches, 2)

        data = {
            "pixel_values": pixel_values,
            "image_position_ids": position_ids,
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Gemma4ImageProcessor"]
