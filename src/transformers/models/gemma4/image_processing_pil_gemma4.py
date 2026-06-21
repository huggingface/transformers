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

import math

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import resize
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_vision_available, logging


if is_vision_available():
    from ...image_utils import PILImageResampling


logger = logging.get_logger(__name__)

_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    """
    Image is resized to preserve aspect ratio so it fits within the patch budget.
    Target dimensions are the largest that:
    1) Produce at most `max_patches` patches when patchified with `patch_size`
    2) Have height and width divisible by `pooling_kernel_size * patch_size`
    """
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    # Round down to nearest multiple of side_mult
    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    # Handle edge cases where one or both dimensions round to 0
    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. Resized height should be divisble by "
            f"`pooling_kernel_size * patch_size`={pooling_kernel_size * patch_size}."
        )

    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(
            int(math.floor(width / height)) * side_mult,
            max_side_length,
        )
    elif target_width == 0:
        target_width = side_mult
        target_height = min(
            int(math.floor(height / width)) * side_mult,
            max_side_length,
        )

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resizing [{height}x{width}] to [{target_height}x{target_width}] "
            f"but this exceeds {max_patches} patches with patch_size {patch_size}"
        )

    return target_height, target_width


# Copied from transformers.models.siglip2.image_processing_pil_siglip2.convert_image_to_patches
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


# Adopted from Siglip2 (mask -> position ids)
def pad_along_first_dim(image: np.ndarray, positions: np.ndarray, target_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad the image along the first dimension.
    """
    current_length = image.shape[0]
    padding_length = target_length - current_length
    if padding_length > 0:
        paddings = [(0, padding_length)] + [(0, 0)] * (image.ndim - 1)
        pos_paddings = [(0, padding_length), (0, 0)]
        image = np.pad(image, paddings, mode="constant", constant_values=0)
        positions = np.pad(positions, pos_paddings, mode="constant", constant_values=-1)
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
class Gemma4ImageProcessorPil(PilBackend):
    valid_kwargs = Gemma4ImageProcessorKwargs
    model_input_names = ["pixel_values", "image_position_ids", "num_soft_tokens_per_image"]

    do_resize = True
    resample = PILImageResampling.BICUBIC
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = False
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]
    do_convert_rgb = True
    patch_size = 16
    max_soft_tokens = 280
    pooling_kernel_size = 3

    def __init__(self, **kwargs: Unpack[Gemma4ImageProcessorKwargs]) -> None:
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

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Gemma4ImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def aspect_ratio_preserving_resize(
        self,
        image: np.ndarray,
        patch_size: int,
        max_patches: int,
        pooling_kernel_size: int,
        resample: PILImageResampling,
    ) -> np.ndarray:
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

        return resize(
            image,
            size=(target_height, target_width),
            resample=resample,
        )

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        resample: "PILImageResampling | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        max_soft_tokens: int | None = None,
        patch_size: int | None = None,
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

            # Step 2: Rescale pixel values from [0, 255] to [0, 1]
            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor)

            # Step 3: Identity normalization because Gemma4 was trained with pixels in [0, 1]
            if do_normalize:
                image = self.normalize(image=image, mean=image_mean, std=image_std)

            # Step 4: Patchify the image
            # image is (C, H, W) numpy array; add batch dimension for reshape
            # (num_channels, height, width) -> (num_patches, patch_size * patch_size * num_channels)
            patches = convert_image_to_patches(image, patch_size)
            num_soft_tokens_per_image.append(patches.shape[0] // pooling_kernel_size**2)

            # Step 5: Compute position IDs
            patch_height = image.shape[-2] // patch_size
            patch_width = image.shape[-1] // patch_size
            grid_x, grid_y = np.meshgrid(np.arange(patch_width), np.arange(patch_height), indexing="xy")
            real_positions = np.stack([grid_x, grid_y], axis=-1).reshape(patches.shape[0], 2)

            patches, positions = pad_along_first_dim(patches, real_positions, max_patches)

            pixel_values.append(patches)
            position_ids.append(positions)

        # Stack into batch arrays and convert to tensors
        pixel_values = np.stack(pixel_values, axis=0)  # (batch, max_patches, patch_pixels)
        position_ids = np.stack(position_ids, axis=0)  # (batch, max_patches, 2)

        data = {
            "pixel_values": pixel_values,
            "image_position_ids": position_ids,
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Gemma4ImageProcessorPil"]
