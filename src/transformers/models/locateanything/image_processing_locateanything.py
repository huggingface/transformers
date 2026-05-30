# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Image processor class for LocateAnything."""

import math

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ImageInput, make_list_of_images, valid_images
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType


MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

__all__ = ["LocateAnythingImageProcessor"]


class LocateAnythingImagesKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to 14):
        Patch size used to split the resized image into vision patches.
    in_token_limit (`int`, *optional*, defaults to 4096):
        Maximum number of image patch tokens before resizing the image.
    merge_kernel_size (`list[int]`, *optional*, defaults to `[2, 2]`):
        Spatial merge kernel size used by the vision encoder.
    """

    patch_size: int
    in_token_limit: int
    merge_kernel_size: list[int] | tuple[int, int]


class LocateAnythingImageProcessor(BaseImageProcessor):
    model_type = "locateanything"
    valid_kwargs = LocateAnythingImagesKwargs

    def __init__(
        self,
        patch_size: int = 14,
        image_mean: tuple[float, float, float] = MEAN,
        image_std: tuple[float, float, float] = STD,
        in_token_limit: int = 4096,
        merge_kernel_size: list[int] | tuple[int, int] | None = None,
        **kwargs: Unpack[LocateAnythingImagesKwargs],
    ):
        super().__init__(**kwargs)
        self.in_token_limit = in_token_limit
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.merge_kernel_size = merge_kernel_size if merge_kernel_size is not None else [2, 2]

    def rescale(self, image: Image.Image, merge_kernel_size: list[int] | tuple[int, int] | None = None) -> Image.Image:
        if merge_kernel_size is None:
            merge_kernel_size = self.merge_kernel_size
        w, h = image.size
        patch_size = self.patch_size

        if (w // patch_size) * (h // patch_size) > self.in_token_limit:
            scale = math.sqrt(self.in_token_limit / ((w // patch_size) * (h // patch_size)))
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        new_w, new_h = image.size
        pad_size_h = merge_kernel_size[0] * patch_size
        pad_size_w = merge_kernel_size[1] * patch_size

        target_w = math.ceil(new_w / pad_size_w) * pad_size_w
        target_h = math.ceil(new_h / pad_size_h) * pad_size_h

        if target_w != new_w or target_h != new_h:
            image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)

        w, h = image.size
        if w // patch_size >= 512 or h // patch_size >= 512:
            raise ValueError("Exceed pos emb")

        return image

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        return TF.to_tensor(image.convert("RGB"))

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        return TF.normalize(image, self.image_mean, self.image_std)

    def patchify(self, image: torch.Tensor) -> tuple[torch.Tensor, list[int, int]]:
        patch_size = self.patch_size
        C, H, W = image.shape
        patches = image.reshape(C, H // patch_size, patch_size, W // patch_size, patch_size)
        patches = patches.permute(1, 3, 0, 2, 4)
        patches = patches.contiguous().view(-1, C, patch_size, patch_size)
        grid_hw = (H // patch_size, W // patch_size)
        return patches, grid_hw

    def _preprocess(self, image: ImageInput) -> tuple[torch.Tensor, list[int, int]]:
        """
        Preprocess image and patchify it.
        Args:
            image (`ImageInput`):
                Image to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
        Returns:
            patches: torch.Tensor
            grid_hw: list[int, int]
        """
        image = self.rescale(image, self.merge_kernel_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        patches, grid_hw = self.patchify(image)
        return patches, grid_hw

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature:
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        pixel_values, image_grid_hws = [], []
        for image in images:
            patches, image_grid_hw = self._preprocess(image)
            pixel_values.append(patches)
            image_grid_hws.append(image_grid_hw)
        pixel_values = torch.concat(pixel_values, dim=0)
        image_grid_hws = np.array(image_grid_hws)
        data = {"pixel_values": pixel_values, "image_grid_hws": image_grid_hws}

        return BatchFeature(data=data, tensor_type=return_tensors)
