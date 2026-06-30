# coding=utf-8
# Copyright 2026 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for LocateAnything (MoonViT native-resolution patchification)."""

import math
from typing import Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_utils import ImageInput, make_list_of_images, to_numpy_array, valid_images
from ...utils import TensorType, is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


class LocateAnythingImageProcessor(BaseImageProcessor):
    r"""
    Constructs a LocateAnything image processor.

    LocateAnything uses a MoonViT native-resolution vision encoder. Each image is rescaled so that the number of
    patches does not exceed `in_token_limit`, padded to a multiple of `merge_kernel_size * patch_size`, normalized,
    and finally split into non-overlapping `patch_size x patch_size` patches.

    Args:
        patch_size (`int`, *optional*, defaults to 14):
            The spatial size of each patch.
        image_mean (`tuple[float, float, float]`, *optional*, defaults to `(0.5, 0.5, 0.5)`):
            Mean used when normalizing the image.
        image_std (`tuple[float, float, float]`, *optional*, defaults to `(0.5, 0.5, 0.5)`):
            Standard deviation used when normalizing the image.
        in_token_limit (`int`, *optional*, defaults to 4096):
            Maximum number of patches kept per image. Larger images are downscaled to satisfy this limit.
        merge_kernel_size (`list[int]`, *optional*, defaults to `[2, 2]`):
            The patch merger kernel size; images are padded so their patch grid is divisible by this kernel.
    """

    model_input_names = ["pixel_values", "image_grid_hws"]

    def __init__(
        self,
        patch_size: int = 14,
        image_mean: tuple[float, float, float] = MEAN,
        image_std: tuple[float, float, float] = STD,
        in_token_limit: int = 4096,
        merge_kernel_size: list[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_token_limit = in_token_limit
        self.patch_size = patch_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.merge_kernel_size = merge_kernel_size if merge_kernel_size is not None else [2, 2]

    def _rescale_image(self, image):
        """Resize a PIL image so its patch grid fits within `in_token_limit` and is divisible by the merge kernel."""
        from PIL import Image

        w, h = image.size
        patch_size = self.patch_size

        if (w // patch_size) * (h // patch_size) > self.in_token_limit:
            scale = math.sqrt(self.in_token_limit / ((w // patch_size) * (h // patch_size)))
            image = image.resize((max(int(w * scale), patch_size), max(int(h * scale), patch_size)), Image.BICUBIC)

        new_w, new_h = image.size
        pad_size_h = self.merge_kernel_size[0] * patch_size
        pad_size_w = self.merge_kernel_size[1] * patch_size
        target_w = max(math.ceil(new_w / pad_size_w), 1) * pad_size_w
        target_h = max(math.ceil(new_h / pad_size_h), 1) * pad_size_h
        if target_w != new_w or target_h != new_h:
            image = image.resize((target_w, target_h), Image.BICUBIC)

        w, h = image.size
        if w // patch_size >= 512 or h // patch_size >= 512:
            raise ValueError("Image exceeds the maximum supported positional embedding size.")
        return image

    def _patchify(self, image: "torch.Tensor"):
        patch_size = self.patch_size
        channels, height, width = image.shape
        patches = image.reshape(channels, height // patch_size, patch_size, width // patch_size, patch_size)
        patches = patches.permute(1, 3, 0, 2, 4)
        patches = patches.contiguous().view(-1, channels, patch_size, patch_size)
        grid_hw = (height // patch_size, width // patch_size)
        return patches, grid_hw

    def _preprocess_single(self, image):
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(to_numpy_array(image).astype("uint8"))
        image = image.convert("RGB")
        image = self._rescale_image(image)

        array = to_numpy_array(image).astype("float32") / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor(self.image_mean).view(-1, 1, 1)
        std = torch.tensor(self.image_std).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        return self._patchify(tensor)

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or "
                "jax.ndarray."
            )

        pixel_values, image_grid_hws = [], []
        for image in images:
            patches, grid_hw = self._preprocess_single(image)
            pixel_values.append(patches)
            image_grid_hws.append(grid_hw)

        pixel_values = torch.cat(pixel_values, dim=0)
        image_grid_hws = np.array(image_grid_hws)
        data = {"pixel_values": pixel_values, "image_grid_hws": image_grid_hws}
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["LocateAnythingImageProcessor"]
