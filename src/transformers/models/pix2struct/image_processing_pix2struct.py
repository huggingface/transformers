# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for PIX2STRUCT."""
import math
from typing import Dict, Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import convert_to_rgb, normalize
from ...image_utils import ChannelDimension, ImageInput, is_batched, to_numpy_array, valid_images
from ...utils import TensorType, is_torch_available, is_vision_available, logging
from ...utils.import_utils import requires_backends


if is_vision_available():
    import PIL

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class Pix2StructImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Pix2Struct image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method. According to
            Pix2Struct paper and code, the image is normalized with its own mean and standard deviation.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        patch_size: Dict[str, int] = {"height": 16, "width": 16},
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb

    # adapted from: https://discuss.pytorch.org/t/tf-image-extract-patches-in-pytorch/171409/2
    def torch_extract_patches(self, image_tensor, patch_height, patch_width):
        """
        Utiliy function to extract patches from a given image tensor. Returns a tensor of shape (1, `patch_height`,
        `patch_width`, `num_channels`x `patch_height` x `patch_width`)

        Args:
            image_tensor (torch.Tensor):
                The image tensor to extract patches from.
            patch_height (int):
                The height of the patches to extract.
            patch_width (int):
                The width of the patches to extract.
        """
        requires_backends(self.torch_extract_patches, ["torch"])

        image_tensor = image_tensor.unsqueeze(0)
        patches = torch.nn.functional.unfold(
            image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width)
        )
        patches = patches.reshape(image_tensor.size(0), image_tensor.size(1), patch_height, patch_width, -1)
        patches = patches.permute(0, 4, 2, 3, 1).reshape(
            image_tensor.size(2) // patch_height,
            image_tensor.size(3) // patch_width,
            image_tensor.size(1) * patch_height * patch_width,
        )
        return patches.unsqueeze(0)

    def extract_flattened_patches(self, image: np.ndarray, max_patches: int, patch_size: dict, **kwargs) -> np.ndarray:
        """
        Extract flattened patches from an image.

        Args:
            image (`np.ndarray`):
                Image to extract flattened patches from.
            max_patches (`int`):
                Maximum number of patches to extract.
            patch_size (`dict`):
                Dictionary containing the patch height and width.
        """
        requires_backends(self.extract_flattened_patches, "torch")

        # convert to torch if
        if not isinstance(image, torch.Tensor):
            # check if channels first:
            if image.shape[0] != 3:
                image = torch.from_numpy(image).permute(2, 0, 1)
            else:
                image = torch.from_numpy(image)

        patch_height, patch_width = patch_size["height"], patch_size["width"]
        _, image_height, image_width = image.shape
        image_height = float(image_height)
        image_width = float(image_width)

        # maximize scale s.t.
        scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
        num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
        num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
        resized_height = max(num_feasible_rows * patch_height, 1)
        resized_width = max(num_feasible_cols * patch_width, 1)

        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

        # [1, rows, columns, patch_height * patch_width * image_channels]
        patches = self.torch_extract_patches(image, patch_height, patch_width)

        patches_shape = patches.shape
        rows = patches_shape[1]
        columns = patches_shape[2]
        depth = patches_shape[3]

        # [rows * columns, patch_height * patch_width * image_channels]
        patches = patches.reshape([rows * columns, depth])

        # [rows * columns, 1]
        row_ids = torch.arange(rows).reshape([rows, 1]).repeat(1, columns).reshape([rows * columns, 1])
        col_ids = torch.arange(columns).reshape([1, columns]).repeat(rows, 1).reshape([rows * columns, 1])

        # Offset by 1 so the ids do not contain zeros, which represent padding.
        row_ids += 1
        col_ids += 1

        # Prepare additional patch features.
        # [rows * columns, 1]
        row_ids = row_ids.to(torch.float32)
        col_ids = col_ids.to(torch.float32)

        # [rows * columns, 2 + patch_height * patch_width * image_channels]
        result = torch.cat([row_ids, col_ids, patches], -1)

        # [max_patches, 2 + patch_height * patch_width * image_channels]
        result = torch.nn.functional.pad(result, [0, 0, 0, max_patches - (rows * columns)]).float()

        result = to_numpy_array(result)

        return result

    def normalize(
        self, image: np.ndarray, data_format: Optional[Union[str, ChannelDimension]] = None, **kwargs
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            data_format (`str`):
                The data format of the image. Can be either "ChannelDimension.channels_first" or
                "ChannelDimension.channels_last".
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32)

        # take mean across the whole `image`
        mean = np.mean(image)
        std = np.std(image)
        adjusted_stddev = max(std, 1.0 / math.sqrt(np.prod(image.shape)))

        return normalize(image, mean=mean, std=adjusted_stddev, data_format=data_format, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        max_patches: Optional[int] = 2048,
        patch_size: Optional[Dict[str, int]] = None,
        do_normalize: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        do_convert_rgb: bool = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Controls the size of the image after `resize`. The shortest edge of the image is resized to
                `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
                is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
                edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        patch_size = patch_size if patch_size is not None else self.patch_size

        if not is_batched(images):
            images = [images]

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # PIL RGBA images are converted to RGB
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, data_format=data_format) for image in images]

        # convert to torch tensor and permute
        images = [
            self.extract_flattened_patches(image=image, max_patches=max_patches, patch_size=patch_size)
            for image in images
        ]

        # create attention mask in numpy
        attention_masks = [(image.sum(axis=-1) != 0).astype(np.float32) for image in images]

        encoded_outputs = BatchFeature(
            data={"pixel_embeds": images, "attention_mask": attention_masks}, tensor_type=return_tensors
        )

        return encoded_outputs
