# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for ViLT."""

from typing import List, Optional, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageFeatureExtractionMixin,
    ImageInput,
    is_torch_tensor,
)
from ...utils import TensorType, is_torch_available, logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class ViltFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a ViLT feature extractor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input based on `size`.
        size (`int`, *optional*, defaults to 384):
            Resize the shorter side of the input to the given size. Should be an integer. The longer side will be
            limited to under int((1333 / 800) * size) while preserving the aspect ratio. Only has an effect if
            `do_resize` is set to `True`.
        size_divisor (`int`, *optional*, defaults to 32):
            The size by which to make sure both the height and width can be divided.
        resample (`int`, *optional*, defaults to `PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`List[int]`, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values", "pixel_mask"]

    def __init__(
        self,
        do_resize=True,
        size=384,
        size_divisor=32,
        resample=Image.BICUBIC,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.size_divisor = size_divisor
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def _resize(self, image, shorter=800, longer=1333, size_divisor=32, resample=Image.BICUBIC):
        """
        Resizes the shorter edge of `image` to `shorter` and limits the longer edge to under `longer`, while preserving
        the aspect ratio. Also makes sure that both the height and width can be divided by `size_divisor`.

        Based on original implementation:
        https://github.com/dandelin/ViLT/blob/3db8b5035464afee84d951bf6322e1b27f1d072d/vilt/transforms/utils.py#L5

        Args:
            image (`PIL.Image`):
                The image to resize.
            shorter (`int`, *optional*, defaults to `800`):
                The size to which to resize the shorter side of the image.
            longer (`int`, *optional*, defaults to `1333`):
                The size by which to limit the longer side of the image, while preserving the aspect ratio.
            size_divisor (`int`, *optional*, defaults to `32`):
                The size by which both the height and the width must be divisible.
            resample (`int`, *optional*, defaults to `PIL.Image.BICUBIC`):
                An optional resampling filter.
        """
        if not isinstance(image, Image.Image):
            image = self.to_pil_image(image)

        w, h = image.size
        min_size = shorter
        max_size = longer
        scale = min_size / min(w, h)
        if h < w:
            newh, neww = min_size, scale * w
        else:
            newh, neww = scale * h, min_size

        if max(newh, neww) > max_size:
            scale = max_size / max(newh, neww)
            newh = newh * scale
            neww = neww * scale

        newh, neww = int(newh + 0.5), int(neww + 0.5)
        newh, neww = newh // size_divisor * size_divisor, neww // size_divisor * size_divisor

        return self.resize(image, size=(neww, newh), resample=resample)

    def _max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def pad_and_create_pixel_mask(
        self, pixel_values_list: List["torch.Tensor"], return_tensors: Optional[Union[str, TensorType]] = None
    ):
        """
        Pad images up to the largest image in a batch and create a corresponding `pixel_mask`.

        Args:
            pixel_values_list (`List[torch.Tensor]`):
                List of images (pixel values) to be padded. Each image should be a tensor of shape (C, H, W).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of NumPy arrays. If set to `'pt'`, return PyTorch `torch.Tensor`
                objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
            - **pixel_mask** -- Pixel mask to be fed to a model (when `pad_and_return_pixel_mask=True` or if
              *"pixel_mask"* is in `self.model_input_names`).
        """

        max_size = self._max_by_axis([list(image.shape) for image in pixel_values_list])
        c, h, w = max_size
        padded_images = []
        pixel_mask = []
        for image in pixel_values_list:
            # create padded image
            padded_image = np.zeros((c, h, w), dtype=np.float32)
            padded_image[: image.shape[0], : image.shape[1], : image.shape[2]] = np.copy(image)
            padded_images.append(padded_image)
            # create pixel mask
            mask = np.zeros((h, w), dtype=np.int64)
            mask[: image.shape[1], : image.shape[2]] = True
            pixel_mask.append(mask)

        # return as BatchFeature
        data = {"pixel_values": padded_images, "pixel_mask": pixel_mask}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def __call__(
        self,
        images: ImageInput,
        pad_and_return_pixel_mask: Optional[bool] = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            pad_and_return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether or not to pad images up to the largest image in a batch and create a pixel mask.

                If left to the default, will return a pixel mask that is:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
              width).
            - **pixel_mask** -- Pixel mask to be fed to a model (when `return_pixel_mask=True` or if *"pixel_mask"* is
              in `self.model_input_names`).
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (resizing + normalization)
        if self.do_resize and self.size is not None:
            longer = int((1333 / 800) * self.size)
            images = [
                self._resize(
                    image=image,
                    shorter=self.size,
                    longer=longer,
                    size_divisor=self.size_divisor,
                    resample=self.resample,
                )
                for image in images
            ]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        if pad_and_return_pixel_mask:
            # pad images up to largest image in batch and create pixel_mask
            max_size = self._max_by_axis([list(image.shape) for image in images])
            c, h, w = max_size
            padded_images = []
            pixel_mask = []
            for image in images:
                # create padded image
                padded_image = np.zeros((c, h, w), dtype=np.float32)
                padded_image[: image.shape[0], : image.shape[1], : image.shape[2]] = np.copy(image)
                padded_images.append(padded_image)
                # create pixel mask
                mask = np.zeros((h, w), dtype=np.int64)
                mask[: image.shape[1], : image.shape[2]] = True
                pixel_mask.append(mask)
            images = padded_images

        # return as BatchFeature
        data = {}
        data["pixel_values"] = images
        if pad_and_return_pixel_mask:
            data["pixel_mask"] = pixel_mask
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
