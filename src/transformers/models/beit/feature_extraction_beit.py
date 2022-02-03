# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for BEiT."""

from typing import Optional, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...file_utils import TensorType
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageFeatureExtractionMixin,
    ImageInput,
    is_torch_tensor,
)
from ...utils import logging


logger = logging.get_logger(__name__)


class BeitFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a BEiT feature extractor.

    This feature extractor inherits from [`~feature_extraction_utils.FeatureExtractionMixin`] which contains most of
    the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int` or `Tuple(int)`, *optional*, defaults to 256):
            Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
            integer is provided, then the input will be resized to (size, size). Only has an effect if `do_resize` is
            set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
            image is padded with 0's and then center cropped.
        crop_size (`int`, *optional*, defaults to 224):
            Desired output size when applying center-cropping. Only has an effect if `do_center_crop` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with `image_mean` and `image_std`.
        image_mean (`List[int]`, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is
            used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The
            background label will be replaced by 255.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=256,
        resample=Image.BICUBIC,
        do_center_crop=True,
        crop_size=224,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        reduce_labels=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.reduce_labels = reduce_labels

    def __call__(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput = None,
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

            segmentation_maps (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                Optionally, the corresponding semantic segmentation maps with the pixel-wise annotations.

            return_tensors (`str` or [`~file_utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
              width).
            - **labels** -- Optional labels to be fed to a model (when `segmentation_maps` are provided)
        """
        # Input type checking for clearer error
        valid_images = False
        valid_segmentation_maps = False

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

        # Check that segmentation maps has a valid type
        if segmentation_maps is not None:
            if isinstance(segmentation_maps, (Image.Image, np.ndarray)) or is_torch_tensor(segmentation_maps):
                valid_segmentation_maps = True
            elif isinstance(segmentation_maps, (list, tuple)):
                if (
                    len(segmentation_maps) == 0
                    or isinstance(segmentation_maps[0], (Image.Image, np.ndarray))
                    or is_torch_tensor(segmentation_maps[0])
                ):
                    valid_segmentation_maps = True

            if not valid_segmentation_maps:
                raise ValueError(
                    "Segmentation maps must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example),"
                    "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
                )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]
            if segmentation_maps is not None:
                segmentation_maps = [segmentation_maps]

        # reduce zero label if needed
        if self.reduce_labels:
            if segmentation_maps is not None:
                for idx, map in enumerate(segmentation_maps):
                    if not isinstance(map, np.ndarray):
                        map = np.array(map)
                    # avoid using underflow conversion
                    map[map == 0] = 255
                    map = map - 1
                    map[map == 254] = 255
                    segmentation_maps[idx] = Image.fromarray(map.astype(np.uint8))

        # transformations (resizing + center cropping + normalization)
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [self.resize(image=image, size=self.size, resample=self.resample) for image in images]
            if segmentation_maps is not None:
                segmentation_maps = [
                    self.resize(map, size=self.size, resample=self.resample) for map in segmentation_maps
                ]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
            if segmentation_maps is not None:
                segmentation_maps = [self.center_crop(map, size=self.crop_size) for map in segmentation_maps]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}

        if segmentation_maps is not None:
            labels = []
            for map in segmentation_maps:
                if not isinstance(map, np.ndarray):
                    map = np.array(map)
                labels.append(map.astype(np.int64))
            # cast to np.int64
            data["labels"] = labels

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
