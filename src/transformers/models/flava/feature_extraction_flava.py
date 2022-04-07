# coding=utf-8
# Copyright 2021 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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
"""Feature extractor class for FLAVA."""

import math
import random
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import ImageFeatureExtractionMixin, is_torch_tensor
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


# These values are taken from CLIP
FLAVA_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
FLAVA_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
FLAVA_CODEBOOK_MEAN = [0.0, 0.0, 0.0]
FLAVA_CODEBOOK_STD = [1.0, 1.0, 1.0]
LOGIT_LAPLACE_EPS: float = 0.1


# Inspired from https://github.com/microsoft/unilm/blob/master/beit/masking_generator.py
class MaskingGenerator:
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        total_mask_patches: int = 75,
        mask_group_max_patches: int = None,
        mask_group_min_patches: Optional[int] = 16,
        mask_group_min_aspect_ratio: float = 0.3,
        mask_group_max_aspect_ratio: Optional[float] = None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.total_mask_patches = total_mask_patches

        self.mask_group_min_patches = mask_group_min_patches
        self.mask_group_max_patches = total_mask_patches if mask_group_max_patches is None else mask_group_max_patches

        mask_group_max_aspect_ratio = mask_group_max_aspect_ratio or 1 / mask_group_min_aspect_ratio
        self.log_aspect_ratio = (math.log(mask_group_min_aspect_ratio), math.log(mask_group_max_aspect_ratio))

    def __repr__(self):
        repr_str = "MaskingGenerator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.mask_group_min_patches,
            self.mask_group_max_patches,
            self.total_mask_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _attempt in range(10):
            target_area = random.uniform(self.mask_group_min_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=int)
        mask_count = 0
        while mask_count < self.total_mask_patches:
            max_mask_patches = self.total_mask_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.mask_group_max_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


class FLAVAFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a FLAVA feature extractor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int`, *optional*, defaults to 224):
            Resize the input to the given size. Only has an effect if `do_resize` is set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
            image is padded with 0's and then center cropped.
        crop_size (`int`, *optional*, defaults to 224):
            Desired output size when applying center-cropping. Only has an effect if `do_center_crop` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with `image_mean` and `image_std`.
        image_mean (`List[int]`, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=224,
        resample=None,
        do_center_crop=True,
        crop_size=224,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        input_size_patches: int = 14,
        total_mask_patches: int = 75,
        mask_group_min_patches: Optional[int] = 16,
        mask_group_max_patches: int = None,
        mask_group_min_aspect_ratio: float = 0.3,
        mask_group_max_aspect_ratio: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample if resample is not None else Image.BICUBIC
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else FLAVA_IMAGE_MEAN
        self.image_std = image_std if image_std is not None else FLAVA_IMAGE_STD
        self.input_size_patches = input_size_patches
        self.total_mask_patches = total_mask_patches
        self.mask_group_min_patches = mask_group_min_patches
        self.mask_group_max_patches = mask_group_max_patches
        self.mask_group_min_aspect_ratio = mask_group_min_aspect_ratio
        self.mask_group_max_aspect_ratio = mask_group_max_aspect_ratio

    @property
    @lru_cache()
    def masking_generator(self):
        return MaskingGenerator(
            input_size=self.input_size_patches,
            total_mask_patches=self.total_mask_patches,
            mask_group_min_patches=self.mask_group_min_patches,
            mask_group_max_patches=self.mask_group_max_patches,
            mask_group_min_aspect_ratio=self.mask_group_min_aspect_ratio,
            mask_group_max_aspect_ratio=self.mask_group_max_aspect_ratio,
        )

    def __call__(
        self,
        images: Union[
            Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]  # noqa
        ],
        return_masks: Optional[bool] = None,
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

            return_masks (`bool`, *optional*, defaults to None):
                If True, the processor will return `bool_masked_pos` suggesting masks for image's patch version.


            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
        """
        # Input type checking for clearer error
        if isinstance(images, (list, tuple)) and len(images) != 0:
            self._ensure_format_supported(images[0])
        else:
            self._ensure_format_supported(images)

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (resizing + center cropping + normalization)
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [self.resize(image=image, size=self.size, resample=self.resample) for image in images]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]
        # return as BatchFeature
        data = {"pixel_values": images}

        if return_masks:
            masks = [self.masking_generator() for _ in images]
            data["bool_masked_pos"] = masks

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs


class FLAVACodebookFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    def __init__(
        self,
        do_resize=True,
        size=112,
        resample=None,
        do_center_crop=True,
        crop_size=112,
        do_map_pixels=True,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample if resample is not None else Image.LANCZOS
        self.do_center_crop = do_center_crop
        self.do_map_pixels = do_map_pixels
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else FLAVA_CODEBOOK_MEAN
        self.image_std = image_std if image_std is not None else FLAVA_CODEBOOK_STD

    def map_pixels(self, x):
        return (1 - 2 * LOGIT_LAPLACE_EPS) * x + LOGIT_LAPLACE_EPS

    def __call__(
        self,
        images: Union[
            Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]  # noqa
        ],
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

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
        """
        # Input type checking for clearer error
        if isinstance(images, (list, tuple)) and len(images) != 0:
            self._ensure_format_supported(images[0])
        else:
            self._ensure_format_supported(images)

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (resizing + center cropping + normalization)
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [self.resize(image=image, size=self.size, resample=self.resample) for image in images]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]
        if self.do_map_pixels:
            images = [self.map_pixels(image) for image in images]
        data = {"pixel_values": images}

        # return as BatchFeature
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
