# coding=utf-8
# Copyright 2022 Meta Platforms authors and The HuggingFace Team. All rights reserved.
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
from typing import Any, List, Optional, Tuple, Union

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
class FlavaMaskingGenerator:
    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 14,
        total_mask_patches: int = 75,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_patches: int = 16,
        mask_group_min_aspect_ratio: Optional[float] = 0.3,
        mask_group_max_aspect_ratio: float = None,
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
            height = int(round(math.sqrt(target_area * aspect_ratio)))
            width = int(round(math.sqrt(target_area / aspect_ratio)))
            if width < self.width and height < self.height:
                top = random.randint(0, self.height - height)
                left = random.randint(0, self.width - width)

                num_masked = mask[top : top + height, left : left + width].sum()
                # Overlap
                if 0 < height * width - num_masked <= max_mask_patches:
                    for i in range(top, top + height):
                        for j in range(left, left + width):
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


class FlavaFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
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
        image_mean (`Tuple[float, float, float]`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`Tuple[float, float, float]`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
        input_size_patches (`int`, *optional*, defaults to 14):
            Number of patches in the image in height and width direction. 14x14 = 196 total patches.
        total_mask_patches (`int`, *optional*, defaults to 75):
            Total number of patches that should be masked.
        mask_group_min_patches (`int`, *optional*, defaults to 16):
            Minimum number of patches that should be masked.
        mask_group_max_patches (`int`, *optional*, defaults to None):
            Maximum number of patches that should be masked.
        mask_group_min_aspect_ratio (`float`, *optional*, defaults to 0.3):
            Minimum aspect ratio of the mask window.
        mask_group_max_aspect_ratio (`float`, *optional*, defaults to None):
            Maximum aspect ratio of the mask window
        codebook_do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input for codebook to a certain `codebook_size`.
        codebook_size (`int`, *optional*, defaults to 224):
            Resize the input for codebook to the given size. Only has an effect if `codebook_do_resize` is set to
            `True`.
        codebook_resample (`int`, *optional*, defaults to `PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
        codebook_do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input for codebook at the center. If the input size is smaller than
            `codebook_crop_size` along any edge, the image is padded with 0's and then center cropped.
        codebook_crop_size (`int`, *optional*, defaults to 224):
            Desired output size for codebook input when applying center-cropping. Only has an effect if
            `codebook_do_center_crop` is set to `True`.
        codebook_do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input for codebook with `codebook_image_mean` and `codebook_image_std`.
        codebook_image_mean (`Tuple[float, float, float]`, *optional*, defaults to `[0, 0, 0]`):
            The sequence of means for each channel, to be used when normalizing images for codebook.
        codebook_image_std (`Tuple[float, float, float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of standard deviations for each channel, to be used when normalizing images for codebook.

    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Union[int, Tuple[int, int]] = 224,
        resample: int = Image.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Union[int, Tuple[int, int]] = 224,
        do_normalize: bool = True,
        image_mean: Tuple[float, float, float] = FLAVA_IMAGE_MEAN,
        image_std: Tuple[float, float, float] = FLAVA_IMAGE_STD,
        # Mask related params
        input_size_patches: int = 14,
        total_mask_patches: int = 75,
        mask_group_min_patches: int = 16,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_aspect_ratio: float = 0.3,
        mask_group_max_aspect_ratio: Optional[float] = None,
        # Codebook related params
        codebook_do_resize: bool = True,
        codebook_size: bool = 112,
        codebook_resample: int = Image.LANCZOS,
        codebook_do_center_crop: bool = True,
        codebook_crop_size: int = 112,
        codebook_do_map_pixels: bool = True,
        codebook_do_normalize: bool = True,
        codebook_image_mean: Tuple[float, float, float] = FLAVA_CODEBOOK_MEAN,
        codebook_image_std: Tuple[float, float, float] = FLAVA_CODEBOOK_STD,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

        self.input_size_patches = input_size_patches
        self.total_mask_patches = total_mask_patches
        self.mask_group_min_patches = mask_group_min_patches
        self.mask_group_max_patches = mask_group_max_patches
        self.mask_group_min_aspect_ratio = mask_group_min_aspect_ratio
        self.mask_group_max_aspect_ratio = mask_group_max_aspect_ratio

        self.codebook_do_resize = codebook_do_resize
        self.codebook_size = codebook_size
        self.codebook_resample = codebook_resample
        self.codebook_do_center_crop = codebook_do_center_crop
        self.codebook_crop_size = codebook_crop_size
        self.codebook_do_map_pixels = codebook_do_map_pixels
        self.codebook_do_normalize = codebook_do_normalize
        self.codebook_image_mean = codebook_image_mean
        self.codebook_image_std = codebook_image_std

    @property
    @lru_cache()
    def masking_generator(self):
        return FlavaMaskingGenerator(
            input_size=self.input_size_patches,
            total_mask_patches=self.total_mask_patches,
            mask_group_min_patches=self.mask_group_min_patches,
            mask_group_max_patches=self.mask_group_max_patches,
            mask_group_min_aspect_ratio=self.mask_group_min_aspect_ratio,
            mask_group_max_aspect_ratio=self.mask_group_max_aspect_ratio,
        )

    def map_pixels(self, x):
        return (1 - 2 * LOGIT_LAPLACE_EPS) * x + LOGIT_LAPLACE_EPS

    def __call__(
        self,
        images: Union[
            Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]  # noqa
        ],
        return_image_mask: Optional[bool] = None,
        return_codebook_pixels: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs: Any
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

            return_image_mask (`bool`, *optional*, defaults to None):
                If True, the processor will return `bool_masked_pos` suggesting masks for image's patch version.

            return_codebook_pixels (`bool`, *optional*, defaults to None):
                If True, the processor will return `codebook_pixel_values` providing image pixels to be used with the
                default FLAVA codebook. Used in pretraining by Masked Image Modeling (MIM) loss.

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

        images_for_codebook = images

        # transformations (resizing + center cropping + normalization)
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [self.resize(image=image, size=self.size, resample=self.resample) for image in images]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]
        # return as BatchFeature
        data = {"pixel_values": images}

        if return_codebook_pixels:
            images = images_for_codebook
            if self.codebook_do_resize and self.codebook_size is not None and self.codebook_resample is not None:
                images = [
                    self.resize(image=image, size=self.codebook_size, resample=self.codebook_resample)
                    for image in images
                ]
            if self.codebook_do_center_crop and self.codebook_crop_size is not None:
                images = [self.center_crop(image, self.codebook_crop_size) for image in images]
            if self.codebook_do_normalize:
                images = [
                    self.normalize(image=image, mean=self.codebook_image_mean, std=self.codebook_image_std)
                    for image in images
                ]
            if self.codebook_do_map_pixels:
                images = [self.map_pixels(image) for image in images]

            data["codebook_pixel_values"] = images

        if return_image_mask:
            masks = [self.masking_generator() for _ in images]
            data["bool_masked_pos"] = masks

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
