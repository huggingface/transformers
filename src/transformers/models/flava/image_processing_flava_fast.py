# coding=utf-8
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
"""Fast Image processor class for Flava."""

import math
import random
from collections.abc import Iterable
from functools import lru_cache
from typing import Any, Optional, Union

import torch
from torchvision.transforms.v2 import functional as F

from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    BatchFeature,
    DefaultFastImageProcessorKwargs,
    get_size_dict,
)
from ...image_transforms import ChannelDimension, group_images_by_shape, reorder_images
from ...image_utils import ImageInput, PILImageResampling, SizeDict, pil_torch_interpolation_mapping
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)
from .image_processing_flava import (
    FLAVA_CODEBOOK_MEAN,
    FLAVA_CODEBOOK_STD,
    FLAVA_IMAGE_MEAN,
    FLAVA_IMAGE_STD,
    LOGIT_LAPLACE_EPS,
)


class FlavaMaskingGenerator:
    def __init__(
        self,
        input_size: Union[int, tuple[int, int]] = 14,
        total_mask_patches: int = 75,
        mask_group_max_patches: Optional[int] = None,
        mask_group_min_patches: int = 16,
        mask_group_min_aspect_ratio: Optional[float] = 0.3,
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
            height = int(round(math.sqrt(target_area * aspect_ratio)))
            width = int(round(math.sqrt(target_area / aspect_ratio)))
            if width < self.width and height < self.height:
                top = random.randint(0, self.height - height)
                left = random.randint(0, self.width - width)

                num_masked = mask[top : top + height, left : left + width].sum()
                # Overlap
                if 0 < height * width - num_masked <= max_mask_patches:
                    zeros_pos = mask[top : top + height, left : left + width] == 0
                    mask[top : top + height, left : left + width][zeros_pos] = 1
                    delta += zeros_pos.sum()

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = torch.zeros(self.get_shape(), dtype=torch.int)
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


class FlavaFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    Args:
        return_image_mask (`bool`, *optional*, defaults to `False`):
            Whether to return the image mask. Can be overridden by the `return_image_mask` parameter in `preprocess`.
        input_size_patches (`int`, *optional*, defaults to 14):
            Number of patches in the image in height and width direction. 14x14 = 196 total patches. Can be overridden
            by the `input_size_patches` parameter in `preprocess`.
        total_mask_patches (`int`, *optional*, defaults to 75):
            Total number of patches that should be masked. Can be overridden by the `total_mask_patches` parameter in
            `preprocess`.
        mask_group_min_patches (`int`, *optional*, defaults to 16):
            Minimum number of patches that should be masked. Can be overridden by the `mask_group_min_patches`
            parameter in `preprocess`.
        mask_group_max_patches (`int`, *optional*):
            Maximum number of patches that should be masked. Can be overridden by the `mask_group_max_patches`
            parameter in `preprocess`.
        mask_group_min_aspect_ratio (`float`, *optional*, defaults to 0.3):
            Minimum aspect ratio of the mask window. Can be overridden by the `mask_group_min_aspect_ratio` parameter
            in `preprocess`.
        mask_group_max_aspect_ratio (`float`, *optional*):
            Maximum aspect ratio of the mask window. Can be overridden by the `mask_group_max_aspect_ratio` parameter
            in `preprocess`.
        return_codebook_pixels (`bool`, *optional*, defaults to `False`):
            Whether to return the codebook pixel values.
        codebook_do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input for codebook to a certain. Can be overridden by the `codebook_do_resize`
            parameter in `preprocess`. `codebook_size`.
        codebook_size (`dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Resize the input for codebook to the given size. Can be overridden by the `codebook_size` parameter in
            `preprocess`.
        codebook_resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.LANCZOS`):
            Resampling filter to use if resizing the codebook image. Can be overridden by the `codebook_resample`
            parameter in `preprocess`.
        codebook_do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input for codebook at the center. If the input size is smaller than
            `codebook_crop_size` along any edge, the image is padded with 0's and then center cropped. Can be
            overridden by the `codebook_do_center_crop` parameter in `preprocess`.
        codebook_crop_size (`dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Desired output size for codebook input when applying center-cropping. Can be overridden by the
            `codebook_crop_size` parameter in `preprocess`.
        codebook_do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input for codebook by the specified scale `codebook_rescale_factor`. Can be
            overridden by the `codebook_do_rescale` parameter in `preprocess`.
        codebook_rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Defines the scale factor to use if rescaling the codebook image. Can be overridden by the
            `codebook_rescale_factor` parameter in `preprocess`.
        codebook_do_map_pixels (`bool`, *optional*, defaults to `True`):
            Whether to map the pixel values of the codebook input to (1 - 2e)x + e. Can be overridden by the
            `codebook_do_map_pixels` parameter in `preprocess`.
        codebook_do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input for codebook with `codebook_image_mean` and `codebook_image_std`. Can
            be overridden by the `codebook_do_normalize` parameter in `preprocess`.
        codebook_image_mean (`Optional[Union[float, Iterable[float]]]`, *optional*, defaults to `[0, 0, 0]`):
            The sequence of means for each channel, to be used when normalizing images for codebook. Can be overridden
            by the `codebook_image_mean` parameter in `preprocess`.
        codebook_image_std (`Optional[Union[float, Iterable[float]]]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of standard deviations for each channel, to be used when normalizing images for codebook. Can
            be overridden by the `codebook_image_std` parameter in `preprocess`.
    """

    # Mask related params
    return_image_mask: Optional[bool]
    input_size_patches: Optional[int]
    total_mask_patches: Optional[int]
    mask_group_min_patches: Optional[int]
    mask_group_max_patches: Optional[int]
    mask_group_min_aspect_ratio: Optional[float]
    mask_group_max_aspect_ratio: Optional[float]
    # Codebook related params
    return_codebook_pixels: Optional[bool]
    codebook_do_resize: Optional[bool]
    codebook_size: Optional[bool]
    codebook_resample: Optional[int]
    codebook_do_center_crop: Optional[bool]
    codebook_crop_size: Optional[int]
    codebook_do_rescale: Optional[bool]
    codebook_rescale_factor: Optional[Union[int, float]]
    codebook_do_map_pixels: Optional[bool]
    codebook_do_normalize: Optional[bool]
    codebook_image_mean: Optional[Union[float, Iterable[float]]]
    codebook_image_std: Optional[Union[float, Iterable[float]]]


@auto_docstring
class FlavaImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = FLAVA_IMAGE_MEAN
    image_std = FLAVA_IMAGE_STD
    size = {"height": 224, "width": 224}
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True

    # Mask related params
    return_image_mask = False
    input_size_patches = 14
    total_mask_patches = 75
    mask_group_min_patches = 16
    mask_group_max_patches = None
    mask_group_min_aspect_ratio = 0.3
    mask_group_max_aspect_ratio = None
    # Codebook related params
    return_codebook_pixels = False
    codebook_do_resize = True
    codebook_size = {"height": 112, "width": 112}
    # LANCZOS resample does not support torch Tensor. Use BICUBIC as closest alternative
    codebook_resample = PILImageResampling.BICUBIC
    codebook_do_center_crop = True
    codebook_crop_size = {"height": 112, "width": 112}
    codebook_do_rescale = True
    codebook_rescale_factor = 1 / 255
    codebook_do_map_pixels = True
    codebook_do_normalize = True
    codebook_image_mean = FLAVA_CODEBOOK_MEAN
    codebook_image_std = FLAVA_CODEBOOK_STD
    valid_kwargs = FlavaFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[FlavaFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[DefaultFastImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    @classmethod
    def from_dict(cls, image_processor_dict: dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure parameters are updated if image processor is
        created using from_dict and kwargs e.g. `FlavaImageProcessor.from_pretrained(checkpoint, codebook_size=600)`
        """
        image_processor_dict = image_processor_dict.copy()
        if "codebook_size" in kwargs:
            image_processor_dict["codebook_size"] = kwargs.pop("codebook_size")
        if "codebook_crop_size" in kwargs:
            image_processor_dict["codebook_crop_size"] = kwargs.pop("codebook_crop_size")
        return super().from_dict(image_processor_dict, **kwargs)

    @lru_cache
    def masking_generator(
        self,
        input_size_patches,
        total_mask_patches,
        mask_group_min_patches,
        mask_group_max_patches,
        mask_group_min_aspect_ratio,
        mask_group_max_aspect_ratio,
    ) -> FlavaMaskingGenerator:
        return FlavaMaskingGenerator(
            input_size=input_size_patches,
            total_mask_patches=total_mask_patches,
            mask_group_min_patches=mask_group_min_patches,
            mask_group_max_patches=mask_group_max_patches,
            mask_group_min_aspect_ratio=mask_group_min_aspect_ratio,
            mask_group_max_aspect_ratio=mask_group_max_aspect_ratio,
        )

    def map_pixels(self, image: "torch.Tensor") -> "torch.Tensor":
        return (1 - 2 * LOGIT_LAPLACE_EPS) * image + LOGIT_LAPLACE_EPS

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        crop_size: Optional[SizeDict] = None,
        default_to_square: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        codebook_size: Optional[SizeDict] = None,
        codebook_crop_size: Optional[SizeDict] = None,
        codebook_image_mean: Optional[Union[float, list[float]]] = None,
        codebook_image_std: Optional[Union[float, list[float]]] = None,
        codebook_resample: Optional[PILImageResampling] = None,
        data_format: Optional[ChannelDimension] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if kwargs is None:
            kwargs = {}
        if size is not None:
            size = SizeDict(**get_size_dict(size=size, default_to_square=default_to_square))
        if crop_size is not None:
            crop_size = SizeDict(**get_size_dict(crop_size, param_name="crop_size"))
        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)
        if data_format is None:
            data_format = ChannelDimension.FIRST
        if codebook_size is not None:
            codebook_size = SizeDict(**get_size_dict(size=codebook_size, default_to_square=default_to_square))
        if codebook_crop_size is not None:
            codebook_crop_size = SizeDict(**get_size_dict(codebook_crop_size, param_name="codebook_crop_size"))
        if isinstance(codebook_image_mean, list):
            codebook_image_mean = tuple(codebook_image_mean)
        if isinstance(codebook_image_std, list):
            codebook_image_std = tuple(codebook_image_std)

        kwargs["size"] = size
        kwargs["crop_size"] = crop_size
        kwargs["image_mean"] = image_mean
        kwargs["image_std"] = image_std
        kwargs["codebook_size"] = codebook_size
        kwargs["codebook_crop_size"] = codebook_crop_size
        kwargs["codebook_image_mean"] = codebook_image_mean
        kwargs["codebook_image_std"] = codebook_image_std
        kwargs["data_format"] = data_format
        kwargs["codebook_interpolation"] = (
            pil_torch_interpolation_mapping[codebook_resample]
            if isinstance(codebook_resample, (PILImageResampling, int))
            else codebook_resample
        )

        # torch resize uses interpolation instead of resample
        # Check if resample is an int before checking if it's an instance of PILImageResampling
        # because if pillow < 9.1.0, resample is an int and PILImageResampling is a module.
        # Checking PILImageResampling will fail with error `TypeError: isinstance() arg 2 must be a type or tuple of types`.
        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        return kwargs

    def _preprocess_image(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        do_map_pixels: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
    ) -> "torch.Tensor":
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if do_map_pixels:
                stacked_images = self.map_pixels(image=stacked_images)
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return processed_images

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        # Mask related params
        return_image_mask: Optional[bool],
        input_size_patches: Optional[int],
        total_mask_patches: Optional[int],
        mask_group_min_patches: Optional[int],
        mask_group_max_patches: Optional[int],
        mask_group_min_aspect_ratio: Optional[float],
        mask_group_max_aspect_ratio: Optional[float],
        # Codebook related params
        return_codebook_pixels: Optional[bool],
        codebook_do_resize: Optional[bool],
        codebook_size: Optional[SizeDict],
        codebook_interpolation: Optional["F.InterpolationMode"],
        codebook_do_center_crop: Optional[bool],
        codebook_crop_size: Optional[SizeDict],
        codebook_do_rescale: Optional[bool],
        codebook_rescale_factor: Optional[float],
        codebook_do_map_pixels: Optional[bool],
        codebook_do_normalize: Optional[bool],
        codebook_image_mean: Optional[Union[float, list[float]]],
        codebook_image_std: Optional[Union[float, list[float]]],
        disable_grouping: Optional[bool],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        processed_images = self._preprocess_image(
            images=images,
            do_resize=do_resize,
            size=size,
            interpolation=interpolation,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            do_map_pixels=False,
            image_mean=image_mean,
            image_std=image_std,
            disable_grouping=disable_grouping,
            return_tensors=return_tensors,
        )
        data = {
            "pixel_values": processed_images,
        }

        if return_codebook_pixels:
            codebook_processed_images = self._preprocess_image(
                images=images,
                do_resize=codebook_do_resize,
                size=codebook_size,
                interpolation=codebook_interpolation,
                do_center_crop=codebook_do_center_crop,
                crop_size=codebook_crop_size,
                do_rescale=codebook_do_rescale,
                rescale_factor=codebook_rescale_factor,
                do_normalize=codebook_do_normalize,
                do_map_pixels=codebook_do_map_pixels,
                image_mean=codebook_image_mean,
                image_std=codebook_image_std,
                disable_grouping=disable_grouping,
                return_tensors=return_tensors,
            )
            data["codebook_pixel_values"] = codebook_processed_images

        if return_image_mask:
            mask_generator = self.masking_generator(
                input_size_patches=input_size_patches,
                total_mask_patches=total_mask_patches,
                mask_group_min_patches=mask_group_min_patches,
                mask_group_max_patches=mask_group_max_patches,
                mask_group_min_aspect_ratio=mask_group_min_aspect_ratio,
                mask_group_max_aspect_ratio=mask_group_max_aspect_ratio,
            )
            masks = [mask_generator() for _ in range(len(images))]
            masks = torch.stack(masks, dim=0) if return_tensors else masks
            data["bool_masked_pos"] = masks

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["FlavaImageProcessorFast"]
