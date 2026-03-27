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
"""Image processor class for Flava."""

from collections.abc import Iterable
from functools import lru_cache
from typing import Any

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
)
from ...utils.import_utils import requires
from .image_processing_flava import (
    FLAVA_CODEBOOK_MEAN,
    FLAVA_CODEBOOK_STD,
    FLAVA_IMAGE_MEAN,
    FLAVA_IMAGE_STD,
    LOGIT_LAPLACE_EPS,
    FlavaImageProcessorKwargs,
    FlavaMaskingGenerator,
)


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class FlavaImageProcessorPil(PilBackend):
    valid_kwargs = FlavaImageProcessorKwargs
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
    codebook_resample = PILImageResampling.LANCZOS
    codebook_do_center_crop = True
    codebook_crop_size = {"height": 112, "width": 112}
    codebook_do_rescale = True
    codebook_rescale_factor = 1 / 255
    codebook_do_map_pixels = True
    codebook_do_normalize = True
    codebook_image_mean = FLAVA_CODEBOOK_MEAN
    codebook_image_std = FLAVA_CODEBOOK_STD

    def __init__(self, **kwargs: Unpack[FlavaImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[FlavaImageProcessorKwargs]) -> BatchFeature:
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
        # Import torch here since FlavaMaskingGenerator uses torch
        return FlavaMaskingGenerator(
            input_size=input_size_patches,
            total_mask_patches=total_mask_patches,
            mask_group_min_patches=mask_group_min_patches,
            mask_group_max_patches=mask_group_max_patches,
            mask_group_min_aspect_ratio=mask_group_min_aspect_ratio,
            mask_group_max_aspect_ratio=mask_group_max_aspect_ratio,
        )

    def map_pixels(self, image: np.ndarray) -> np.ndarray:
        return (1 - 2 * LOGIT_LAPLACE_EPS) * image + LOGIT_LAPLACE_EPS

    def _standardize_kwargs(
        self,
        size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        crop_size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        default_to_square: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        codebook_size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        codebook_crop_size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        codebook_image_mean: float | list[float] | None = None,
        codebook_image_std: float | list[float] | None = None,
        codebook_resample: "PILImageResampling | int | None" = None,
        data_format: ChannelDimension | None = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        kwargs = super()._standardize_kwargs(
            size=size,
            crop_size=crop_size,
            default_to_square=default_to_square,
            image_mean=image_mean,
            image_std=image_std,
            data_format=data_format,
            **kwargs,
        )
        if codebook_size is not None and not isinstance(codebook_size, SizeDict):
            codebook_size = SizeDict(**get_size_dict(size=codebook_size, default_to_square=default_to_square))
        if codebook_crop_size is not None and not isinstance(codebook_crop_size, SizeDict):
            codebook_crop_size = SizeDict(**get_size_dict(codebook_crop_size, param_name="codebook_crop_size"))
        if isinstance(codebook_image_mean, list):
            codebook_image_mean = tuple(codebook_image_mean)
        if isinstance(codebook_image_std, list):
            codebook_image_std = tuple(codebook_image_std)

        kwargs["codebook_size"] = codebook_size
        kwargs["codebook_crop_size"] = codebook_crop_size
        kwargs["codebook_image_mean"] = codebook_image_mean
        kwargs["codebook_image_std"] = codebook_image_std
        # Store codebook_resample as-is - resize() will handle conversion
        kwargs["codebook_resample"] = codebook_resample

        return kwargs

    def _preprocess_image(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        do_map_pixels: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
    ) -> list[np.ndarray]:
        # Process images one by one (no batching in PIL backend)
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image=image, size=size, resample=resample)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            if do_map_pixels:
                image = self.map_pixels(image)
            processed_images.append(image)

        return processed_images

    def _preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        # Mask related params
        return_image_mask: bool | None,
        input_size_patches: int | None,
        total_mask_patches: int | None,
        mask_group_min_patches: int | None,
        mask_group_max_patches: int | None,
        mask_group_min_aspect_ratio: float | None,
        mask_group_max_aspect_ratio: float | None,
        # Codebook related params
        return_codebook_pixels: bool | None,
        codebook_do_resize: bool | None,
        codebook_size: SizeDict | None,
        codebook_resample: "PILImageResampling | int | None",
        codebook_do_center_crop: bool | None,
        codebook_crop_size: SizeDict | None,
        codebook_do_rescale: bool | None,
        codebook_rescale_factor: float | None,
        codebook_do_map_pixels: bool | None,
        codebook_do_normalize: bool | None,
        codebook_image_mean: float | list[float] | None,
        codebook_image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        processed_images = self._preprocess_image(
            images=images,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            do_map_pixels=False,
            image_mean=image_mean,
            image_std=image_std,
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
                resample=codebook_resample,
                do_center_crop=codebook_do_center_crop,
                crop_size=codebook_crop_size,
                do_rescale=codebook_do_rescale,
                rescale_factor=codebook_rescale_factor,
                do_normalize=codebook_do_normalize,
                do_map_pixels=codebook_do_map_pixels,
                image_mean=codebook_image_mean,
                image_std=codebook_image_std,
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
            data["bool_masked_pos"] = masks

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["FlavaImageProcessorPil"]
