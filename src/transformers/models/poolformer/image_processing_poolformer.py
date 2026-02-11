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
"""Image processor class for PoolFormer."""

import numpy as np
import torch

from ...image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    PilBackend,
    TorchVisionBackend,
)
from ...image_transforms import get_resize_output_image_size
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs
from ...utils import TensorType, auto_docstring, is_torch_available, is_torchvision_available, logging


if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF

logger = logging.get_logger(__name__)


class PoolFormerImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    crop_pct (`float`, *optional*, defaults to `self.crop_pct`):
        Percentage of the image to crop. Only has an effect if `do_resize` is set to `True`.
    """

    crop_pct: float


class PoolFormerTorchVisionBackend(TorchVisionBackend):
    """TorchVision backend for PoolFormer with custom resize (crop_pct)."""

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: PILImageResampling | tvF.InterpolationMode | int | None = None,
        crop_pct: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Resize with crop_pct: scale size by 1/crop_pct when crop_pct is set."""
        if crop_pct is not None:
            if size.shortest_edge:
                scale_size = int(size.shortest_edge / crop_pct)
            elif size.height and size.width:
                if size.height == size.width:
                    scale_size = int(size.height / crop_pct)
                else:
                    scale_size = (int(size.height / crop_pct), int(size.width / crop_pct))
            else:
                raise ValueError(f"Invalid size for resize: {size}")
            new_size = get_resize_output_image_size(
                image, size=scale_size, default_to_square=False, input_data_format=ChannelDimension.FIRST
            )
            size = SizeDict(height=new_size[0], width=new_size[1])
        return super().resize(image, size=size, resample=resample, **kwargs)

    def preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | tvF.InterpolationMode | int | None,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        crop_pct: float = 0.9,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for PoolFormer (pass crop_pct to resize)."""
        from ...image_transforms import group_images_by_shape, reorder_images

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(stacked_images, size, resample, crop_pct=crop_pct)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            stacked_images = self._rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        if do_pad:
            processed_images = self.pad(processed_images, pad_size=pad_size, disable_grouping=disable_grouping)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


class PoolFormerPilBackend(PilBackend):
    """PIL backend for PoolFormer with custom resize (crop_pct)."""

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: PILImageResampling | tvF.InterpolationMode | int | None = None,
        crop_pct: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Resize with crop_pct: scale size by 1/crop_pct when crop_pct is set."""

        if crop_pct is not None:
            if size.shortest_edge:
                scale_size = int(size.shortest_edge / crop_pct)
            elif size.height and size.width:
                if size.height == size.width:
                    scale_size = int(size.height / crop_pct)
                else:
                    scale_size = (int(size.height / crop_pct), int(size.width / crop_pct))
            else:
                raise ValueError(f"Invalid size for resize: {size}")
            output_size = get_resize_output_image_size(
                image, size=scale_size, default_to_square=False, input_data_format=ChannelDimension.FIRST
            )
            size = SizeDict(height=output_size[0], width=output_size[1])
        return super().resize(
            image,
            size=size,
            resample=resample,
            **kwargs,
        )

    def preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | tvF.InterpolationMode | int | None,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        crop_pct: float = 0.9,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for PoolFormer (pass crop_pct to resize)."""
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size, resample, crop_pct=crop_pct)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        if do_pad:
            processed_images = self.pad(processed_images, pad_size=pad_size)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


@auto_docstring(custom_intro="Constructs a PoolFormer image processor.")
class PoolFormerImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]
    valid_kwargs = PoolFormerImageProcessorKwargs

    _backend_classes = {
        "torchvision": PoolFormerTorchVisionBackend,
        "pil": PoolFormerPilBackend,
    }

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    crop_pct = 0.9
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None


__all__ = ["PoolFormerImageProcessor", "PoolFormerImageProcessorKwargs"]
