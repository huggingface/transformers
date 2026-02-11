# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for ConvNeXT."""

import numpy as np

from ...image_processing_backends import PilBackend, TorchVisionBackend
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import get_resize_output_image_size, group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs
from ...utils import TensorType, auto_docstring, is_torchvision_available, logging


if is_torchvision_available():
    import torch
    from torchvision.transforms.v2 import functional as tvF

logger = logging.get_logger(__name__)


class ConvNextImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    crop_pct (`float`, *optional*, defaults to `self.crop_pct`):
        Percentage of the image to crop. Only has an effect if size < 384.
    """

    crop_pct: float


class ConvNextTorchVisionBackend(TorchVisionBackend):
    """TorchVision backend for ConvNeXT with custom resize."""

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        crop_pct: float = 224 / 256,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize with crop_pct support."""
        if not size.shortest_edge:
            raise ValueError(f"Size dictionary must contain 'shortest_edge' key. Got {size.keys()}")
        shortest_edge = size.shortest_edge

        if shortest_edge < 384:
            # maintain same ratio, resizing shortest edge to shortest_edge/crop_pct
            resize_shortest_edge = int(shortest_edge / crop_pct)
            resize_size = get_resize_output_image_size(
                image, size=resize_shortest_edge, default_to_square=False, input_data_format=ChannelDimension.FIRST
            )
            image = super().resize(
                image,
                SizeDict(height=resize_size[0], width=resize_size[1]),
                resample=resample,
                **kwargs,
            )
            # then crop to (shortest_edge, shortest_edge)
            return self.center_crop(
                image,
                SizeDict(height=shortest_edge, width=shortest_edge),
                **kwargs,
            )
        else:
            # warping (no cropping) when evaluated at 384 or larger
            return super().resize(
                image,
                SizeDict(height=shortest_edge, width=shortest_edge),
                resample=resample,
                **kwargs,
            )

    def preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
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
        crop_pct: float = 224 / 256,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for ConvNeXT."""
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(stacked_images, size, resample, crop_pct)
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

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


class ConvNextPilBackend(PilBackend):
    """PIL backend for ConvNeXT with custom resize."""

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        crop_pct: float = 224 / 256,
        **kwargs,
    ) -> np.ndarray:
        """Resize with crop_pct support."""
        if not size.shortest_edge:
            raise ValueError(f"Size dictionary must contain 'shortest_edge' key. Got {size.keys()}")
        shortest_edge = size.shortest_edge

        if shortest_edge < 384:
            # maintain same ratio, resizing shortest edge to shortest_edge/crop_pct
            resize_shortest_edge = int(shortest_edge / crop_pct)
            resize_size = get_resize_output_image_size(
                image, size=resize_shortest_edge, default_to_square=False, input_data_format=ChannelDimension.FIRST
            )
            image = super().resize(
                image,
                size=SizeDict(height=resize_size[0], width=resize_size[1]),
                resample=resample,
                **kwargs,
            )
            # then crop to (shortest_edge, shortest_edge)
            return super().center_crop(
                image,
                size=SizeDict(height=shortest_edge, width=shortest_edge),
                **kwargs,
            )
        else:
            # warping (no cropping) when evaluated at 384 or larger
            return super().resize(
                image,
                size=SizeDict(height=shortest_edge, width=shortest_edge),
                resample=resample,
                **kwargs,
            )

    def preprocess(
        self,
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
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
        crop_pct: float = 224 / 256,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for ConvNeXT."""
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size, resample, crop_pct)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


@auto_docstring(custom_intro="Constructs a ConvNeXT image processor.")
class ConvNextImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]
    valid_kwargs = ConvNextImageProcessorKwargs

    _backend_classes = {
        "torchvision": ConvNextTorchVisionBackend,
        "pil": ConvNextPilBackend,
    }

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 384}
    default_to_square = False
    do_resize = True
    do_rescale = True
    do_normalize = True
    crop_pct = 224 / 256


__all__ = ["ConvNextImageProcessor"]
