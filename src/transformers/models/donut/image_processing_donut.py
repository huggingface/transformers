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
"""Image processor class for Donut."""

import numpy as np

from ...image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    PilBackend,
    TorchVisionBackend,
)
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torchvision_available, logging


if is_torchvision_available():
    import torch
    from torchvision.transforms.v2 import functional as tvF

logger = logging.get_logger(__name__)


class DonutImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    do_thumbnail (`bool`, *optional*, defaults to `self.do_thumbnail`):
        Whether to resize the image using thumbnail method.
    do_align_long_axis (`bool`, *optional*, defaults to `self.do_align_long_axis`):
        Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
    """

    do_thumbnail: bool
    do_align_long_axis: bool


class DonutTorchVisionBackend(TorchVisionBackend):
    """TorchVision backend for Donut with align_long_axis, thumbnail, and pad_image."""

    def align_long_axis(
        self,
        image: "torch.Tensor",
        size: SizeDict,
    ) -> "torch.Tensor":
        """Align the long axis of the image to the longest axis of the specified size."""
        input_height, input_width = image.shape[-2:]
        output_height, output_width = size.height, size.width

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            height_dim, width_dim = image.dim() - 2, image.dim() - 1
            image = torch.rot90(image, 3, dims=[height_dim, width_dim])

        return image

    def pad_image(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        random_padding: bool = False,
    ) -> "torch.Tensor":
        """Pad the image to the specified size."""
        output_height, output_width = size.height, size.width
        input_height, input_width = image.shape[-2:]

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        if random_padding:
            pad_top = torch.randint(0, delta_height + 1, ()).item()
            pad_left = torch.randint(0, delta_width + 1, ()).item()
        else:
            pad_top = delta_height // 2
            pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return tvF.pad(image, padding)

    def thumbnail(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize the image to make a thumbnail."""
        input_height, input_width = image.shape[-2:]
        output_height, output_width = size.height, size.width

        height = min(input_height, output_height)
        width = min(input_width, output_width)

        if height == input_height and width == input_width:
            return image

        if input_height > input_width:
            width = int(input_width * height / input_height)
        elif input_width > input_height:
            height = int(input_height * width / input_width)

        return super().resize(
            image,
            size=SizeDict(width=width, height=height),
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
        do_thumbnail: bool = True,
        do_align_long_axis: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for Donut."""
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_align_long_axis:
                stacked_images = self.align_long_axis(stacked_images, size)
            if do_resize:
                shortest_edge = min(size.height, size.width)
                stacked_images = self.resize(stacked_images, SizeDict(shortest_edge=shortest_edge), resample)
            if do_thumbnail:
                stacked_images = self.thumbnail(stacked_images, size, resample)
            if do_pad:
                stacked_images = self.pad_image(stacked_images, size, random_padding=False)
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


class DonutPilBackend(PilBackend):
    """PIL backend for Donut with align_long_axis, thumbnail, and pad_image."""

    def align_long_axis(
        self,
        image: np.ndarray,
        size: SizeDict,
    ) -> np.ndarray:
        """Align the long axis of the image to the longest axis of the specified size."""
        input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        output_height, output_width = size.height, size.width

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = np.rot90(image, 3, axes=(1, 2))

        return image

    def pad_image(
        self,
        image: np.ndarray,
        size: SizeDict,
        random_padding: bool = False,
    ) -> np.ndarray:
        """Pad the image to the specified size."""
        from ...image_transforms import PaddingMode
        from ...image_transforms import pad as np_pad

        output_height, output_width = size.height, size.width
        input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        if random_padding:
            pad_top = int(np.random.randint(low=0, high=delta_height + 1))
            pad_left = int(np.random.randint(low=0, high=delta_width + 1))
        else:
            pad_top = delta_height // 2
            pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        # pad() expects (height_pad, width_pad) and adds channel dimension
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        return np_pad(
            image,
            padding,
            mode=PaddingMode.CONSTANT,
            constant_values=0,
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

    def thumbnail(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        **kwargs,
    ) -> np.ndarray:
        """Resize the image to make a thumbnail."""
        input_height, input_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        output_height, output_width = size.height, size.width

        height = min(input_height, output_height)
        width = min(input_width, output_width)

        if height == input_height and width == input_width:
            return image

        if input_height > input_width:
            width = int(input_width * height / input_height)
        elif input_width > input_height:
            height = int(input_height * width / input_width)

        return self.resize(
            image,
            size=SizeDict(width=width, height=height),
            resample=resample or PILImageResampling.BICUBIC,
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
        do_thumbnail: bool = True,
        do_align_long_axis: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for Donut."""
        processed_images = []
        for image in images:
            if do_align_long_axis:
                image = self.align_long_axis(image, size)
            if do_resize:
                shortest_edge = min(size.height, size.width)
                image = self.resize(image, SizeDict(shortest_edge=shortest_edge), resample)
            if do_thumbnail:
                image = self.thumbnail(image, size, resample)
            if do_pad:
                image = self.pad_image(image, size, random_padding=False)
            if do_center_crop:
                image = self.center_crop(image, crop_size)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


@auto_docstring(custom_intro="Constructs a Donut image processor.")
class DonutImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]
    valid_kwargs = DonutImageProcessorKwargs

    _backend_classes = {
        "torchvision": DonutTorchVisionBackend,
        "pil": DonutPilBackend,
    }

    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 2560, "width": 1920}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_thumbnail = True
    do_align_long_axis = False
    do_pad = True

    def __init__(self, **kwargs: Unpack[DonutImageProcessorKwargs]):
        size = kwargs.pop("size", None)
        if isinstance(size, (tuple, list)):
            size = size[::-1]
        if size is not None:
            kwargs["size"] = size
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[DonutImageProcessorKwargs],
    ) -> BatchFeature:
        kwargs = dict(kwargs)
        if "size" in kwargs:
            size = kwargs["size"]
            if isinstance(size, (tuple, list)):
                kwargs["size"] = size[::-1]
        return super().preprocess(images, **kwargs)


__all__ = ["DonutImageProcessor"]
