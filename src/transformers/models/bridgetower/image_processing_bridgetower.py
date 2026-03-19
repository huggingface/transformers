# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for BridgeTower."""

from typing import Union

import numpy as np

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torch_available, is_torchvision_available


if is_torch_available():
    import torch

if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


def get_resize_output_image_size(
    input_image: Union[np.ndarray, "torch.Tensor"],
    shorter: int = 800,
    longer: int = 1333,
    size_divisor: int = 32,
) -> tuple[int, int]:
    """Get output image size after resizing with size_divisor."""
    if is_torch_available() and isinstance(input_image, torch.Tensor):
        input_height, input_width = input_image.shape[-2:]
    else:
        input_height, input_width = get_image_size(input_image, channel_dim=ChannelDimension.FIRST)

    min_size, max_size = shorter, longer
    scale = min_size / min(input_height, input_width)

    if input_height < input_width:
        new_height = min_size
        new_width = scale * input_width
    else:
        new_height = scale * input_height
        new_width = min_size

    if max(new_height, new_width) > max_size:
        scale = max_size / max(new_height, new_width)
        new_height = scale * new_height
        new_width = scale * new_width

    new_height, new_width = int(new_height + 0.5), int(new_width + 0.5)
    new_height = new_height // size_divisor * size_divisor
    new_width = new_width // size_divisor * size_divisor

    return new_height, new_width


class BridgeTowerImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    size_divisor (`int`, *optional*, defaults to `self.size_divisor`):
        The size by which to make sure both the height and width can be divided.
    """

    size_divisor: int


@auto_docstring
class BridgeTowerImageProcessor(TorchvisionBackend):
    """Torchvision backend for BridgeTower with custom resize and center_crop."""

    valid_kwargs = BridgeTowerImageProcessorKwargs
    model_input_names = ["pixel_values", "pixel_mask"]

    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 288}
    default_to_square = False
    crop_size = {"shortest_edge": 288}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_pad = True
    size_divisor = 32

    def __init__(self, **kwargs: Unpack[BridgeTowerImageProcessorKwargs]):
        super().__init__(**kwargs)

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        size_divisor: int = 32,
        **kwargs,
    ) -> "torch.Tensor":
        """Resize with size_divisor support."""
        if not size.shortest_edge:
            raise ValueError(f"The `size` dictionary must contain the key `shortest_edge`. Got {size.keys()}")
        shorter = size.shortest_edge
        longer = int(1333 / 800 * shorter)
        output_height, output_width = get_resize_output_image_size(
            image, shorter=shorter, longer=longer, size_divisor=size_divisor
        )
        return super().resize(
            image=image,
            size=SizeDict(height=output_height, width=output_width),
            resample=resample,
            **kwargs,
        )

    def _preprocess(
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
        size_divisor: int = 32,
        **kwargs,
    ) -> BatchFeature:
        """Custom preprocessing for BridgeTower."""
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(stacked_images, size=size, resample=resample, size_divisor=size_divisor)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(
                    stacked_images, size=SizeDict(height=crop_size.shortest_edge, width=crop_size.shortest_edge)
                )
            # Use fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        data = {}
        if do_pad:
            processed_images, processed_masks = self.pad(
                processed_images, return_mask=True, disable_grouping=disable_grouping
            )
            data["pixel_mask"] = processed_masks

        data["pixel_values"] = processed_images

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["BridgeTowerImageProcessor"]
