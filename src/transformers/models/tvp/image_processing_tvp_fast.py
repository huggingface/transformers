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
"""Fast Image processor class for TVP."""

from typing import Optional, Union

from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
    make_nested_list_of_images,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
)


if is_torch_available():
    import torch

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F


class TvpFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    r"""
    do_flip_channel_order (`bool`, *optional*):
        Whether to flip the channel order of the image from RGB to BGR.
    do_pad (`bool`, *optional*):
        Whether to pad the image.
    pad_size (`Dict[str, int]` or `SizeDict`, *optional*):
        Size dictionary specifying the desired height and width for padding.
    constant_values (`float` or `List[float]`, *optional*):
        Value used to fill the padding area when `pad_mode` is `'constant'`.
    pad_mode (`str`, *optional*):
        Padding mode to use — `'constant'`, `'edge'`, `'reflect'`, or `'symmetric'`.
    """

    do_flip_channel_order: Optional[bool]
    do_pad: Optional[bool]
    pad_size: Optional[SizeDict]
    constant_values: Optional[Union[float, list[float]]]
    pad_mode: Optional[str]


@auto_docstring
class TvpImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"longest_edge": 448}
    default_to_square = False
    crop_size = {"height": 448, "width": 448}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_pad = True
    pad_size = {"height": 448, "width": 448}
    constant_values = 0
    pad_mode = "constant"
    do_normalize = True
    do_flip_channel_order = True
    valid_kwargs = TvpFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[TvpFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        videos: Union[ImageInput, list[ImageInput], list[list[ImageInput]]],
        **kwargs: Unpack[TvpFastImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess videos using the fast image processor.
        """
        return super().preprocess(videos, **kwargs)

    def _further_process_kwargs(
        self,
        pad_size: Optional[SizeDict] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if pad_size is not None:
            pad_size = SizeDict(**get_size_dict(pad_size, param_name="pad_size"))
        kwargs["pad_size"] = pad_size

        return super()._further_process_kwargs(**kwargs)

    def _prepare_images_structure(
        self,
        images: ImageInput,
        **kwargs,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        return make_nested_list_of_images(images, **kwargs)

    def _pad_frames(
        self,
        frames: "torch.Tensor",
        pad_size: Union[SizeDict, dict],
        constant_values: Union[float, list[float]],
        pad_mode: str,
    ) -> "torch.Tensor":
        """Pad frames to the specified size."""
        height, width = pad_size.height, pad_size.width

        if frames.shape[-2:] == (height, width):
            return frames

        # Calculate padding
        current_height, current_width = frames.shape[-2:]
        pad_bottom = height - current_height
        pad_right = width - current_width

        if pad_bottom < 0 or pad_right < 0:
            raise ValueError("The padding size must be greater than frame size")

        # Apply padding
        padding = [0, 0, pad_right, pad_bottom]  # [left, top, right, bottom]
        return F.pad(frames, padding, fill=constant_values, padding_mode=pad_mode)

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to the specified size.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict` or `dict`):
                Size dictionary. If `size` has `longest_edge`, resize the longest edge to that value
                while maintaining aspect ratio. Otherwise, use the base class resize method.
            interpolation (`F.InterpolationMode`, *optional*):
                Interpolation method to use.
            antialias (`bool`, *optional*, defaults to `True`):
                Whether to use antialiasing.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR

        # Handle longest_edge case (TVP-specific)
        if size.longest_edge:
            # Get current dimensions
            current_height, current_width = image.shape[-2:]

            # Calculate new dimensions maintaining aspect ratio
            if current_height >= current_width:
                ratio = current_width * 1.0 / current_height
                new_height = size.longest_edge
                new_width = int(new_height * ratio)
            else:
                ratio = current_height * 1.0 / current_width
                new_width = size.longest_edge
                new_height = int(new_width * ratio)

            return super().resize(
                image, SizeDict(height=new_height, width=new_width), interpolation=interpolation, antialias=antialias
            )

        # Use base class resize method for other cases
        return super().resize(image, size, interpolation, antialias, **kwargs)

    def _flip_channel_order(self, frames: "torch.Tensor") -> "torch.Tensor":
        """
        Flip channel order from RGB to BGR.

        The slow processor puts the red channel at the end (BGR format),
        but the channel order is different. We need to match the exact
        channel order of the slow processor:

        Slow processor:
        - Channel 0: Blue (originally Red)
        - Channel 1: Green
        - Channel 2: Red (originally Blue)
        """
        # Assuming frames are in channels_first format (..., C, H, W)
        frames = frames.flip(-3)

        return frames

    def _preprocess(
        self,
        images: list[list["torch.Tensor"]],
        do_resize: bool,
        size: Union[SizeDict, dict],
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: Union[SizeDict, dict],
        do_rescale: bool,
        rescale_factor: float,
        do_pad: bool,
        pad_size: Union[SizeDict, dict],
        constant_values: Union[float, list[float]],
        pad_mode: str,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        do_flip_channel_order: bool,
        return_tensors: Optional[Union[str, TensorType]],
        disable_grouping: Optional[bool],
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess videos using the fast image processor.

        This method processes each video frame through the same pipeline as the original
        TVP image processor but uses torchvision operations for better performance.
        """
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping, is_nested=True
        )
        processed_images_grouped = {}
        for shape, stacked_frames in grouped_images.items():
            # Resize if needed
            if do_resize:
                stacked_frames = self.resize(stacked_frames, size, interpolation)

            # Center crop if needed
            if do_center_crop:
                stacked_frames = self.center_crop(stacked_frames, crop_size)

            # Rescale and normalize using fused method for consistency
            stacked_frames = self.rescale_and_normalize(
                stacked_frames, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )

            # Pad if needed
            if do_pad:
                stacked_frames = self._pad_frames(stacked_frames, pad_size, constant_values, pad_mode)

            # Flip channel order if needed (RGB to BGR)
            if do_flip_channel_order:
                stacked_frames = self._flip_channel_order(stacked_frames)

            processed_images_grouped[shape] = stacked_frames

        processed_images = reorder_images(processed_images_grouped, grouped_images_index, is_nested=True)
        if return_tensors == "pt":
            processed_images = [torch.stack(images, dim=0) for images in processed_images]
            processed_images = torch.stack(processed_images, dim=0)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["TvpImageProcessorFast"]
