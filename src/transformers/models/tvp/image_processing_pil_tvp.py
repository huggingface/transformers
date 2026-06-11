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
"""Image processor class for TVP."""

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import PaddingMode, flip_channel_order
from ...image_transforms import pad as np_pad
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    make_nested_list_of_images,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torchvision_available


if is_torchvision_available():
    pass


def get_resize_output_image_size(
    input_image: np.ndarray,
    max_size: int = 448,
    input_data_format: str | ChannelDimension | None = None,
) -> tuple[int, int]:
    height, width = get_image_size(input_image, input_data_format)
    if height >= width:
        ratio = width * 1.0 / height
        new_height = max_size
        new_width = new_height * ratio
    else:
        ratio = height * 1.0 / width
        new_width = max_size
        new_height = new_width * ratio
    size = (int(new_height), int(new_width))

    return size


# Adapted from transformers.models.tvp.image_processing_tvp.TvpImageProcessorKwargs
class TvpImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    do_flip_channel_order (`bool`, *optional*, defaults to `self.do_flip_channel_order`):
        Whether to flip the channel order of the image from RGB to BGR.
    constant_values (`float` or `List[float]`, *optional*, defaults to `self.constant_values`):
        Value used to fill the padding area when `pad_mode` is `'constant'`.
    pad_mode (`str`, *optional*, defaults to `self.pad_mode`):
        Padding mode to use — `'constant'`, `'edge'`, `'reflect'`, or `'symmetric'`.
    """

    do_flip_channel_order: bool
    constant_values: float | list[float] | None
    pad_mode: str | None


@auto_docstring
class TvpImageProcessorPil(PilBackend):
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
    valid_kwargs = TvpImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[TvpImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        videos: ImageInput | list[ImageInput] | list[list[ImageInput]],
        **kwargs: Unpack[TvpImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        videos (`ImageInput` or `list[ImageInput]` or `list[list[ImageInput]]`):
            Frames to preprocess.
        """
        return super().preprocess(videos, **kwargs)

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
        images = self.fetch_images(images)
        return make_nested_list_of_images(images, **kwargs)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
    ) -> np.ndarray:
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`SizeDict`):
                Size of the output image. If `size` has `longest_edge`, resize the longest edge to that value
                while maintaining aspect ratio. Otherwise, use the base class resize method.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.
        """
        # Handle longest_edge case (TVP-specific)
        if size.longest_edge:
            output_size = get_resize_output_image_size(
                image, size.longest_edge, input_data_format=ChannelDimension.FIRST
            )
            return super().resize(image, SizeDict(height=output_size[0], width=output_size[1]), resample)

        # Use base class resize method for other cases
        return super().resize(image, size, resample)

    def pad_image(
        self,
        image: np.ndarray,
        pad_size: SizeDict | None = None,
        constant_values: float | list[float] = 0,
        pad_mode: str | PaddingMode = PaddingMode.CONSTANT,
    ):
        """
        Pad an image with zeros to the given size.

        Args:
            image (`np.ndarray`):
                Image to pad.
            pad_size (`SizeDict`):
                Size of the output image with pad.
            constant_values (`Union[float, Iterable[float]]`):
                The fill value to use when padding the image.
            pad_mode (`PaddingMode` or `str`):
                The pad mode, default to PaddingMode.CONSTANT
        """
        height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        max_height = pad_size.height
        max_width = pad_size.width

        pad_right, pad_bottom = max_width - width, max_height - height
        if pad_right < 0 or pad_bottom < 0:
            raise ValueError("The padding size must be greater than image size")

        padding = ((0, pad_bottom), (0, pad_right))
        padded_image = np_pad(
            image,
            padding=padding,
            mode=pad_mode if isinstance(pad_mode, PaddingMode) else PaddingMode(pad_mode),
            constant_values=constant_values,
            data_format=ChannelDimension.FIRST,
            input_data_format=ChannelDimension.FIRST,
        )

        return padded_image

    def _flip_channel_order(self, image: np.ndarray) -> np.ndarray:
        """
        Flip channel order from RGB to BGR.
        """
        return flip_channel_order(image=image, input_data_format=ChannelDimension.FIRST)

    def _preprocess(
        self,
        images: list[list[np.ndarray]],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | None,
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_pad: bool,
        pad_size: SizeDict,
        constant_values: float | list[float],
        pad_mode: str,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_flip_channel_order: bool,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess videos using PIL backend.

        This method processes each video frame through the same pipeline as the original
        TVP image processor but uses PIL/NumPy operations.
        """
        processed_videos = []
        for video in images:
            processed_frames = []
            for frame in video:
                if do_resize:
                    frame = self.resize(frame, size, resample)
                if do_center_crop:
                    frame = self.center_crop(frame, crop_size)
                if do_rescale:
                    frame = self.rescale(frame, rescale_factor)
                if do_normalize:
                    frame = self.normalize(frame, image_mean, image_std)
                if do_pad:
                    pad_mode_enum = pad_mode if isinstance(pad_mode, PaddingMode) else PaddingMode(pad_mode)
                    frame = self.pad_image(frame, pad_size, constant_values, pad_mode_enum)
                if do_flip_channel_order:
                    frame = self._flip_channel_order(frame)
                processed_frames.append(frame)
            processed_videos.append(processed_frames)

        if return_tensors == "pt":
            from ...utils import is_torch_available

            if not is_torch_available():
                raise ImportError("PyTorch is required to return tensors")
            import torch

            processed_videos = [
                torch.stack([torch.from_numpy(frame.copy()) for frame in video], dim=0) for video in processed_videos
            ]
            processed_videos = torch.stack(processed_videos, dim=0)

        return BatchFeature(data={"pixel_values": processed_videos}, tensor_type=return_tensors)


__all__ = ["TvpImageProcessorPil"]
