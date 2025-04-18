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
"""Fast Image processor class for ViTMatte."""

from functools import partial
from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    get_image_size,
    make_list_of_images,
    validate_kwargs,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    add_start_docstrings,
    filter_out_non_signature_kwargs,
    is_torch_available,
    is_vision_available,
    logging,
)


if is_vision_available():
    pass


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class VitMatteFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    do_pad: Optional[bool]
    size_divisibility: int


@add_start_docstrings(
    "Constructs a fast VitMatte image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to make the width and height divisible by `size_divisibility`. Can be overridden
            by the `do_pad` parameter in the `preprocess` method.
        size_divisibility (`int`, *optional*, defaults to 32):
            The width and height of the image will be padded to be divisible by this number.
    """,
)
class VitMatteImageProcessorFast(BaseImageProcessorFast):
    do_rescale: bool = True
    rescale_factor: Union[int, float] = 1 / 255
    do_normalize: bool = True
    image_mean: Optional[Union[float, list[float]]] = IMAGENET_STANDARD_MEAN
    image_std: Optional[Union[float, list[float]]] = IMAGENET_STANDARD_STD
    do_pad: bool = True
    size_divisibility: int = 32
    # input_data_format = ChannelDimension.FIRST
    valid_kwargs = VitMatteFastImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[VitMatteFastImageProcessorKwargs]) -> None:
        super().__init__(**kwargs)

    @add_start_docstrings(
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to make the width and height divisible by `size_divisibility`. Can be overridden
            by the `do_pad` parameter in the `preprocess` method.
        size_divisibility (`int`, *optional*, defaults to 32):
            The width and height of the image will be padded to be divisible by this number.
    """,
    )
    def preprocess(
        self,
        images: list["torch.Tensor"],
        trimaps: list["torch.Tensor"],
        **kwargs: Unpack[VitMatteFastImageProcessorKwargs],
    ) -> BatchFeature:
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_kwargs.__annotations__.keys())
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.

        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")

        # Prepare input images
        images = self._prepare_input_images(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )

        # Prepare inpout trimaps
        trimaps = self._prepare_input_trimaps(trimaps=trimaps, device=device)

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("resample")
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")
        kwargs.pop("do_resize")
        kwargs.pop("do_center_crop")
        kwargs.pop("size")
        kwargs.pop("crop_size")

        return self._preprocess(images=images, trimaps=trimaps, **kwargs)

    def _prepare_input_trimaps(
        self, trimaps: ImageInput, device: Optional["torch.device"] = None
    ) -> list["torch.Tensor"]:
        """
        Prepare input trimaps for processing,m this can not yet deal with nested list

        Args:
            trimaps ('ImageInout):
                The input trimaps to be process, should not be nested
            device('Optional['torch.device'] defaults to 'self.device'):
                The device to process the trimaps on

        Returns:
            list['torch.Tensor']:
                Input trimaps converted to a list of tensors
        """
        # from batch or single image to list, and insert channel dimension
        trimaps = make_list_of_images(trimaps, expected_ndims=2)

        # passing ChannelDimension.First achieves correct functionality on grayscale/single channel
        process_image_fn = partial(
            self._process_image,
            input_data_format=ChannelDimension.FIRST,
            device=device,
        )

        processed_trimaps = []
        for trimap in trimaps:
            processed_trimaps.append(torch.unsqueeze(process_image_fn(trimap), dim=0))

        return processed_trimaps

    def _pad_image(
        self,
        image: "torch.tensor",
        size_divisibility: int = 32,
    ) -> "torch.tensor":
        """
        Pads an image constantly so that width and height are divisible by size_divisibility

        Args:
            image (`torch,tensor`):
                Image to pad.
            size_divisibility (`int`, *optional*, defaults to 32):
                The width and height of the image will be padded to be divisible by this number.

        Returns:
            'torch.tensor':
                padded tensor
        """

        height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)

        pad_height = 0 if height % size_divisibility == 0 else size_divisibility - height % size_divisibility
        pad_width = 0 if width % size_divisibility == 0 else size_divisibility - width % size_divisibility

        if pad_width + pad_height > 0:
            padding = (0, pad_width, 0, pad_height)
            image = torch.nn.functional.pad(image, padding)

        return image

    @filter_out_non_signature_kwargs()
    def _preprocess(
        self,
        images: list["torch.Tensor"],
        trimaps: list["torch.Tensor"],
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_pad: Optional[bool] = None,
        size_divisibility: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # same for trimaps
        grouped_trimaps, grouped_trimaps_index = group_images_by_shape(trimaps)
        processed_trimaps_grouped = {}
        for shape, stacked_trimaps in grouped_trimaps.items():
            # no normalization for trimaps
            stacked_trimaps = self.rescale_and_normalize(
                stacked_trimaps, do_rescale, rescale_factor, False, image_mean, image_std
            )
            processed_trimaps_grouped[shape] = stacked_trimaps
        processed_trimaps = reorder_images(processed_trimaps_grouped, grouped_trimaps_index)

        # concatenate images and trimaps
        processed_images = [
            torch.cat([image, trimap], dim=0) for image, trimap in zip(processed_images, processed_trimaps)
        ]

        # do the padding
        if do_pad:
            processed_images = [self._pad_image(image, size_divisibility) for image in processed_images]

        # finish things up
        grouped_processed, grouped_processed_index = group_images_by_shape(processed_images)
        processed_images_grouped = {}
        for shape, stacked_processed_images in grouped_processed.items():
            processed_images_grouped[shape] = stacked_processed_images
        processed_images = reorder_images(processed_images_grouped, grouped_processed_index)
        processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)


__all__ = ["VitMatteImageProcessorFast"]
