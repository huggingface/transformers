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
"""Fast Image processor class for Video-LLaVA."""

from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
    VideoInput,
    make_batched_videos,
    make_flat_list_of_images,
    valid_images,
    validate_kwargs,
)
from ...processing_utils import Unpack
from ...utils import (
    add_start_docstrings,
    is_torch_available,
    is_vision_available,
    logging,
)


if is_torch_available():
    import torch

if is_vision_available():
    from ...image_utils import pil_torch_interpolation_mapping


logger = logging.get_logger(__name__)

VIDEOLLAVA_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = r"""
    Preprocess an image or batch of images.

    Args:
        images (`ImageInput`):
            Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
            passing in images with pixel values between 0 and 1, set `do_rescale=False`.
        videos (`VideoInput`):
            Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
            passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*, defaults to `self.size`):
            Describes the maximum input dimensions to the model.
        resample (`PILImageResampling` or `InterpolationMode`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
            has an effect if `do_resize` is set to `True`.
        do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
            Whether to center crop the image.
        crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
            Size of the output image after applying `center_crop`.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the image.
        rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
            Rescale factor to rescale the image by if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
            `True`.
        do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
            Whether to convert the image to RGB.
        return_tensors (`str` or `TensorType`, *optional*, defaults to `self.return_tensors`):
            Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `self.data_format`):
            Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.
        input_data_format (`ChannelDimension` or `str`, *optional*, defaults to `self.input_data_format`):
            The channel dimension format for the input image. If unset, the channel dimension format is inferred
            from the input image. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        device (`torch.device`, *optional*, defaults to `self.device`):
            The device to process the images on. If unset, the device is inferred from the input images."""


@add_start_docstrings(
    "Constructs a fast VideoLlava image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class VideoLlavaImageProcessorFast(BaseImageProcessorFast):
    do_resize = True
    size = {"shortest_edge": 224}
    resample = PILImageResampling.BICUBIC
    do_center_crop = True
    crop_size = {"height": 224, "width": 224}
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    default_to_square = False
    do_convert_rgb = True

    def __init__(self, **kwargs: Unpack[DefaultFastImageProcessorKwargs]):
        super().__init__(**kwargs)

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
        **kwargs,
    ) -> BatchFeature:
        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, interpolation=interpolation)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        return processed_images

    @add_start_docstrings(VIDEOLLAVA_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS)
    def preprocess(
        self,
        images: ImageInput = None,
        videos: VideoInput = None,
        **kwargs: Unpack[DefaultFastImageProcessorKwargs],
    ) -> BatchFeature:
        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_kwargs.__annotations__.keys())
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")
        return_tensors = kwargs.pop("return_tensors")

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        if images is not None:
            images = make_flat_list_of_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        data = {}
        if videos is not None:
            pixel_values_videos = []
            for image in videos:
                image = self._prepare_input_images(
                    images=image, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
                )
                pixel_value_image = self._preprocess(image, **kwargs)
                pixel_values_videos.append(torch.stack(pixel_value_image))
            pixel_values = torch.stack(pixel_values_videos, dim=0)
            data["pixel_values_videos"] = pixel_values

        if images is not None:
            pixel_values_images = []
            for image in images:
                image = self._prepare_input_images(
                    images=image, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
                )
                pixel_value_image = self._preprocess(image, **kwargs)
                pixel_values_images.extend(pixel_value_image)
            pixel_values = torch.stack(pixel_values_images, dim=0)
            data["pixel_values_images"] = pixel_values

        encoded_outputs = BatchFeature(data, tensor_type=return_tensors)

        return encoded_outputs


__all__ = ["VideoLlavaImageProcessorFast"]
