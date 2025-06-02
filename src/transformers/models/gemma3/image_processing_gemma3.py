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
"""Image processor class for Gemma3."""

import itertools
import math
from typing import Dict, List, Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from ...utils import TensorType, filter_out_non_signature_kwargs, is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    import PIL


class Gemma3ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a SigLIP image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after resizing. Can be overridden by `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image by the specified mean and standard deviation. Can be overridden by
            `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        do_pan_and_scan (`bool`, *optional*):
            Whether to apply `pan_and_scan` to images.
        pan_and_scan_min_crop_size (`int`, *optional*):
            Minimum size of each crop in pan and scan.
        pan_and_scan_max_num_crops (`int`, *optional*):
            Maximum number of crops per image in pan and scan.
        pan_and_scan_min_ratio_to_activate (`float`, *optional*):
            Minimum aspect ratio to activate pan and scan.
    """

    model_input_names = ["pixel_values", "num_crops"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
        do_pan_and_scan: Optional[bool] = None,
        pan_and_scan_min_crop_size: Optional[int] = None,
        pan_and_scan_max_num_crops: Optional[int] = None,
        pan_and_scan_min_ratio_to_activate: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size, default_to_square=True)
        image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_convert_rgb = do_convert_rgb
        self.do_pan_and_scan = do_pan_and_scan
        self.pan_and_scan_min_crop_size = pan_and_scan_min_crop_size
        self.pan_and_scan_max_num_crops = pan_and_scan_max_num_crops
        self.pan_and_scan_min_ratio_to_activate = pan_and_scan_min_ratio_to_activate

    def pan_and_scan(
        self,
        image: np.ndarray,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pan and Scan and image, by cropping into smaller images when the aspect ratio exceeds
        minimum allowed ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            pan_and_scan_min_crop_size (`int`, *optional*):
                Minimum size of each crop in pan and scan.
            pan_and_scan_max_num_crops (`int`, *optional*):
                Maximum number of crops per image in pan and scan.
            pan_and_scan_min_ratio_to_activate (`float`, *optional*):
                Minimum aspect ratio to activate pan and scan.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        height, width = get_image_size(image)

        # Square or landscape image.
        if width >= height:
            # Only apply PaS if the image is sufficiently exaggerated
            if width / height < pan_and_scan_min_ratio_to_activate:
                return []

            # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
            num_crops_w = int(math.floor(width / height + 0.5))  # Half round up rounding.
            num_crops_w = min(int(math.floor(width / pan_and_scan_min_crop_size)), num_crops_w)

            # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
            num_crops_w = max(2, num_crops_w)
            num_crops_w = min(pan_and_scan_max_num_crops, num_crops_w)
            num_crops_h = 1

        # Portrait image.
        else:
            # Only apply PaS if the image is sufficiently exaggerated
            if height / width < pan_and_scan_min_ratio_to_activate:
                return []

            # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
            num_crops_h = int(math.floor(height / width + 0.5))
            num_crops_h = min(int(math.floor(height / pan_and_scan_min_crop_size)), num_crops_h)

            # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
            num_crops_h = max(2, num_crops_h)
            num_crops_h = min(pan_and_scan_max_num_crops, num_crops_h)
            num_crops_w = 1

        crop_size_w = int(math.ceil(width / num_crops_w))
        crop_size_h = int(math.ceil(height / num_crops_h))

        # Don't apply PaS if crop size is too small.
        if min(crop_size_w, crop_size_h) < pan_and_scan_min_crop_size:
            return []

        crop_positions_w = [crop_size_w * i for i in range(num_crops_w)]
        crop_positions_h = [crop_size_h * i for i in range(num_crops_h)]

        if input_data_format == ChannelDimension.LAST:
            image_crops = [
                image[pos_h : pos_h + crop_size_h, pos_w : pos_w + crop_size_w]
                for pos_h, pos_w in itertools.product(crop_positions_h, crop_positions_w)
            ]
        else:
            image_crops = [
                image[:, pos_h : pos_h + crop_size_h, pos_w : pos_w + crop_size_w]
                for pos_h, pos_w in itertools.product(crop_positions_h, crop_positions_w)
            ]

        return image_crops

    def _process_images_for_pan_and_scan(
        self,
        images: List[np.ndarray],
        do_pan_and_scan: bool,
        pan_and_scan_min_crop_size: int,
        pan_and_scan_max_num_crops: int,
        pan_and_scan_min_ratio_to_activate: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        pas_images_list = []
        num_crops = []
        for image in images:
            pas_images = self.pan_and_scan(
                image=image,
                pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
                pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
                pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            pas_images_list.extend([image] + pas_images)
            num_crops.append(len(pas_images))
        return pas_images_list, num_crops

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        do_convert_rgb: Optional[bool] = None,
        do_pan_and_scan: Optional[bool] = None,
        pan_and_scan_min_crop_size: Optional[int] = None,
        pan_and_scan_max_num_crops: Optional[int] = None,
        pan_and_scan_min_ratio_to_activate: Optional[float] = None,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
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
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            do_pan_and_scan (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to apply `pan_and_scan` to images.
            pan_and_scan_min_crop_size (`int`, *optional*, defaults to `self.pan_and_scan_min_crop_size`):
                Minimum size of each crop in pan and scan.
            pan_and_scan_max_num_crops (`int`, *optional*, defaults to `self.pan_and_scan_max_num_crops`):
                Maximum number of crops per image in pan and scan.
            pan_and_scan_min_ratio_to_activate (`float`, *optional*, defaults to `self.pan_and_scan_min_ratio_to_activate`):
                Minimum aspect ratio to activate pan and scan.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_pan_and_scan = do_pan_and_scan if do_pan_and_scan is not None else self.do_pan_and_scan
        pan_and_scan_min_crop_size = (
            pan_and_scan_min_crop_size if pan_and_scan_min_crop_size is not None else self.pan_and_scan_min_crop_size
        )
        pan_and_scan_max_num_crops = (
            pan_and_scan_max_num_crops if pan_and_scan_max_num_crops is not None else self.pan_and_scan_max_num_crops
        )
        pan_and_scan_min_ratio_to_activate = (
            pan_and_scan_min_ratio_to_activate
            if pan_and_scan_min_ratio_to_activate is not None
            else self.pan_and_scan_min_ratio_to_activate
        )

        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_pan_and_scan:
            images, num_crops = self._process_images_for_pan_and_scan(
                images=images,
                do_pan_and_scan=do_pan_and_scan,
                pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
                pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
                pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
                data_format=data_format,
                input_data_format=input_data_format,
            )

        else:
            num_crops = [0 for _ in images]

        processed_images = []
        for image in images:
            if do_resize:
                height, width = size["height"], size["width"]
                image = resize(
                    image=image, size=(height, width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        data = {"pixel_values": processed_images, "num_crops": num_crops}
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Gemma3ImageProcessor"]
