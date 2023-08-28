# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Beit3"""

from typing import Dict, List, Optional, Union

import PIL
import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    center_crop,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
    to_numpy_array,
    valid_images, infer_channel_dimension_format, is_scaled_image,
)
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class Beit3ImageProcessor(BaseImageProcessor):
    r"""
    Constructs an Beit3 image processor.

    This image processor inherits from [`ImageProcessingMixin`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the shorter edge of the input to a certain `size`.
        size (`Dict[str, int]`, *optional*, defaults to {"height": 768, "width": 768}):
            The size to use for resizing the image. Only has an effect if `do_resize` is set to `True`. If `size` is a
            sequence like (h, w), output size will be matched to this. If `size` is an int, then image will be resized
            to (size, size).
        resample (`int`, *optional*, defaults to `PIL.Image.Resampling.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`,
            `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`,
            `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set
            to `True`.
        do_center_crop (`bool`, *optional*, defaults to `False`):
            Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
            image is padded with 0's and then center cropped.
        crop_size (`int`, *optional*, defaults to {"height": 768, "width": 768}):
            The size to use for center cropping the image. Only has an effect if `do_center_crop` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input by a certain factor.
        rescale_factor (`float`, *optional*, defaults to `1/255`):
            The factor to use for rescaling the image. Only has an effect if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with `image_mean` and `image_std`. Desired output size when applying
            center-cropping. Only has an effect if `do_center_crop` is set to `True`.
        image_mean (`List[int]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=None,
        resample=PILImageResampling.BICUBIC,
        do_center_crop=False,
        crop_size=None,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        do_reduce_labels: bool = False,
        **kwargs,
    ):
        size = size if size is not None else {"height": 768, "width": 768}
        size = get_size_dict(size, default_to_square=True)

        crop_size = crop_size if crop_size is not None else {"height": 768, "width": 768}
        crop_size = get_size_dict(crop_size, default_to_square=True)

        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_reduce_labels = do_reduce_labels

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to a certain size.
        """
        size = get_size_dict(size, default_to_square=True)
        if "height" not in size or "width" not in size:
            raise ValueError("size dictionary must contain height and width keys")

        return resize(image, (size["height"], size["width"]), resample=resample, data_format=data_format, **kwargs)

    def center_crop(
        self,
        image: np.ndarray,
        crop_size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Center crop an image to a certain size.
        """
        crop_size = get_size_dict(crop_size, default_to_square=True)
        if "height" not in crop_size or "width" not in crop_size:
            raise ValueError("crop_size dictionary must contain height and width keys")

        return center_crop(image, (crop_size["height"], crop_size["width"]), data_format=data_format, **kwargs)

    def rescale(
        self,
        image: np.ndarray,
        rescale_factor: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Rescale an image by a certain factor.
        """
        return rescale(image, rescale_factor, data_format=data_format, **kwargs)

    def normalize(
        self,
        image: np.ndarray,
        mean: List[float],
        std: List[float],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image with a certain mean and standard deviation.
        """
        return normalize(image, mean, std, data_format=data_format, **kwargs)

    def _preprocess_segmentation_map(
        self,
        segmentation_map: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_reduce_labels: bool = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """Preprocesses a single segmentation map."""
        # All transformations expect numpy arrays.
        segmentation_map = to_numpy_array(segmentation_map)
        # Add an axis to the segmentation maps for transformations.
        if segmentation_map.ndim == 2:
            segmentation_map = segmentation_map[None, ...]
            added_dimension = True
            input_data_format = ChannelDimension.FIRST
        else:
            added_dimension = False
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)
        segmentation_map = self._preprocess(
            image=segmentation_map,
            do_reduce_labels=do_reduce_labels,
            do_resize=do_resize,
            resample=resample,
            size=size,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_normalize=False,
            do_rescale=False,
            input_data_format=ChannelDimension.FIRST,
        )
        # Remove extra axis if added
        if added_dimension:
            segmentation_map = np.squeeze(segmentation_map, axis=0)
        segmentation_map = segmentation_map.astype(np.int64)
        return segmentation_map

    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_reduce_labels: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
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
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the image after center crop. If one edge the image is smaller than `crop_size`, it will be
                padded with zeros and then cropped
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
            do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
                Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
                is used for background, and background itself is not included in all classes of a dataset (e.g.
                ADE20k). The background label will be replaced by 255.
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
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, default_to_square=True, param_name="size")
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_reduce_labels = do_reduce_labels if do_reduce_labels is not None else self.do_reduce_labels

        images = make_list_of_images(images)
        if segmentation_maps is not None:
            segmentation_maps = make_list_of_images(segmentation_maps, expected_ndims=2)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if segmentation_maps is not None and not valid_images(segmentation_maps):
            raise ValueError(
                "Invalid segmentation map type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None or resample is None:
            raise ValueError("Size and resample must be specified if do_resize is True.")

        if do_center_crop and crop_size is None:
            raise ValueError("Crop size must be specified if do_center_crop is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        images = [
            self._preprocess_image(
                image=img,
                do_resize=do_resize,
                do_center_crop=do_center_crop,
                do_rescale=do_rescale,
                do_normalize=do_normalize,
                resample=resample,
                size=size,
                rescale_factor=rescale_factor,
                crop_size=crop_size,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for img in images
        ]

        data = {"pixel_values": images}

        if segmentation_maps is not None:
            segmentation_maps = [
                self._preprocess_segmentation_map(
                    segmentation_map=segmentation_map,
                    do_reduce_labels=do_reduce_labels,
                    do_resize=do_resize,
                    resample=resample,
                    size=size,
                    do_center_crop=do_center_crop,
                    crop_size=crop_size,
                )
                for segmentation_map in segmentation_maps
            ]
            data["labels"] = segmentation_maps

        return BatchFeature(data=data, tensor_type=return_tensors)

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        # All transformations expect numpy arrays.
        image = to_numpy_array(image)
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        image = self._preprocess(
            image,
            do_reduce_labels=False,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            input_data_format=input_data_format,
        )
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

    def _preprocess(
        self,
        image: ImageInput,
        do_reduce_labels: bool = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        if do_reduce_labels:
            image = self.reduce_label(image)

        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        if do_center_crop:
            image = self.center_crop(image=image, crop_size=crop_size, input_data_format=input_data_format)

        if do_rescale:
            image = self.rescale(image=image, rescale_factor=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        return image