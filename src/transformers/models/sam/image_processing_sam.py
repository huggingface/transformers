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
"""Image processor class for SAM."""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, normalize, pad, rescale, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available, logging, requires_backends


if is_torch_available():
    import torch.nn.functional as F


logger = logging.get_logger(__name__)


class SamImageProcessor(BaseImageProcessor):
    r"""
    Constructs a SAM image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"longest_edge": 1024}`):
            Size of the output image after resizing. Resizes the longest edge of the image to match
            `size["longest_edge"]` while maintaining the aspect ratio. Can be overridden by the `size` parameter in the
            `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Wwhether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to the specified `pad_size`. Can be overridden by the `do_pad` parameter in the
            `preprocess` method.
        pad_size (`dict`, *optional*, defaults to `{"height": 1024, "width": 1024}`):
            Size of the output image after padding. Can be overridden by the `pad_size` parameter in the `preprocess`
            method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: bool = True,
        pad_size: int = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"longest_edge": 1024}
        size = get_size_dict(max_size=size, default_to_square=False) if not isinstance(size, dict) else size

        pad_size = pad_size if pad_size is not None else {"height": 1024, "width": 1024}
        pad_size = get_size_dict(pad_size, default_to_square=True)

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.do_convert_rgb = do_convert_rgb

    def pad_image(
        self,
        image: np.ndarray,
        pad_size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Pad an image to `(pad_size["height"], pad_size["width"])` with zeros to the right and bottom.

        Args:
            image (`np.ndarray`):
                Image to pad.
            pad_size (`Dict[str, int]`):
                Size of the output image after padding.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the image. Can be either "channels_first" or "channels_last". If `None`, the
                `data_format` of the `image` will be used.
        """
        output_height, output_width = pad_size["height"], pad_size["width"]
        input_height, input_width = get_image_size(image)

        pad_width = output_width - input_width
        pad_height = output_height - input_height

        padded_image = pad(image, ((0, pad_height), (0, pad_width)), data_format=data_format, **kwargs)
        return padded_image

    def _get_preprocess_shape(self, old_shape: Tuple[int, int], longest_edge: int):
        """
        Compute the output size given input size and target long side length.
        """
        oldh, oldw = old_shape
        scale = longest_edge * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"longest_edge": int}` specifying the size of the output image. The longest
                edge of the image will be resized to the specified size, while the other edge will be resized to
                maintain the aspect ratio.
            resample:
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        Returns:
            `np.ndarray`: The resized image.
        """
        size = get_size_dict(size)
        if "longest_edge" not in size:
            raise ValueError(f"The `size` dictionary must contain the key `longest_edge`. Got {size.keys()}")
        input_size = get_image_size(image)
        output_height, output_width = self._get_preprocess_shape(input_size, size["longest_edge"])
        return resize(image, size=(output_height, output_width), resample=resample, data_format=data_format, **kwargs)

    def rescale(
        self,
        image: np.ndarray,
        scale: Union[int, float],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Rescale an image by a scale factor. image = image * scale.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            scale (`int` or `float`):
                Scale to apply to the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return rescale(image, scale=scale, data_format=data_format, **kwargs)

    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            mean (`float` or `List[float]`):
                Image mean.
            std (`float` or `List[float]`):
                Image standard deviation.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        return normalize(image, mean=mean, std=std, data_format=data_format, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional["PILImageResampling"] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        pad_size: Optional[Dict[str, int]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Controls the size of the image after `resize`. The longest edge of the image is resized to
                `size["longest_edge"]` whilst preserving the aspect ratio.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image pixel values by rescaling factor.
            rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to apply to the image pixel values.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image.
            pad_size (`Dict[str, int]`, *optional*, defaults to `self.pad_size`):
                Controls the size of the padding applied to the image. The image is padded to `pad_size["height"]` and
                `pad_size["width"]` if `do_pad` is set to `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
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
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(max_size=size, default_to_square=False) if not isinstance(size, dict) else size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad
        pad_size = pad_size if pad_size is not None else self.pad_size
        pad_size = get_size_dict(pad_size, default_to_square=True)
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and (size is None or resample is None):
            raise ValueError("Size and resample must be specified if do_resize is True.")

        if do_rescale and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        if do_pad and pad_size is None:
            raise ValueError("Pad size must be specified if do_pad is True.")

        # PIL RGBA images are converted to RGB
        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        original_sizes = [get_image_size(image) for image in images]

        if do_resize:
            images = [self.resize(image=image, size=size, resample=resample) for image in images]

        reshaped_input_sizes = [get_image_size(image) for image in images]

        if do_rescale:
            images = [self.rescale(image=image, scale=rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image=image, mean=image_mean, std=image_std) for image in images]

        if do_pad:
            images = [self.pad_image(image=image, pad_size=pad_size) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]

        data = {"pixel_values": images, "original_sizes": original_sizes, "reshaped_input_sizes": reshaped_input_sizes}
        encoded_outputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_outputs

    def post_process_masks(
        self, masks, original_sizes, reshaped_input_sizes, mask_threshold=0.0, binarize=True, pad_size=None
    ):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`torch.Tensor`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`torch.Tensor`):
                The original size of the images before resizing for input to the model, in (height, width) format.
            reshaped_input_sizes (`torch.Tensor`):
                The size of the image input to the model, in (height, width) format. Used to remove padding.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                The threshold to use for binarizing the masks.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            pad_size (`int`, *optional*, defaults to `self.pad_size`):
                The target size the images were padded to before being passed to the model. If None, the target size is
                assumed to be the processor's `pad_size`.
        Returns:
            (`torch.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width)
            is given by original_size.
        """
        requires_backends(self, ["torch"])
        pad_size = self.pad_size if pad_size is None else pad_size
        target_image_size = (pad_size["height"], pad_size["width"])

        output_masks = []
        for i, original_size in enumerate(original_sizes):
            interpolated_mask = F.interpolate(masks[i], target_image_size, mode="bilinear", align_corners=False)
            interpolated_mask = interpolated_mask[..., : reshaped_input_sizes[i][0], : reshaped_input_sizes[i][1]]
            interpolated_mask = F.interpolate(
                interpolated_mask, [*original_size.numpy()], mode="bilinear", align_corners=False
            )
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            output_masks.append(interpolated_mask)

        return output_masks
