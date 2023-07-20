# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Idefics."""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

import torchvision.transforms as transforms

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import PaddingMode, pad
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    get_image_size,
    infer_channel_dimension_format,
    make_list_of_images,
    valid_images,
)
from ...utils import TensorType, is_vision_available, logging


if is_vision_available():
    import PIL
    from PIL import Image

logger = logging.get_logger(__name__)


def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


def get_max_height_width(images: List[np.ndarray]) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    input_channel_dimension = infer_channel_dimension_format(images[0])

    if input_channel_dimension == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_channel_dimension == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_channel_dimension}")
    return (max_height, max_width)


def convert_to_rgb_transform(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


class IdeficsImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Idefics image processor.

    Args:
        image_size (`int` *optional*, defaults to `224`):
            Resize to image size
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to the `(max_height, max_width)` of the images in the batch. Can be overridden by
            the `do_pad` parameter in the `preprocess` method.
        eval_mode (`bool`, *optional*, defaults to `True`):
            Whether to prepare the image for evaluation or training. If `eval_mode` is `True` resize will be performed,
            otherwise a `RandomResizeCrop`

    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        image_size: int = 224,
        image_mean: Optional[Union[float, List[float]]] = IMAGENET_STANDARD_MEAN,
        image_std: Optional[Union[float, List[float]]] = IMAGENET_STANDARD_STD,
        do_pad: bool = True,
        eval_mode: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_pad = do_pad

        print(f"{self.image_size=}")

    def _pad_image(
        self,
        image: np.ndarray,
        output_size: Tuple[int, int],
        constant_values: Union[float, Iterable[float]] = 0,
        data_format: Optional[ChannelDimension] = None,
    ) -> np.ndarray:
        """
        Pad an image with zeros to the given size.
        """
        input_height, input_width = get_image_size(image)
        output_height, output_width = output_size

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width
        padding = ((0, pad_bottom), (0, pad_right))
        padded_image = pad(
            image, padding, mode=PaddingMode.CONSTANT, constant_values=constant_values, data_format=data_format
        )
        return padded_image

    def pad(
        self,
        images: List[np.ndarray],
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = None,
    ) -> BatchFeature:
        """
        Pads a batch of images with zeros to the size of largest height and width in the batch and optionally returns
        their corresponding pixel mask.

        Args:
            images (`List[np.ndarray]`):
                Batch of images to pad.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        pad_size = get_max_height_width(images)
        padded_images = [
            self._pad_image(image=image, output_size=pad_size, data_format=data_format) for image in images
        ]
        data = {"pixel_values": padded_images}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def preprocess(
        self,
        images: ImageInput,
        image_size: Optional[Dict[str, int]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_pad: Optional[bool] = None,
        eval_mode: Optional[bool] = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess.
            image_size (`int`, *optional*, defaults to `self.image_size`):
                Controls the size of the image after `resize`.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image to the (max_height, max_width) in the batch. If `True`, a pixel mask is also
                created and returned.
            eval_mode (`bool`, *optional*, defaults to `self.eval_mode`):
                Whether to prepare the image for evaluation or training. If `eval_mode` is `True` resize will be
                performed, otherwise a `RandomResizeCrop`
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        image_size = image_size if image_size is not None else self.image_size
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_pad = do_pad if do_pad is not None else self.do_pad

        print(f"{image_size}")
        print(f"{image_mean}")
        # die

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # XXX: ideally the transformers set should be in __init__ but then `__repr__` can't handle Compose or Sequential
        # TypeError: Object of type Sequential is not JSON serializable
        # alternatively have to override `__repr__` here to do something about it
        interpolation = transforms.InterpolationMode.BICUBIC
        if eval_mode:
            resize_transform = transforms.Resize((self.image_size, self.image_size), interpolation=interpolation)
        else:
            resize_transform = transforms.RandomResizedCrop(
                (self.image_size, self.image_size), scale=(0.9, 1.0), interpolation=interpolation
            )
        image_transforms = transforms.Compose(
            [
                convert_to_rgb_transform,
                resize_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )

        images = [image_transforms(image) for image in images if image is not None]

        if do_pad:
            encoded_outputs = self.pad(images, return_tensors=return_tensors)
        else:
            encoded_outputs = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)

        return encoded_outputs
