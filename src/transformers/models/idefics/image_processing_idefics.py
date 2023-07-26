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

from typing import Dict, List, Optional, Union

from ...image_processing_utils import BaseImageProcessor
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    make_list_of_images,
    valid_images,
)
from ...utils import TensorType, is_torchvision_available, is_vision_available, logging


if is_torchvision_available():
    import torchvision.transforms as transforms

if is_vision_available():
    import PIL
    from PIL import Image

logger = logging.get_logger(__name__)


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
        image_num_channels (`int`, *optional*, defaults to `3`):
            Defines the number of image channels
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        eval_mode (`bool`, *optional*, defaults to `True`):
            Whether to prepare the image for evaluation or training. If `eval_mode` is `True` resize will be performed,
            otherwise a `RandomResizeCrop`

    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        image_size: int = 224,
        image_num_channels: int = 3,
        image_mean: Optional[Union[float, List[float]]] = IMAGENET_STANDARD_MEAN,
        image_std: Optional[Union[float, List[float]]] = IMAGENET_STANDARD_STD,
        eval_mode: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.image_size = image_size
        self.image_num_channels = image_num_channels
        self.image_mean = image_mean
        self.image_std = image_std

    def preprocess(
        self,
        images: ImageInput,
        image_size: Optional[Dict[str, int]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
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
            image_num_channels (`int`, *optional*, defaults to `self.image_size`):
                Defines the number of image channels
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
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

        if len(images) == 0:
            return []

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

        return images
