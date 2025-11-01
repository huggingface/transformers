# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

"""
Image Processor class for Phi3-V.
"""

from typing import List, Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import convert_to_rgb
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    make_list_of_images,
    valid_images,
)
from ...utils import TensorType, is_torch_available, is_torchvision_available, is_vision_available, logging


if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

if is_torchvision_available():
    import torchvision


logger = logging.get_logger(__name__)


def padding_336(b):
    width, height = b.size
    tar = int(np.ceil(height / 336) * 336)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = torchvision.transforms.functional.pad(
        b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255, 255, 255]
    )

    return b


def calc_padded_size(width, height, padding_unit=336):
    target_height = int(np.ceil(height / padding_unit) * padding_unit)
    top_padding = int((target_height - height) / 2)
    bottom_padding = target_height - height - top_padding
    left_padding = 0
    right_padding = 0
    padded_width = width + left_padding + right_padding
    padded_height = height + top_padding + bottom_padding
    return padded_width, padded_height


def HD_transform(img, hd_num=16):
    is_tensor = isinstance(img, torch.Tensor)
    is_numpy = isinstance(img, np.ndarray)

    if is_numpy:
        img = Image.fromarray(img)
    elif is_tensor:
        img = torchvision.transforms.functional.to_pil_image(img)
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * 336)
    new_h = int(new_w / ratio)

    img = torchvision.transforms.functional.resize(
        img,
        [new_h, new_w],
    )
    img = padding_336(img)
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img


def calc_hd_transform_size(width, height, hd_num=16):
    transposed = False
    if width < height:
        width, height = height, width
        transposed = True

    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1

    new_width = int(scale * 336)
    new_height = int(new_width / ratio)

    padded_width, padded_height = calc_padded_size(new_width, new_height)

    if transposed:
        padded_width, padded_height = padded_height, padded_width

    return padded_width, padded_height


def pad_to_max_num_crops_tensor(images, max_crops=5):
    """
    images: B x 3 x H x W, B<=max_crops
    """
    B, _, H, W = images.shape
    if B < max_crops:
        pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
        images = torch.cat([images, pad], dim=0)
    return images


class Phi3VImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Phi3 image processor. Based on [`CLIPImageProcessor`] with incorporation of additional techniques
    for processing high resolution images as explained in the [InternLM-XComposer2-4KHD](https://arxiv.org/pdf/2404.06512)

    Args:
        num_crops (`int`, *optional*, defaults to 1): Number of different image crops to use.
        num_img_tokens (`int`, *optional*, defaults to 144): Number of tokens to generate per image
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        num_crops: int = 1,
        num_img_tokens: int = 144,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_crops = num_crops
        self.num_img_tokens = num_img_tokens
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb

    def calc_num_image_tokens(self, images: ImageInput):
        """Calculate the number of image tokens for each image.
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
        """
        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        images = [image.convert("RGB") for image in images]
        # (H, W, C)
        elems = [HD_transform(im, hd_num=self.num_crops) for im in images]
        shapes = [[im.size[1], im.size[0]] for im in elems]
        num_img_tokens = [int((h // 336 * w // 336 + 1) * 144 + 1 + (h // 336 + 1) * 12) for h, w in shapes]
        return num_img_tokens

    def calc_num_image_tokens_from_image_size(self, width, height):
        """
        Calculate the number of image tokens for a given image size.
        Args:
            width (`int`): Width of the image.
            height (`int`): Height of the image.
        """
        new_width, new_height = calc_hd_transform_size(width, height, hd_num=self.num_crops)
        num_img_tokens = int((new_height // 336 * new_width // 336 + 1) * 144 + 1 + (new_height // 336 + 1) * 12)
        return num_img_tokens

    def preprocess(
        self,
        images: ImageInput,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        """
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if not isinstance(images, List):
            images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        img_processor = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(image_mean, image_std)]
        )

        # PIL images
        # HD_transform pad images to size of multiply of 336, 336
        # convert to RGB first
        images = [convert_to_rgb(image) for image in images]
        elems = [HD_transform(im, hd_num=self.num_crops) for im in images]
        # tensor transform and normalize
        hd_images = [img_processor(im) for im in elems]
        # create global image
        global_image = [
            torch.nn.functional.interpolate(
                im.unsqueeze(0).float(),
                size=(336, 336),
                mode="bicubic",
            ).to(im.dtype)
            for im in hd_images
        ]

        # [(3, h, w)], where h, w is multiple of 336
        shapes = [[im.size(1), im.size(2)] for im in hd_images]
        num_img_tokens = [int(((h // 336) * (w // 336) + 1) * 144 + 1 + (h // 336 + 1) * 12) for h, w in shapes]
        # reshape to channel dimension -> (num_images, num_crops, 3, 336, 336)
        # (1, 3, h//336, 336, w//336, 336) -> (1, h//336, w//336, 3, 336, 336) -> (h//336*w//336, 3, 336, 336)
        hd_images_reshape = [
            im.reshape(1, 3, h // 336, 336, w // 336, 336)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(-1, 3, 336, 336)
            .contiguous()
            for im, (h, w) in zip(hd_images, shapes)
        ]
        # concat global image and local image
        hd_images_reshape = [
            torch.cat([_global_image] + [_im], dim=0) for _global_image, _im in zip(global_image, hd_images_reshape)
        ]

        # pad to max_num_crops
        image_transformed = [pad_to_max_num_crops_tensor(im, self.num_crops + 1) for im in hd_images_reshape]
        image_transformed = torch.stack(image_transformed, dim=0)
        padded_images = image_transformed
        image_sizes = shapes

        data = {"pixel_values": padded_images, "image_sizes": image_sizes, "num_img_tokens": num_img_tokens}

        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["Phi3VImageProcessor"]
