# Create the file with content
$content = @'
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Image processor class for ResNet."""

from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image

from ...image_processing_utils import BaseImageProcessor, BatchImageProcessor, get_size_dict
from ...image_transforms import (
    normalize,
    resize,
    to_channel_dimension_format,
    to_numpy_array,
)
from ...utils import is_vision_available, logging

if is_vision_available():
    pass

logger = logging.get_logger(__name__)

IMAGENET_STANDARD_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_STD = [0.229, 0.224, 0.225]


class ResNetImageProcessor(BaseImageProcessor):
    r"""
    Constructs a ResNet image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`.
        size (`dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the output image after resizing.
        resample (`int`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
            Uses BICUBIC (3) to match the original ResNet torchvision implementation.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Controlled by the `image_mean` and `image_std` attributes.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: int = Image.BICUBIC,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 224, "width": 224}
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def preprocess(
        self,
        images: Union[Image.Image, np.ndarray],
        size: Optional[Dict[str, int]] = None,
        resample: int = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images: Image to preprocess.
            size: Size of the output image.
            resample: Resampling filter.
            do_normalize: Whether to normalize.
            image_mean: Mean for normalization.
            image_std: Std for normalization.
            return_tensors: Type of tensors to return.

        Returns:
            Preprocessed image.
        """
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = to_numpy_array(images)

        if self.do_resize:
            images = resize(images, size=size, resample=resample)

        if do_normalize:
            images = normalize(images, mean=image_mean, std=image_std)

        if return_tensors is not None:
            images = to_channel_dimension_format(images, return_tensors)

        return BatchImageProcessor(images, return_tensors=return_tensors)
'@

New-Item -Path . -Name "image_processing_resnet.py" -Type File -Value $content
