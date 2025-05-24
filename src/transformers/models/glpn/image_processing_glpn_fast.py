# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

from typing import Optional, Union, Dict, List
from PIL import Image

import numpy as np
import torchvision.transforms as T

from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from ...image_transforms import to_channel_dimension_format
from ...utils import logging, PILImageResampling

logger = logging.get_logger(__name__)


class GLPNImageProcessorFast(BaseImageProcessorFast):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size_divisor: int = 32,
        do_rescale: bool = True,
        resample=Image.Resampling.BILINEAR,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size_divisor = size_divisor
        self.do_rescale = do_rescale
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def preprocess(
        self,
        images: Union[Image.Image, np.ndarray],
        return_tensors: Optional[str] = None,
        data_format: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        if isinstance(images, list):
            raise ValueError("GLPNImageProcessorFast does not support batched inputs")

        # Convert to numpy and channel format
        image = self.to_numpy_array(images)
        image = to_channel_dimension_format(image, "channels_last", data_format=data_format)

        # Resize to closest multiple of size_divisor
        if self.do_resize:
            height, width = image.shape[:2]
            new_height = height - (height % self.size_divisor)
            new_width = width - (width % self.size_divisor)
            image = np.array(Image.fromarray(image).resize((new_width, new_height), resample=self.resample))

        # Rescale
        if self.do_rescale:
            image = image / 255.0

        # Normalize
        if self.do_normalize:
            image = (image - self.image_mean) / self.image_std

        # Convert to CHW format
        image = to_channel_dimension_format(image, "channels_first", data_format="channels_last")

        if return_tensors == "pt":
            import torch

            image = torch.tensor(image).unsqueeze(0)

        return {"pixel_values": image}