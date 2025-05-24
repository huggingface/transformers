# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling
from ...utils import add_start_docstrings
import torch
import torchvision.transforms as T


# @add_start_docstrings(
#     "Constructs a fast GLPN image processor.",
#     BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
# )
class GLPNImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 384, "width": 384}
    crop_size = None
    do_resize = True
    do_rescale = False
    do_center_crop = False
    do_normalize = True
    do_convert_rgb = True

    def _preprocess(self, images, resample=None, size=None, image_mean=None, image_std=None, **kwargs):
        transform = T.Compose([
            T.Resize((size["height"], size["width"]), interpolation=T.InterpolationMode.BICUBIC),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=image_mean, std=image_std)
        ])
        return [transform(img) for img in images]
