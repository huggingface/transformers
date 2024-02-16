
"""Image processor class for Hirea."""

from typing import Dict, List, Optional, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import rescale, resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_vision_available, logging
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import requests


if is_vision_available():
    import PIL


logger = logging.get_logger(__name__)


class HieraImageProcessor(BaseImageProcessor):
    def __init__(self, size):
        self.size = size
        self.transform_list = [
            transforms.Resize(int((256 / 224) * self.size), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.size)
        ]
        self.transform_vis = transforms.Compose(self.transform_list)
        self.transform_norm = transforms.Compose(self.transform_list + [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
    
    def process_image(self, image_url):
        # Load the image
        img = Image.open(requests.get(image_url, stream=True).raw)
        
        # Apply transformations
        img_vis = self.transform_vis(img)
        img_norm = self.transform_norm(img)
        
        return img_norm