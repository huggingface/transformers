# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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

"""Image processor class for Hiera."""


import requests
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from ...image_processing_utils import BaseImageProcessor
from ...utils import is_vision_available, logging


if is_vision_available():
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode


logger = logging.get_logger(__name__)


class HieraImageProcessor(BaseImageProcessor):
    def __init__(self, size):
        self.size = size
        self.transform_list = [
            transforms.Resize(int((256 / 224) * self.size), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.size),
        ]
        self.transform_vis = transforms.Compose(self.transform_list)
        self.transform_norm = transforms.Compose(
            self.transform_list
            + [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )

    def process_image(self, image_url):
        # Load the image
        img = Image.open(requests.get(image_url, stream=True).raw)

        # Apply transformations
        # img_vis = self.transform_vis(img)
        img_norm = self.transform_norm(img)

        return img_norm
