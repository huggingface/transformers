# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import shutil
import tempfile
import unittest
import requests

import numpy as np

from transformers.testing_utils import (
    is_pt_tf_cross_test,
    require_tf,
    require_torch,
    require_torchvision,
    require_vision,
)

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs

from transformers.utils import is_torch_available, is_vision_available

if is_vision_available():
    from PIL import Image
    from transformers import RtDetrImageProcessor

if is_torch_available():
    import torch


class RtDetrImageProcessingTester(unittest.TestCase):
    def __init__(
        self,
        do_resize=True,
        size=640,
        do_rescale=True,
        rescale_factor=1 / 255,
        return_tensors="pt",
    ):
        self.do_resize = do_resize
        self.size = size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.return_tensors = return_tensors
        self.num_channels = 3
        self.batch_size = 8

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "size": self.size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "return_tensors": self.return_tensors,
            "num_channels": self.num_channels
        }
    
    # def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
    #     url_images = [
    #         "http://images.cocodataset.org/val2017/000000000139.jpg",
    #         "http://images.cocodataset.org/val2017/000000000285.jpg",
    #         "http://images.cocodataset.org/val2017/000000000632.jpg",
    #         "http://images.cocodataset.org/val2017/000000000724.jpg",
    #         "http://images.cocodataset.org/val2017/000000000776.jpg",
    #         "http://images.cocodataset.org/val2017/000000000785.jpg",
    #         "http://images.cocodataset.org/val2017/000000000802.jpg",
    #         "http://images.cocodataset.org/val2017/000000000872.jpg"]
    #     images = [Image.open(requests.get(url, stream=True).raw) for url in url_images]
    #     image_inputs = [torch.from_numpy(np.array(image)) for image in images]
    #     return image_inputs

    def get_expected_values(self):
        return self.size, self.size

    def expected_output_image_shape(self, images):
        height, width = self.get_expected_values()
        return self.num_channels, height, width

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=1,
            max_resolution=1024,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )

        
@require_torch
@require_vision
class RtDetrImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = RtDetrImageProcessor if is_vision_available() else None
    
    def setUp(self):
        self.image_processor_tester = RtDetrImageProcessingTester()

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        image_processing = self.image_processing_class(**self.image_processor_dict)
        self.assertTrue(hasattr(image_processing, "do_resize"))
        self.assertTrue(hasattr(image_processing, "size"))
        self.assertTrue(hasattr(image_processing, "do_rescale"))
        self.assertTrue(hasattr(image_processing, "rescale_factor"))
        self.assertTrue(hasattr(image_processing, "return_tensors"))


    
