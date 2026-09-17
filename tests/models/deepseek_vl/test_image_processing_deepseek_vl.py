# Copyright 2025 HuggingFace Inc.
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


import unittest

from transformers.testing_utils import require_torch, require_vision

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


# Copied from tests.models.vit.test_image_processing_vit.ViTImageProcessingTester with ViT->DeepseekVL
class DeepseekVLImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"height": 18, "width": 18})
        super().__init__(**kwargs)

    # Ignore copy
    def expected_output_image_shape(self, images):
        max_size = max(self.size["height"], self.size["width"])
        return self.num_channels, max_size, max_size


@require_torch
@require_vision
class DeepseekVLImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = DeepseekVLImageProcessingTester

    @property
    def image_processor_dict(self):
        return self.image_processing_tester.prepare_image_processor_dict()

    # Ignore copy
    @unittest.skip(reason="Not supported")
    def test_call_numpy_4_channels(self):
        pass
