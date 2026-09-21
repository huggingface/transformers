# Copyright 2024 HuggingFace Inc.
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
from transformers.utils import is_vision_available

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


if is_vision_available():
    from PIL import Image


class Siglip2ImageProcessingTester(ImageProcessingTester):
    # Image processor init kwargs
    size = {"height": 18, "width": 18}
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    resample = Image.Resampling.BILINEAR
    patch_size = 16
    max_num_patches = 256

    def expected_output_image_shape(self, images):
        return self.max_num_patches, self.patch_size * self.patch_size * self.num_channels


@require_torch
@require_vision
class Siglip2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = Siglip2ImageProcessingTester

    @unittest.skip(reason="not supported")
    def test_call_numpy_4_channels(self):
        pass
