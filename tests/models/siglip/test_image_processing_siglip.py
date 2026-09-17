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

from ...test_image_processing_common import ImageProcessingTester, ImageProcessingTestMixin


class SiglipImageProcessingTester(ImageProcessingTester):
    def __init__(self, **kwargs):
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"height": 18, "width": 18})
        kwargs.setdefault("do_rescale", True)
        kwargs.setdefault("rescale_factor", 1 / 255)
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [0.5, 0.5, 0.5])
        kwargs.setdefault("image_std", [0.5, 0.5, 0.5])
        super().__init__(**kwargs)


@require_torch
@require_vision
# Copied from tests.models.clip.test_image_processing_clip.CLIPImageProcessingTest with CLIP->Siglip
class SiglipImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_tester_class = SiglipImageProcessingTester

    @property
    def image_processor_dict(self):
        return self.image_processing_tester.prepare_image_processor_dict()

    # Ignore copy
    # Ignore copy
    @unittest.skip(reason="not supported")
    # Ignore copy
    def test_call_numpy_4_channels(self):
        pass
