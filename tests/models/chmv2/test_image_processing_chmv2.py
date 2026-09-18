# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from ...test_image_processing_common import (
    ImageProcessingTester,
    ImageProcessingTestMixin,
    PostProcessSemanticSegmentationTestMixin,
)


class CHMv2ImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, num_labels=5, **kwargs):
        kwargs.setdefault("image_mean", [0.485, 0.456, 0.406])
        kwargs.setdefault("image_std", [0.229, 0.224, 0.225])
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"height": 512, "width": 512})
        kwargs.setdefault("keep_aspect_ratio", False)
        kwargs.setdefault("do_pad", False)
        super().__init__(parent, **kwargs)
        self.num_labels = num_labels


@require_torch
@require_vision
class CHMv2ImageProcessingTest(ImageProcessingTestMixin, PostProcessSemanticSegmentationTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = CHMv2ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    @unittest.skip(reason="CHMv2 only has a fast image processor, no slow version")
    def test_image_processor_save_load_with_autoimageprocessor(self):
        pass
