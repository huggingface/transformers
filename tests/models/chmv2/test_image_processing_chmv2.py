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
    num_labels = 5

    # Image processor init kwargs
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    do_normalize = True
    do_resize = True
    size = {"height": 512, "width": 512}
    keep_aspect_ratio = False
    do_pad = False


@require_torch
@require_vision
class CHMv2ImageProcessingTest(ImageProcessingTestMixin, PostProcessSemanticSegmentationTestMixin, unittest.TestCase):
    image_processing_tester_class = CHMv2ImageProcessingTester

    @unittest.skip(reason="CHMv2 only has a fast image processor, no slow version")
    def test_image_processor_save_load_with_autoimageprocessor(self):
        pass
