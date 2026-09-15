# Copyright 2026 HuggingFace Inc.
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


class PPDocLayoutV2ImageProcessingTester(ImageProcessingTester):
    def __init__(self, parent, **kwargs):
        kwargs.setdefault("do_resize", True)
        kwargs.setdefault("size", {"height": 40, "width": 40})
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", [0.0, 0.0, 0.0])
        kwargs.setdefault("image_std", [1.0, 1.0, 1.0])
        super().__init__(parent, **kwargs)


@require_torch
@require_vision
class PPDocLayoutV2ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = PPDocLayoutV2ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    @unittest.skip(
        reason="PPDocLayoutV2 uses antialias=False which is not supported for 4-channel images consistently"
    )
    def test_call_numpy_4_channels(self):
        pass
