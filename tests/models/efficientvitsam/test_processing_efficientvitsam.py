# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torchvision, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import EfficientViTSamImageProcessor, EfficientViTSamProcessor


@require_vision
@require_torchvision
class EfficientViTSamProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = EfficientViTSamProcessor

    def test_processor_uses_efficientvit_image_processor(self):
        image_processor = EfficientViTSamImageProcessor(
            size={"longest_edge": 32}, pad_size={"height": 32, "width": 32}
        )
        processor = self.processor_class(image_processor=image_processor)
        self.assertEqual(processor.target_size, 32)
