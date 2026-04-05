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
from transformers.utils import is_vision_available


if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
class Sam3ProcessorTest(unittest.TestCase):
    def setUp(self):
        from transformers.models.sam3.image_processing_sam3 import Sam3ImageProcessor
        from transformers.models.sam3.processing_sam3 import Sam3Processor

        image_processor = Sam3ImageProcessor()
        # Use a small public tokenizer for testing
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.processor = Sam3Processor(image_processor=image_processor, tokenizer=tokenizer)
        self.image = Image.new("RGB", (640, 480), color=(128, 128, 128))

    def test_mixed_none_input_boxes_generates_labels(self):
        """Test that a batch with mixed None/real input_boxes produces input_boxes_labels
        that mark None entries with the pad value (-10) so the model can mask them out.
        Regression test for https://github.com/huggingface/transformers/issues/45059
        """
        inputs = self.processor(
            images=[self.image, self.image],
            text=["cat", None],
            input_boxes=[None, [[100, 100, 200, 200]]],
            return_tensors="pt",
        )

        # Both input_boxes and input_boxes_labels should be present
        self.assertIn("input_boxes", inputs)
        self.assertIn("input_boxes_labels", inputs)

        # The None entry (index 0) should have label -10 (pad value)
        self.assertEqual(inputs["input_boxes_labels"][0, 0].item(), -10)
        # The real entry (index 1) should have label 1 (positive)
        self.assertEqual(inputs["input_boxes_labels"][1, 0].item(), 1)

    def test_all_real_boxes_generates_positive_labels(self):
        """When all entries have real boxes, all labels should be 1."""
        inputs = self.processor(
            images=[self.image, self.image],
            text=["cat", "dog"],
            input_boxes=[[[50, 50, 150, 150]], [[200, 200, 300, 300]]],
            return_tensors="pt",
        )

        self.assertIn("input_boxes_labels", inputs)
        self.assertTrue((inputs["input_boxes_labels"] == 1).all())

    def test_no_input_boxes_omits_key(self):
        """When input_boxes=None, no input_boxes key should be in the output."""
        inputs = self.processor(
            images=[self.image],
            text=["cat"],
            input_boxes=None,
            return_tensors="pt",
        )

        self.assertNotIn("input_boxes", inputs)
        self.assertNotIn("input_boxes_labels", inputs)

    def test_user_provided_labels_preserved(self):
        """User-provided input_boxes_labels should not be overwritten."""
        inputs = self.processor(
            images=[self.image, self.image],
            text=["cat", "dog"],
            input_boxes=[[[50, 50, 150, 150]], [[200, 200, 300, 300]]],
            input_boxes_labels=[[1], [0]],
            return_tensors="pt",
        )

        self.assertEqual(inputs["input_boxes_labels"][0, 0].item(), 1)
        self.assertEqual(inputs["input_boxes_labels"][1, 0].item(), 0)
