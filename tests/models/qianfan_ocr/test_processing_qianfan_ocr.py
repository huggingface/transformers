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
"""Testing suite for the QianfanOCR processor."""

import unittest

from transformers import QianfanOCRProcessor
from transformers.testing_utils import require_torch, require_vision, slow

from ...test_processing_common import ProcessorTesterMixin


@slow
@require_vision
class QianfanOCRProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = QianfanOCRProcessor
    # Tiny processor created with make_tiny_processor.py from "bairongz/QianfanOCR"
    tiny_model_id = "hf-internal-testing/tiny-processor-qianfan_ocr"
    # QianfanOCR has no video support; images and pixel values share the same tensor key
    videos_input_name = "pixel_values"

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        # Default size=448x448 with max_patches=12 produces up to 27 MB pixel_values tensors.
        # Use 64x64 with max_patches=1 for tests — assertions only check patch count, not spatial dims.
        return image_processor_class.from_pretrained(
            cls.tiny_model_id, size={"height": 64, "width": 64}, max_patches=1
        )

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_placeholder_token

    @unittest.skip("QianfanOCR does not support video processing")
    def test_process_interleaved_images_videos(self):
        pass

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image"])
        image_input = self.prepare_images_inputs()
        inputs = processor(text=text, images=image_input, return_tensors="pt")

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    @staticmethod
    def prepare_processor_dict():
        return {"image_seq_length": 2}

    @require_torch
    def test_get_num_vision_tokens(self):
        """Tests general functionality of the helper used internally in vLLM."""
        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertIn("num_image_tokens", output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertIn("num_image_patches", output)
        self.assertEqual(len(output["num_image_patches"]), 3)
