# coding=utf-8
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

import tempfile
import unittest

from transformers import AutoTokenizer, DeepseekOcrProcessor
from transformers.testing_utils import get_tests_dir, require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import DeepseekOcrImageProcessorFast


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_vision
class DeepseekOcrProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = DeepseekOcrProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = DeepseekOcrImageProcessorFast()
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
        processor_kwargs = self.prepare_processor_dict()
        processor = self.processor_class(
            image_processor=image_processor,
            tokenizer=tokenizer,
            **processor_kwargs,
        )
        processor.save_pretrained(self.tmpdirname)
        self.image_token = processor.image_token

    @staticmethod
    def prepare_processor_dict():
        return {
            "image_token": "<image>",
        }

    def get_tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_image_processor(self, **kwargs):
        processor = self.processor_class.from_pretrained(self.tmpdirname, **kwargs)
        return processor.image_processor

    @require_torch
    def test_image_token_expansion(self):
        from PIL import Image

        processor = self.processor_class.from_pretrained(self.tmpdirname)
        image = Image.new("RGB", (64, 64), color="red")

        text = f"{processor.image_token} Describe this image."
        inputs = processor(text=text, images=image, return_tensors="pt")

        self.assertIn("input_ids", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertIn("image_attention_mask", inputs)
        self.assertIn("num_img_tokens", inputs)

        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        num_image_tokens = (inputs["input_ids"][0] == image_token_id).sum().item()

        self.assertGreater(num_image_tokens, 1)

    @require_torch
    def test_image_attention_mask_generation(self):
        import torch
        from PIL import Image

        processor = self.processor_class.from_pretrained(self.tmpdirname)
        image = Image.new("RGB", (64, 64), color="blue")

        text = f"{processor.image_token} What is in this image?"
        inputs = processor(text=text, images=image, return_tensors="pt")

        image_attention_mask = inputs["image_attention_mask"]
        self.assertIsInstance(image_attention_mask, torch.Tensor)
        self.assertEqual(image_attention_mask.dtype, torch.bool)

        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        expected_mask = inputs["input_ids"] == image_token_id
        torch.testing.assert_close(image_attention_mask, expected_mask)

    @require_torch
    def test_num_img_tokens_handling(self):
        from PIL import Image

        processor = self.processor_class.from_pretrained(self.tmpdirname)
        image1 = Image.new("RGB", (64, 64), color="red")
        image2 = Image.new("RGB", (128, 128), color="green")

        text = [
            f"{processor.image_token} First image.",
            f"{processor.image_token} Second image.",
        ]
        inputs = processor(text=text, images=[image1, image2], return_tensors="pt")

        self.assertIn("num_img_tokens", inputs)
        self.assertIsInstance(inputs["num_img_tokens"], list)
        self.assertEqual(len(inputs["num_img_tokens"]), 2)

    @require_torch
    def test_processor_with_multiple_images(self):
        from PIL import Image

        processor = self.processor_class.from_pretrained(self.tmpdirname)
        image1 = Image.new("RGB", (64, 64), color="red")
        image2 = Image.new("RGB", (64, 64), color="blue")

        text = f"{processor.image_token}{processor.image_token} Two images here."
        inputs = processor(text=text, images=[image1, image2], return_tensors="pt")

        self.assertIn("input_ids", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertEqual(len(inputs["num_img_tokens"]), 2)

    @require_torch
    def test_processor_error_on_token_mismatch(self):
        from PIL import Image

        processor = self.processor_class.from_pretrained(self.tmpdirname)
        image = Image.new("RGB", (64, 64), color="red")

        text = "No image token here."

        with self.assertRaises(ValueError) as context:
            processor(text=text, images=image, return_tensors="pt")

        self.assertIn("does not match", str(context.exception))
