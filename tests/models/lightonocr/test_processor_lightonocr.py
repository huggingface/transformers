# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest

import numpy as np

from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image

    from transformers import LightOnOCRProcessor

if is_torch_available():
    import torch


@require_vision
@require_torch
class LightOnOCRProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    """Test suite for LightOnOCR processor."""

    processor_class = LightOnOCRProcessor

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdirname = tempfile.mkdtemp()

        # Create a Pixtral image processor (LightOnOCR uses Pixtral vision architecture)
        image_processor = AutoImageProcessor.from_pretrained(
            "mistral-community/pixtral-12b", size={"longest_edge": 1024}
        )

        # Create a tokenizer (using Qwen2 as base)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        # Add special tokens for LightOnOCR
        special_tokens_dict = {
            "additional_special_tokens": [
                "<|image_pad|>",
                "<|vision_pad|>",
                "<|vision_end|>",
            ]
        }
        tokenizer.add_special_tokens(special_tokens_dict)

        # Set special token attributes on the tokenizer for multimodal processing
        tokenizer.image_token = "<|image_pad|>"
        tokenizer.image_break_token = "<|vision_pad|>"
        tokenizer.image_end_token = "<|vision_end|>"
        tokenizer.image_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_token)
        tokenizer.image_break_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_break_token)
        tokenizer.image_end_token_id = tokenizer.convert_tokens_to_ids(tokenizer.image_end_token)

        # Add a basic multimodal-aware chat template to the tokenizer
        # This template extracts text from the multimodal content format
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if loop.first and messages[0]['role'] != 'system' %}"
            "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
            "{% endif %}"
            "{{'<|im_start|>' + message['role'] + '\\n' }}"
            "{% if message['content'] is string %}"
            "{{ message['content'] }}"
            "{% else %}"
            "{% for content in message['content'] %}"
            "{% if content['type'] == 'text' %}"
            "{{ content['text'] }}"
            "{% elif content['type'] == 'image' %}"
            "{{ '<|image_pad|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% endif %}"
            "{{ '<|im_end|>\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\\n' }}"
            "{% endif %}"
        )

        # Create and save processor
        processor = LightOnOCRProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=14,
            spatial_merge_size=2,
        )
        processor.save_pretrained(self.tmpdirname)

        self.image_token = processor.image_token

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.tmpdirname, ignore_errors=True)

    def get_tokenizer(self, **kwargs):
        """Get tokenizer from saved processor."""
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        """Get image processor from saved processor."""
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_processor(self, **kwargs):
        """Get processor from saved directory."""
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def prepare_image_inputs(self, batch_size=None):
        """Prepare small dummy image inputs."""
        image = Image.new("RGB", (112, 112), color="red")
        if batch_size is None:
            return image
        return [image] * batch_size

    def test_processor_creation(self):
        """Test that processor can be created and loaded."""
        processor = self.get_processor()
        self.assertIsInstance(processor, LightOnOCRProcessor)
        self.assertIsNotNone(processor.tokenizer)
        self.assertIsNotNone(processor.image_processor)

    def test_processor_with_text_only(self):
        """Test processor with text input only."""
        processor = self.get_processor()
        text = "This is a test sentence."

        inputs = processor(text=text, return_tensors="pt")

        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertEqual(inputs["input_ids"].shape[0], 1)  # batch size

    def test_processor_with_image_and_text(self):
        """Test processor with both image and text inputs."""
        processor = self.get_processor()
        image = self.prepare_image_inputs()
        text = f"{self.image_token} Extract text from this image."

        inputs = processor(images=image, text=text, return_tensors="pt")

        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertIn("image_sizes", inputs)

        # Check shapes
        self.assertEqual(inputs["input_ids"].shape[0], 1)  # batch size
        self.assertEqual(len(inputs["pixel_values"].shape), 4)  # (batch, channels, height, width)
        self.assertEqual(len(inputs["image_sizes"]), 1)  # one image

    def test_processor_image_token_expansion(self):
        """Test that image token is properly expanded based on image size."""
        processor = self.get_processor()
        image = self.prepare_image_inputs()
        text = f"{self.image_token} Describe this image."

        inputs = processor(images=image, text=text, return_tensors="pt")

        # The image token should be expanded to multiple tokens based on patch size
        # Count occurrences of image_token_id in input_ids
        image_token_id = processor.image_token_id
        num_image_tokens = (inputs["input_ids"] == image_token_id).sum().item()

        # Should have multiple image tokens (one per patch after spatial merging)
        self.assertGreater(num_image_tokens, 1)

    def test_processor_batch_processing(self):
        """Test processor with batch of inputs."""
        processor = self.get_processor()
        images = self.prepare_image_inputs(batch_size=2)
        texts = [f"{self.image_token} Extract text." for _ in range(2)]

        inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)

        self.assertEqual(inputs["input_ids"].shape[0], 2)  # batch size
        self.assertEqual(inputs["pixel_values"].shape[0], 2)  # two images

    def test_processor_model_input_names(self):
        """Test that processor returns correct model input names."""
        processor = self.get_processor()

        expected_keys = {"input_ids", "attention_mask", "pixel_values", "image_sizes"}
        model_input_names = set(processor.model_input_names)

        # Check that all expected keys are in model_input_names
        for key in expected_keys:
            self.assertIn(key, model_input_names)

    def test_processor_without_images(self):
        """Test that processor handles text-only input correctly."""
        processor = self.get_processor()
        text = "This is text without any images."

        inputs = processor(text=text, return_tensors="pt")

        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertNotIn("pixel_values", inputs)
        self.assertNotIn("image_sizes", inputs)

    def test_processor_special_tokens(self):
        """Test that special tokens are properly registered."""
        processor = self.get_processor()

        # Check that image tokens are properly defined
        self.assertEqual(processor.image_token, "<|image_pad|>")
        self.assertEqual(processor.image_break_token, "<|vision_pad|>")
        self.assertEqual(processor.image_end_token, "<|vision_end|>")

        # Check that tokens have valid IDs
        self.assertIsInstance(processor.image_token_id, int)
        self.assertIsInstance(processor.image_break_token_id, int)
        self.assertIsInstance(processor.image_end_token_id, int)

    def test_processor_return_types(self):
        """Test different return types (pt, np, list)."""
        processor = self.get_processor()
        image = self.prepare_image_inputs()
        text = f"{self.image_token} Test image."

        # Test PyTorch tensors
        inputs_pt = processor(images=image, text=text, return_tensors="pt")
        self.assertIsInstance(inputs_pt["input_ids"], torch.Tensor)

        # Test NumPy arrays
        inputs_np = processor(images=image, text=text, return_tensors="np")
        self.assertIsInstance(inputs_np["input_ids"], np.ndarray)

        # Test lists
        inputs_list = processor(images=image, text=text, return_tensors=None)
        self.assertIsInstance(inputs_list["input_ids"], list)

    def test_image_sizes_output(self):
        """Test that image_sizes are correctly computed."""
        processor = self.get_processor()
        image = Image.new("RGB", (300, 400), color="blue")  # Different size
        text = f"{self.image_token} Test."

        inputs = processor(images=image, text=text, return_tensors="pt")

        self.assertIn("image_sizes", inputs)
        self.assertEqual(len(inputs["image_sizes"]), 1)
        # Image size should be a tuple of (height, width)
        self.assertEqual(len(inputs["image_sizes"][0]), 2)
