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
"""Testing suite for the PyTorch LightOnOCR model."""

import gc
import unittest

import requests
from PIL import Image

from transformers import LightOnOCRConfig, LightOnOCRForConditionalGeneration, is_torch_available
from transformers.testing_utils import (
    backend_empty_cache,
    require_torch,
    require_vision,
    slow,
    torch_device,
)


if is_torch_available():
    import torch


@require_torch
class LightOnOCRModelTest(unittest.TestCase):
    """Basic tests for LightOnOCR model."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LightOnOCRConfig()
        self.config.text_config.num_hidden_layers = 2
        self.config.vision_config.num_hidden_layers = 2

    def tearDown(self):
        """Clean up after each test."""
        gc.collect()
        backend_empty_cache(torch_device)

    def test_model_creation(self):
        """Test that model can be created from config."""
        model = LightOnOCRForConditionalGeneration(self.config)
        self.assertIsInstance(model, LightOnOCRForConditionalGeneration)

    @require_vision
    def test_model_forward_with_image(self):
        """Test forward pass with image input."""
        model = LightOnOCRForConditionalGeneration(self.config)
        model.eval()

        # Create dummy inputs
        batch_size = 1
        seq_len = 20

        # Create input_ids with image token
        input_ids = torch.randint(0, self.config.text_config.vocab_size, (batch_size, seq_len), device=torch_device)
        # Replace some tokens with image token
        input_ids[:, 5:15] = self.config.image_token_id

        # Create dummy pixel values (batch, channels, height, width)
        pixel_values = torch.randn(
            batch_size,
            self.config.vision_config.num_channels,
            self.config.vision_config.image_size,
            self.config.vision_config.image_size,
            device=torch_device,
        )

        # Image sizes (height, width)
        image_sizes = [(self.config.vision_config.image_size, self.config.vision_config.image_size)]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, image_sizes=image_sizes)

        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape[0], batch_size)
        self.assertEqual(outputs.logits.shape[1], seq_len)
        self.assertEqual(outputs.logits.shape[2], self.config.text_config.vocab_size)

    @require_vision
    def test_generation_with_image(self):
        """Test that model can generate text from image input (OCR task)."""
        model = LightOnOCRForConditionalGeneration(self.config)
        model.eval()

        batch_size = 1
        seq_len = 20

        # Create input_ids with image tokens and some text
        input_ids = torch.randint(0, self.config.text_config.vocab_size, (batch_size, seq_len), device=torch_device)
        input_ids[:, 0:10] = self.config.image_token_id  # First 10 tokens are image tokens

        # Create dummy pixel values
        pixel_values = torch.randn(
            batch_size,
            self.config.vision_config.num_channels,
            self.config.vision_config.image_size,
            self.config.vision_config.image_size,
            device=torch_device,
        )

        image_sizes = [(self.config.vision_config.image_size, self.config.vision_config.image_size)]

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids, pixel_values=pixel_values, image_sizes=image_sizes, max_new_tokens=10, do_sample=False
            )

        self.assertEqual(generated_ids.shape[0], batch_size)
        self.assertGreater(generated_ids.shape[1], input_ids.shape[1])  # should have generated tokens

    @require_vision
    def test_model_outputs_with_labels(self):
        """Test that loss is computed when labels are provided (for OCR training)."""
        model = LightOnOCRForConditionalGeneration(self.config)
        model.train()

        batch_size = 1
        seq_len = 15

        # Create input_ids with image tokens
        input_ids = torch.randint(0, self.config.text_config.vocab_size, (batch_size, seq_len), device=torch_device)
        input_ids[:, 0:5] = self.config.image_token_id  # First 5 tokens are image tokens

        # Create dummy pixel values
        pixel_values = torch.randn(
            batch_size,
            self.config.vision_config.num_channels,
            self.config.vision_config.image_size,
            self.config.vision_config.image_size,
            device=torch_device,
        )

        image_sizes = [(self.config.vision_config.image_size, self.config.vision_config.image_size)]
        labels = torch.randint(0, self.config.text_config.vocab_size, (batch_size, seq_len), device=torch_device)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, image_sizes=image_sizes, labels=labels)

        self.assertIsNotNone(outputs.loss)
        self.assertIsInstance(outputs.loss.item(), float)


@slow
@require_torch
@require_vision
class LightOnOCRIntegrationTest(unittest.TestCase):
    """Integration tests with actual model checkpoints (slow tests)."""

    def setUp(self):
        """Set up test fixtures."""
        # URL for a test image (simple OCR-like text image)
        self.image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"

    def tearDown(self):
        """Clean up after each test."""
        gc.collect()
        backend_empty_cache(torch_device)

    def get_test_image(self):
        """Download and return test image."""
        image = Image.open(requests.get(self.image_url, stream=True).raw)
        return image

    @unittest.skip("No public pretrained LightOnOCR model available yet")
    def test_inference_with_pretrained_model(self):
        """
        Test inference with a pretrained model.
        This test should be enabled once a pretrained model is available.
        """
        # Example code for when a pretrained model is available:
        # processor = LightOnOCRProcessor.from_pretrained("lighton/lightonocr-base")
        # model = LightOnOCRForConditionalGeneration.from_pretrained(
        #     "lighton/lightonocr-base", device_map="auto"
        # )
        # model.eval()
        #
        # image = self.get_test_image()
        # prompt = "<|image_pad|>Extract text from this image:"
        #
        # inputs = processor(images=image, text=prompt, return_tensors="pt")
        # inputs = {k: v.to(model.device) for k, v in inputs.items()}
        #
        # with torch.no_grad():
        #     generated_ids = model.generate(**inputs, max_new_tokens=50)
        #
        # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # self.assertIsInstance(generated_text, str)
        # self.assertGreater(len(generated_text), 0)
        pass


if __name__ == "__main__":
    unittest.main()
