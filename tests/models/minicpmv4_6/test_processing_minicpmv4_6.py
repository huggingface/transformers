# Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import MiniCPMV4_6Processor

if is_torch_available():
    import torch


@require_vision
@require_torch
@require_torchvision
@unittest.skip("Model not yet released, tests will fail to download processor config")
class MiniCPMV4_6ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MiniCPMV4_6Processor
    # TODO: the repo isn't yet there, we need to test it works before release and skip all tests right before merge
    # Re-enable tests after release
    model_id = "openbmb/MiniCPM-V-4_6"

    video_text_kwargs_max_length = 600
    video_text_kwargs_override_max_length = 550
    video_unstructured_max_length = 600

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token

    def test_image_processing(self):
        """Test that the processor correctly handles image inputs."""
        processor = self.get_processor()
        text = self.prepare_text_inputs(modalities=["image"])
        image_input = self.prepare_image_inputs()
        inputs = processor(text=text, images=image_input, return_tensors="pt")

        self.assertIn("pixel_values", inputs)
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("target_sizes", inputs)
        self.assertIsInstance(inputs["pixel_values"], torch.Tensor)
        self.assertEqual(inputs["pixel_values"].shape[0], 1)

    def test_video_processing(self):
        """Test that the processor correctly handles video inputs."""
        processor = self.get_processor()
        text = self.prepare_text_inputs(modalities=["video"])
        video_input = self.prepare_video_inputs()
        inputs = processor(text=text, videos=video_input, return_tensors="pt")

        self.assertIn("pixel_values_videos", inputs)
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("target_sizes_videos", inputs)
        self.assertIsInstance(inputs["pixel_values_videos"], torch.Tensor)
        self.assertEqual(inputs["pixel_values_videos"].shape[0], 1)

    def test_text_only_processing(self):
        """Test that the processor works with text-only input (no images)."""
        processor = self.get_processor()
        text = "Hello, how are you?"
        inputs = processor(text=text, return_tensors="pt")

        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertEqual(inputs["input_ids"].ndim, 2)
        self.assertEqual(inputs["attention_mask"].ndim, 2)

    def test_batch_text_only(self):
        """Test batch text-only processing."""
        processor = self.get_processor()
        texts = ["Hello", "World, this is a longer sentence"]
        inputs = processor(text=texts, return_tensors="pt")

        self.assertEqual(inputs["input_ids"].shape[0], 2)
        self.assertEqual(inputs["attention_mask"].shape[0], 2)

    def test_post_process_image_text_to_text(self):
        """Test the post-processing method."""
        processor = self.get_processor()
        generated_ids = torch.tensor([[1, 2, 3, 4, 5]])
        texts = processor.post_process_image_text_to_text(generated_ids)
        self.assertEqual(len(texts), 1)
        self.assertIsInstance(texts[0], str)

    def test_post_process_skip_special_tokens_param(self):
        """Verify skip_special_tokens can be passed as argument without conflict."""
        processor = self.get_processor()
        generated_ids = torch.tensor([[1, 2, 3, 4, 5]])
        texts_skip = processor.post_process_image_text_to_text(generated_ids, skip_special_tokens=True)
        texts_no_skip = processor.post_process_image_text_to_text(generated_ids, skip_special_tokens=False)
        self.assertEqual(len(texts_skip), 1)
        self.assertEqual(len(texts_no_skip), 1)

    def test_use_image_id_kwarg(self):
        """Test that use_image_id is correctly routed through _merge_kwargs."""
        processor = self.get_processor()
        text = f"{self.image_token}Describe."
        image_input = self.prepare_image_inputs()

        inputs_with_id = processor(text=text, images=image_input, use_image_id=True, return_tensors="pt")
        inputs_without_id = processor(text=text, images=image_input, use_image_id=False, return_tensors="pt")

        # With use_image_id=True, input_ids should contain image_id tokens -> different sequences
        self.assertFalse(
            torch.equal(inputs_with_id["input_ids"], inputs_without_id["input_ids"]),
            "use_image_id should produce different input_ids when True vs False",
        )
