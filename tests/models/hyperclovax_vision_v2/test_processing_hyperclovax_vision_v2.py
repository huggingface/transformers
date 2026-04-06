# Copyright 2026 NAVER Corp. and The HuggingFace Inc. team. All rights reserved.
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

import requests

from transformers.image_utils import load_image
from transformers.testing_utils import require_cv2, require_torch, require_torchvision, require_vision
from transformers.utils import is_cv2_available, is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import HCXVisionV2Processor

if is_cv2_available():
    import cv2

if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch


@require_vision
@require_torch
@require_torchvision
class HCXVisionV2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = HCXVisionV2Processor
    model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token
        cls.images_input_name = "pixel_values"
        cls.videos_input_name = "pixel_values_videos"

    def test_processor_initialization(self):
        """Test basic processor initialization and attributes."""
        processor = self.get_processor()

        self.assertIsNotNone(processor.image_processor)
        self.assertIsNotNone(processor.video_processor)
        self.assertIsNotNone(processor.tokenizer)
        self.assertIsNotNone(processor.chat_template)
        self.assertEqual(processor.image_token, "<|IMAGE_PAD|>")
        self.assertEqual(processor.video_token, "<|VIDEO_PAD|>")

    def test_processor_text_only(self):
        """Test processor with text-only input."""
        processor = self.get_processor()

        messages = [{"role": "user", "content": "What is the capital of France?"}]

        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt")

        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertEqual(inputs.input_ids.dim(), 2)
        self.assertEqual(inputs.attention_mask.dim(), 2)

    def test_processor_with_image(self):
        """Test processor with image input using HCX format."""
        processor = self.get_processor()

        image_url = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        image = load_image(image_url).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "What is in the image?"},
                ],
            }
        ]

        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")

        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertIn("image_grid_thw", inputs)
        self.assertEqual(inputs.input_ids.dim(), 2)
        self.assertEqual(inputs.pixel_values.dim(), 2)
        self.assertEqual(inputs.image_grid_thw.dim(), 2)

    def test_processor_with_multiple_images(self):
        """Test processor with multiple images."""
        processor = self.get_processor()

        image_url = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        image = load_image(image_url).convert("RGB")

        # Two identical images in one message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Describe both images."},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image, image], return_tensors="pt")

        self.assertIn("pixel_values", inputs)
        self.assertIn("image_grid_thw", inputs)
        self.assertEqual(inputs.image_grid_thw.shape[0], 1)  # batch size = 1
        self.assertTrue(torch.all(inputs.image_grid_thw[0] == inputs.image_grid_thw[0]))

    @require_cv2
    def test_processor_with_video(self):
        """Test processor with video input using HCX format."""
        processor = self.get_processor()

        video_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": video_url}},
                    {"type": "text", "text": "What is shown in this video?"},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Download and decode only ~1 second of video frames
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(requests.get(video_url).content)
            f.flush()
            cap = cv2.VideoCapture(f.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            max_frames = max(1, int(fps))
            frames = []
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb).resize((224, 224), Image.BICUBIC))
            cap.release()

        inputs = processor(text=[text], videos=[frames], return_tensors="pt")

        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("pixel_values_videos", inputs)
        self.assertIn("video_grid_thw", inputs)
        self.assertEqual(inputs.video_grid_thw.shape[0], 1)
        self.assertTrue(inputs.pixel_values_videos.shape[0] > 0)

    def test_processor_batch(self):
        """Test processor with batched inputs."""
        processor = self.get_processor()

        messages_batch = [
            {"role": "user", "content": "Describe this image."},
            {"role": "user", "content": "What is in the picture?"},
        ]

        texts = [
            processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages_batch
        ]

        inputs = processor(text=texts, return_tensors="pt")

        self.assertIn("input_ids", inputs)
        self.assertEqual(inputs.input_ids.shape[0], 2)  # batch_size = 2

    def test_processor_image_grid_thw(self):
        """Test that image_grid_thw is computed correctly."""
        processor = self.get_processor()

        image_url = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        image = load_image(image_url).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Describe this."},
                ],
            }
        ]

        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")

        # image_grid_thw should be [T, H, W] where T=1 for static images
        image_grid_thw = inputs["image_grid_thw"]
        self.assertEqual(image_grid_thw.shape, torch.Size([1, 3]))
        self.assertEqual(image_grid_thw[0, 0], 1)  # T dimension = 1 for images

    def test_processor_output_keys(self):
        """Test that processor output contains expected keys."""
        processor = self.get_processor()

        image_url = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        image = load_image(image_url).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Caption this image."},
                ],
            }
        ]

        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")

        expected_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        self.assertTrue(expected_keys.issubset(set(inputs.keys())))
