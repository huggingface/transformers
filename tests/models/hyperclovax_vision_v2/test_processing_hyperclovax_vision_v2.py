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

import numpy as np
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
        # image_grid_thw has one row per image (not per batch item): shape[0] == num_images == 2
        self.assertEqual(inputs.image_grid_thw.shape[0], 2)
        self.assertTrue(torch.all(inputs.image_grid_thw[0] == inputs.image_grid_thw[1]))

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

        inputs = processor(text=texts, return_tensors="pt", padding=True)

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

    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data: list,
    ):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")
        if processor_name not in self.processor_class.get_attributes():
            self.skipTest(f"{processor_name} attribute not present in {self.processor_class}")
        if getattr(processor, processor_name).__class__.__name__.endswith("Fast"):
            return_tensors = "pt"

        # Messages that satisfy BOTH the HCX jinja (list user content, no
        # system role) AND the framework's tokenize=True visuals-extraction.
        batch_messages = [[{"role": "user", "content": [{"type": "text", "text": "Describe this."}]}]] * batch_size

        # Test that tokenizing with template and directly with `self.tokenizer` gives same output
        formatted_prompt = processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), batch_size)

        # Test that tokenizing with template and directly with `self.tokenizer` gives same output
        formatted_prompt_tokenized = processor.apply_chat_template(
            batch_messages, add_generation_prompt=True, tokenize=True, return_tensors=return_tensors
        )
        add_special_tokens = not (
            processor.tokenizer.bos_token and formatted_prompt[0].startswith(processor.tokenizer.bos_token)
        )
        tok_output = processor.tokenizer(
            formatted_prompt, return_tensors=return_tensors, add_special_tokens=add_special_tokens
        )
        self.assertListEqual(tok_output.input_ids.tolist(), formatted_prompt_tokenized.tolist())

        # Test that kwargs passed to processor's `__call__` are actually used
        tokenized_max = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=return_tensors,
            processor_kwargs={
                "padding": "max_length",
                "truncation": True,
                "max_length": self.chat_template_max_length,
            },
        )
        self.assertEqual(len(tokenized_max[0]), self.chat_template_max_length)

        # Test that `return_dict=True` returns text related inputs in the dict
        out_dict_text = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
        )
        self.assertTrue(all(k in out_dict_text for k in ["input_ids", "attention_mask"]))
        self.assertEqual(len(out_dict_text["input_ids"]), batch_size)
        self.assertEqual(len(out_dict_text["attention_mask"]), batch_size)

        # Test media handling
        # When input_data contains numpy arrays (decoded video), there is no URL
        # to embed in the HCX template.  That case is tested separately in
        # test_apply_chat_template_decoded_video_0.
        if not input_data or not isinstance(input_data[0], str):
            return

        # Build HCX-format messages with media.
        # HCX template distinguishes image vs video by file extension in the URL;
        # a dummy filename is enough – the content is never fetched here.
        dummy_url = "test_video.mp4" if modality == "video" else "test_image.jpg"
        # Include string system content here; we use tokenizer.apply_chat_template
        # (no framework visuals-extraction) so string system content is fine.
        media_messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {"type": "image_url", "image_url": {"url": dummy_url}},
                    ],
                },
            ]
            for _ in range(batch_size)
        ]

        # tokenizer.apply_chat_template does NOT normalise "image_url" → "image",
        # so the HCX jinja receives the correct type and emits the media token.
        formatted_with_media = [
            processor.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            for msgs in media_messages
        ]
        expected_token = self.video_token if modality == "video" else self.image_token
        for prompt in formatted_with_media:
            self.assertIn(expected_token, prompt)

        # Process with synthetic media (avoids network / av dependency).
        synthetic_frame = Image.fromarray(np.random.randint(0, 255, (30, 400, 3), dtype=np.uint8))
        if modality == "image":
            out = processor(
                text=formatted_with_media,
                images=[synthetic_frame] * batch_size,
                return_tensors=return_tensors,
                padding=True,
            )
        else:
            out = processor(
                text=formatted_with_media,
                videos=[[synthetic_frame, synthetic_frame]] * batch_size,
                return_tensors=return_tensors,
                padding=True,
            )

        input_attr = getattr(self, input_name)
        self.assertIn(input_attr, out)
        self.assertEqual(out["input_ids"].shape[0], batch_size)

        assistant_message = {"role": "assistant", "content": "It is the sound of"}
        continue_messages = [msgs + [assistant_message] for msgs in media_messages]
        continue_prompts = [
            processor.tokenizer.apply_chat_template(msgs, continue_final_message=True, tokenize=False)
            for msgs in continue_messages
        ]
        for prompt in continue_prompts:
            self.assertTrue(prompt.endswith("It is the sound of"))

    @require_torch
    def test_apply_chat_template_decoded_video_0(self):
        processor = self.get_processor()
        if "video_processor" not in self.processor_class.get_attributes():
            self.skipTest("video_processor not in processor attributes")

        dummy_frames_np = self.prepare_video_inputs()  # shape [8, 3, 30, 400]
        pil_frames = [Image.fromarray(np.moveaxis(frame.astype(np.uint8), 0, -1)) for frame in dummy_frames_np]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "dummy_decoded_video.mp4"}},
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]

        formatted = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertIn(self.video_token, formatted)

        inputs = processor(text=[formatted], videos=[pil_frames], return_tensors="pt")
        self.assertIn(self.videos_input_name, inputs)
        self.assertIn("video_grid_thw", inputs)
        self.assertIn("input_ids", inputs)
        self.assertTrue(inputs[self.videos_input_name].shape[0] > 0)

    @require_torch
    def test_apply_chat_template_video_frame_sampling(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")
        if "video_processor" not in self.processor_class.get_attributes():
            self.skipTest("video_processor not in processor attributes")

        dummy_video_url = "test_frame_sampling.mp4"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": dummy_video_url}},
                    {"type": "text", "text": "What is shown in this video?"},
                ],
            }
        ]

        formatted = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertIn(self.video_token, formatted)

        def _make_frames(n):
            return [Image.fromarray(np.random.randint(0, 255, (30, 400, 3), dtype=np.uint8)) for _ in range(n)]

        out_few = processor(text=[formatted], videos=[_make_frames(2)], return_tensors="pt")
        self.assertIn(self.videos_input_name, out_few)
        self.assertIn("video_grid_thw", out_few)

        out_more = processor(text=[formatted], videos=[_make_frames(6)], return_tensors="pt")
        self.assertIn(self.videos_input_name, out_more)
        self.assertIn("video_grid_thw", out_more)

        t_few = out_few["video_grid_thw"][0][0].item()
        t_more = out_more["video_grid_thw"][0][0].item()
        self.assertLessEqual(t_few, t_more, "More input frames should yield a larger or equal temporal dimension")
