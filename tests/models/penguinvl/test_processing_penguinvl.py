# Copyright 2025 Tencent and The HuggingFace Team. All rights reserved.
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

import numpy as np
from PIL import Image

from transformers.testing_utils import require_torch, require_vision, slow
from transformers.utils import is_torch_available, is_vision_available


if is_vision_available():
    from transformers import PenguinVLImageProcessor
    from transformers import PenguinVLProcessor
    from transformers.models.penguinvl.image_processing_penguinvl import _make_batched_clips

if is_torch_available():
    import torch


def _make_dummy_pil_image(width=224, height=224, mode="RGB"):
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@require_vision
@require_torch
class MakeBatchedClipsTest(unittest.TestCase):
    """Unit tests for the _make_batched_clips helper function."""

    def test_single_image(self):
        img = _make_dummy_pil_image()
        result = _make_batched_clips(img)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 1)
        self.assertIs(result[0][0], img)

    def test_list_of_images(self):
        images = [_make_dummy_pil_image() for _ in range(3)]
        result = _make_batched_clips(images)
        self.assertEqual(len(result), 3)
        for i, clip in enumerate(result):
            self.assertEqual(len(clip), 1)
            self.assertIs(clip[0], images[i])

    def test_nested_clips(self):
        img1 = _make_dummy_pil_image()
        frames = [_make_dummy_pil_image() for _ in range(4)]
        nested = [[img1], frames]
        result = _make_batched_clips(nested)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 4)


@require_vision
@require_torch
class PenguinVLImageProcessorTest(unittest.TestCase):
    """Tests for PenguinVLImageProcessor with image/video/multi-image inputs."""

    def setUp(self):
        self.image_processor = PenguinVLImageProcessor(
            min_pixels=28 * 28,
            max_pixels=56 * 56 * 4,
            patch_size=14,
            merge_size=1,
        )

    def test_single_image_output_keys(self):
        img = _make_dummy_pil_image(224, 224)
        out = self.image_processor(img, return_tensors="pt")
        self.assertIn("pixel_values", out)
        self.assertIn("image_grid_thw", out)
        self.assertIn("image_merge_sizes", out)
        self.assertIn("num_frames_per_clip", out)

    def test_single_image_shapes(self):
        img = _make_dummy_pil_image(224, 224)
        out = self.image_processor(img, return_tensors="pt")
        # pixel_values: [num_patches, C*P^2]
        self.assertEqual(out.pixel_values.ndim, 2)
        # image_grid_thw: [1, 3] — one entry for the single image
        self.assertEqual(out.image_grid_thw.shape[0], 1)
        self.assertEqual(out.image_grid_thw.shape[1], 3)
        # merge_sizes: [1]
        self.assertEqual(out.image_merge_sizes.shape[0], 1)

    def test_multi_image_output_shapes(self):
        images = [_make_dummy_pil_image(224, 224) for _ in range(3)]
        out = self.image_processor(images, merge_size=1, return_tensors="pt")
        # 3 images → 3 entries in grid_thw
        self.assertEqual(out.image_grid_thw.shape[0], 3)
        self.assertEqual(len(out.num_frames_per_clip), 3)
        for n in out.num_frames_per_clip:
            self.assertEqual(n, 1)

    def test_video_clip_output_shapes(self):
        frames = [_make_dummy_pil_image(112, 112) for _ in range(4)]
        video_clip = [frames]  # wrap in outer list to form one clip
        out = self.image_processor(video_clip, merge_size=2, return_tensors="pt")
        # 4 frames → 4 entries in grid_thw
        self.assertEqual(out.image_grid_thw.shape[0], 4)
        # All frames should have merge_size=2
        self.assertTrue((out.image_merge_sizes == 2).all())
        # 1 clip
        self.assertEqual(len(out.num_frames_per_clip), 1)
        self.assertEqual(out.num_frames_per_clip[0], 4)

    def test_mixed_image_and_video(self):
        """Test nested input: [[single_image], [frame1, frame2, frame3]]."""
        img = _make_dummy_pil_image(112, 112)
        frames = [_make_dummy_pil_image(112, 112) for _ in range(3)]
        nested = [[img], frames]
        out = self.image_processor(nested, merge_size=[1, 2], return_tensors="pt")
        # 1 + 3 = 4 total frame entries
        self.assertEqual(out.image_grid_thw.shape[0], 4)
        self.assertEqual(len(out.num_frames_per_clip), 2)
        self.assertEqual(out.num_frames_per_clip[0], 1)
        self.assertEqual(out.num_frames_per_clip[1], 3)
        # First frame: merge_size=1, rest: merge_size=2
        self.assertEqual(int(out.image_merge_sizes[0]), 1)
        self.assertTrue((out.image_merge_sizes[1:] == 2).all())

    def test_frame_types_change_resolution(self):
        """Key frames should have same or higher resolution than intermediate frames."""
        frames = [_make_dummy_pil_image(112, 112) for _ in range(4)]
        video_clip = [frames]
        frame_types = [[0, 1, 0, 1]]  # 0=keyframe, 1=intermediate

        out = self.image_processor(video_clip, merge_size=2, frame_types=frame_types, return_tensors="pt")
        grids = out.image_grid_thw  # [4, 3]

        key_area = int(grids[0][1]) * int(grids[0][2])
        inter_area = int(grids[1][1]) * int(grids[1][2])
        self.assertGreaterEqual(key_area, inter_area)

    def test_different_sized_images(self):
        """Test that images of different sizes are handled correctly."""
        images = [
            _make_dummy_pil_image(112, 112),
            _make_dummy_pil_image(224, 112),
            _make_dummy_pil_image(56, 168),
        ]
        out = self.image_processor(images, return_tensors="pt")
        # Should succeed with 3 entries
        self.assertEqual(out.image_grid_thw.shape[0], 3)

    def test_return_tensors_pt(self):
        img = _make_dummy_pil_image(112, 112)
        out = self.image_processor(img, return_tensors="pt")
        self.assertIsInstance(out.pixel_values, torch.Tensor)
        self.assertIsInstance(out.image_grid_thw, torch.Tensor)

    def test_return_tensors_np(self):
        img = _make_dummy_pil_image(112, 112)
        out = self.image_processor(img, return_tensors="np")
        self.assertIsInstance(out.pixel_values, np.ndarray)


@require_vision
@require_torch
class PenguinVLProcessorUnitTest(unittest.TestCase):
    """
    Unit tests for PenguinVLProcessor that do not require a pre-trained tokenizer.
    These tests verify the image token expansion logic and process_vision_info.
    """

    @classmethod
    def setUpClass(cls):
        """Try to load a PenguinVL tokenizer for testing; skip if unavailable."""
        try:
            from transformers import AutoTokenizer

            cls.tokenizer = AutoTokenizer.from_pretrained("tencent/Penguin-VL-8B", trust_remote_code=True)
        except Exception:
            cls.tokenizer = None

    def _make_processor(self, min_pixels=28 * 28, max_pixels=56 * 56 * 4):
        if self.tokenizer is None:
            self.skipTest("PenguinVL tokenizer not available (requires network access)")
        return PenguinVLProcessor.from_pretrained("tencent/Penguin-VL-8B", trust_remote_code=True)

    def test_processor_attributes(self):
        processor = self._make_processor()
        self.assertTrue(hasattr(processor, "image_processor"))
        self.assertTrue(hasattr(processor, "tokenizer"))
        self.assertEqual(processor.image_token, "<image>")
        self.assertEqual(processor.image_merge_size, 1)
        self.assertEqual(processor.video_merge_size, 2)

    def test_processor_model_input_names(self):
        processor = self._make_processor()
        input_names = processor.model_input_names
        self.assertIn("input_ids", input_names)
        self.assertIn("pixel_values", input_names)

    def test_process_vision_info_single_image(self):
        processor = self._make_processor()
        img = _make_dummy_pil_image(112, 112)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        images, frame_types = processor.process_vision_info(messages)
        self.assertIsNotNone(images)
        self.assertEqual(len(images), 1)
        self.assertEqual(len(images[0]), 1)
        self.assertIsNone(frame_types[0])  # images have None frame_types

    def test_process_vision_info_multi_image(self):
        processor = self._make_processor()
        img1 = _make_dummy_pil_image(112, 112)
        img2 = _make_dummy_pil_image(224, 224)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": "Compare these images."},
                ],
            }
        ]
        images, frame_types = processor.process_vision_info(messages)
        self.assertEqual(len(images), 2)
        self.assertIsNone(frame_types[0])
        self.assertIsNone(frame_types[1])

    def test_process_vision_info_video_frames(self):
        processor = self._make_processor()
        frames = [_make_dummy_pil_image(112, 112) for _ in range(4)]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]
        images, frame_types = processor.process_vision_info(messages)
        self.assertEqual(len(images), 1)
        self.assertEqual(len(images[0]), 4)  # 4 frames in the clip
        self.assertIsNotNone(frame_types[0])  # videos have frame_types
        self.assertEqual(len(frame_types[0]), 4)
        # First frame is always a keyframe (0)
        self.assertEqual(frame_types[0][0], 0)

    def test_process_vision_info_no_visuals(self):
        processor = self._make_processor()
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        images, frame_types = processor.process_vision_info(messages)
        self.assertIsNone(images)
        self.assertIsNone(frame_types)

    def test_processor_single_image_call(self):
        processor = self._make_processor()
        img = _make_dummy_pil_image(112, 112)

        # Get the number of image tokens for this image
        ip_out = processor.image_processor(img, return_tensors="pt")
        thw = ip_out.image_grid_thw[0]
        ms = int(ip_out.image_merge_sizes[0])
        expected_tokens = int(thw[0]) * int(thw[1] // ms) * int(thw[2] // ms)

        text = "<image>"
        out = processor(images=img, text=text, return_tensors="pt")
        self.assertIn("input_ids", out)
        self.assertIn("pixel_values", out)
        self.assertIn("image_grid_thw", out)

        # Count image tokens in input_ids
        image_token_id = processor.image_token_id
        n_image_tokens = (out.input_ids == image_token_id).sum().item()
        self.assertEqual(n_image_tokens, expected_tokens)

    def test_processor_multi_image_call(self):
        processor = self._make_processor()
        images = [_make_dummy_pil_image(112, 112), _make_dummy_pil_image(56, 56)]
        # Two image tokens in text, one per image
        text = "<image><image>"

        out = processor(images=images, text=text, return_tensors="pt")
        self.assertIn("input_ids", out)
        self.assertIn("pixel_values", out)

        # image_grid_thw should have 2 entries (one per image)
        self.assertEqual(out.image_grid_thw.shape[0], 2)

    def test_processor_video_call(self):
        processor = self._make_processor()
        frames = [_make_dummy_pil_image(112, 112) for _ in range(3)]
        # A video clip as a list of frames
        video_clip = [frames]
        text = "<image>"

        out = processor(images=video_clip, text=text, return_tensors="pt")
        self.assertIn("input_ids", out)
        self.assertIn("pixel_values", out)
        # Should have 3 frame entries in image_grid_thw
        self.assertEqual(out.image_grid_thw.shape[0], 3)

    def test_processor_batch_call(self):
        processor = self._make_processor()
        img1 = _make_dummy_pil_image(112, 112)
        img2 = _make_dummy_pil_image(224, 224)

        out = processor(
            images=[img1, img2],
            text=["<image>", "<image>"],
            padding=True,
            return_tensors="pt",
        )
        self.assertEqual(out.input_ids.shape[0], 2)
        self.assertEqual(out.image_grid_thw.shape[0], 2)

    def test_apply_chat_template(self):
        processor = self._make_processor()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        EXPECTED_TEXT = "<|im_start|>user\n<image>\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.assertEqual(text, EXPECTED_TEXT)

    def test_convert_messages_for_chat_template_image(self):
        processor = self._make_processor()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://example.com/img.jpg"},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ]
        converted = processor._convert_messages_for_chat_template(messages)
        content = converted[0]["content"]
        image_items = [c for c in content if c.get("type") == "image"]
        self.assertEqual(len(image_items), 1)
        # URL should be stripped
        self.assertEqual(image_items[0], {"type": "image"})

    def test_convert_messages_for_chat_template_video_with_num_frames(self):
        processor = self._make_processor()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "https://example.com/vid.mp4", "num_frames": 4, "timestamps": [0, 1, 2, 3]},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ]
        converted = processor._convert_messages_for_chat_template(messages)
        content = converted[0]["content"]
        video_items = [c for c in content if c.get("type") == "video"]
        self.assertEqual(len(video_items), 1)
        self.assertEqual(video_items[0]["num_frames"], 4)
        self.assertEqual(video_items[0]["timestamps"], [0, 1, 2, 3])

    def test_convert_messages_for_chat_template_video_without_num_frames(self):
        """Video items without num_frames should fall back to plain image."""
        processor = self._make_processor()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "https://example.com/vid.mp4"},
                    {"type": "text", "text": "Describe."},
                ],
            }
        ]
        converted = processor._convert_messages_for_chat_template(messages)
        content = converted[0]["content"]
        # Without num_frames, falls back to image type
        self.assertEqual(content[0], {"type": "image"})

    def test_batch_decode(self):
        processor = self._make_processor()
        # Just check batch_decode delegates to tokenizer
        token_ids = [[1, 2, 3], [4, 5, 6]]
        result = processor.batch_decode(token_ids, skip_special_tokens=True)
        EXPECTED_TEXT = ['"#$', "%&'"]
        self.assertEqual(result, EXPECTED_TEXT)

    def test_decode(self):
        processor = self._make_processor()
        token_ids = [1, 2, 3]
        result = processor.decode(token_ids, skip_special_tokens=True)
        EXPECTED_TEXT = '"#$'
        self.assertEqual(result, EXPECTED_TEXT)


@require_vision
@require_torch
@slow
class PenguinVLProcessorIntegrationTest(unittest.TestCase):
    """
    Integration tests for PenguinVLProcessor using the real PenguinVL model.
    These tests require network access and the actual model checkpoint.
    """

    model_id = "tencent/Penguin-VL-8B"

    @classmethod
    def setUpClass(cls):
        from transformers import PenguinVLProcessor

        cls.processor = PenguinVLProcessor.from_pretrained(cls.model_id)

    def _make_image(self, width=224, height=224):
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_process_single_image(self):
        img = self._make_image()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What do you see?"},
                ],
            }
        ]
        images, frame_types = self.processor.process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        out = self.processor(images=images, text=text, frame_types=frame_types, return_tensors="pt")

        self.assertIn("input_ids", out)
        self.assertIn("pixel_values", out)
        self.assertIn("image_grid_thw", out)
        self.assertEqual(out.image_grid_thw.shape[0], 1)

    def test_process_multi_image(self):
        img1 = self._make_image(224, 224)
        img2 = self._make_image(336, 224)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img1},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": "Are these the same?"},
                ],
            }
        ]
        images, frame_types = self.processor.process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        out = self.processor(images=images, text=text, frame_types=frame_types, return_tensors="pt")

        # 2 images → 2 entries in image_grid_thw
        self.assertEqual(out.image_grid_thw.shape[0], 2)

    def test_process_video_frames(self):
        frames = [self._make_image(112, 112) for _ in range(6)]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": "What happens in this video?"},
                ],
            }
        ]
        images, frame_types = self.processor.process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        out = self.processor(images=images, text=text, frame_types=frame_types, return_tensors="pt")

        # 6 video frames → 6 entries in image_grid_thw
        self.assertEqual(out.image_grid_thw.shape[0], 6)
        # Video uses video_merge_size=2
        self.assertTrue((out.image_merge_sizes == 2).all())

    def test_process_mixed_image_and_video(self):
        """Test mixed image + video in the same message."""
        img = self._make_image(224, 224)
        frames = [self._make_image(112, 112) for _ in range(3)]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "video", "video": frames},
                    {"type": "text", "text": "Describe both."},
                ],
            }
        ]
        images, frame_types = self.processor.process_vision_info(messages)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        out = self.processor(images=images, text=text, frame_types=frame_types, return_tensors="pt")

        # 1 image + 3 video frames = 4 entries
        self.assertEqual(out.image_grid_thw.shape[0], 4)
        # Image merge_size=1, video frames merge_size=2
        self.assertEqual(int(out.image_merge_sizes[0]), 1)
        self.assertTrue((out.image_merge_sizes[1:] == 2).all())

    def test_batch_processing(self):
        img1 = self._make_image(112, 112)
        img2 = self._make_image(224, 224)
        messages1 = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img1}, {"type": "text", "text": "Describe."}],
            }
        ]
        messages2 = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img2}, {"type": "text", "text": "What is this?"}],
            }
        ]
        images1, ft1 = self.processor.process_vision_info(messages1)
        images2, ft2 = self.processor.process_vision_info(messages2)
        text1 = self.processor.apply_chat_template(messages1, add_generation_prompt=True)
        text2 = self.processor.apply_chat_template(messages2, add_generation_prompt=True)

        all_images = images1 + images2
        all_fts = ft1 + ft2 if ft1 and ft2 else None
        out = self.processor(
            images=all_images,
            text=[text1, text2],
            frame_types=all_fts,
            padding=True,
            return_tensors="pt",
        )
        self.assertEqual(out.input_ids.shape[0], 2)
