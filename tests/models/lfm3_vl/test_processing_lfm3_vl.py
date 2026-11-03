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
"""Testing suite for the LFM3-VL processor."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_tokenizers_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Lfm3VlProcessor, Lfm3VlVideoProcessor, Qwen2VLImageProcessor

if is_tokenizers_available():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = {
    "pad_token": "<|pad|>",
    "bos_token": "<|startoftext|>",
    "eos_token": "<|im_end|>",
    "unk_token": "<unk>",
}
VISION_TOKENS = {
    "image_token": "<|image|>",
    "video_token": "<|video|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
}
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' %}<|vision_start|><|image|><|vision_end|>"
    "{% elif content['type'] == 'video' %}<|vision_start|><|video|><|vision_end|>"
    "{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}"
    "{% endfor %}{% endif %}"
    "<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@require_vision
@require_torch
@require_torchvision
class Lfm3VlProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Lfm3VlProcessor
    images_text_kwargs_max_length = 30
    images_text_kwargs_override_max_length = 25
    images_unstructured_max_length = 40
    videos_text_kwargs_max_length = 120
    videos_text_kwargs_override_max_length = 110
    videos_unstructured_max_length = 130

    @classmethod
    def _setup_tokenizer(cls):
        """A word-level tokenizer carrying the LFM3-VL vision tokens as tokenizer attributes, like the released one."""
        words = ["▁", "▁lower", "▁newer", "▁upper", "▁older", "▁longer", "▁string", "<|im_start|>", "▁seconds"]
        vocabulary = {token: index for index, token in enumerate([*SPECIAL_TOKENS.values(), *VISION_TOKENS.values()])}
        for word in words:
            vocabulary[word] = len(vocabulary)
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Metaspace()
        backend.decoder = decoders.Metaspace()
        return PreTrainedTokenizerFast(tokenizer_object=backend, extra_special_tokens=VISION_TOKENS, **SPECIAL_TOKENS)

    @classmethod
    def _setup_image_processor(cls):
        # the released patch, merge and normalization values with small pixel budgets
        return Qwen2VLImageProcessor(
            size={"shortest_edge": 32 * 32, "longest_edge": 64 * 64},
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )

    @classmethod
    def _setup_video_processor(cls):
        return Lfm3VlVideoProcessor(size={"shortest_edge": 32 * 32, "longest_edge": 64 * 64 * 4})

    @staticmethod
    def prepare_processor_dict():
        return {"chat_template": CHAT_TEMPLATE}

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "expected_dim": 0, "output_length": 32},
            {"num_frames": None, "fps": 18, "expected_dim": 0, "output_length": 12},
            {"do_sample_frames": False, "fps": 2, "expected_dim": 0, "output_length": 24},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 24},
            {"expected_dim": 0, "output_length": 16},
        ]

    def test_processor_uses_the_tokenizer_vision_tokens(self):
        processor = self.get_processor()
        tokenizer = processor.tokenizer
        self.assertEqual(processor.image_token, "<|image|>")
        self.assertEqual(processor.video_token, "<|video|>")
        self.assertEqual(processor.image_token_id, tokenizer.convert_tokens_to_ids("<|image|>"))
        self.assertEqual(processor.video_token_id, tokenizer.convert_tokens_to_ids("<|video|>"))
        self.assertEqual(processor.vision_start_token_id, tokenizer.convert_tokens_to_ids("<|vision_start|>"))
        self.assertEqual(processor.vision_end_token_id, tokenizer.convert_tokens_to_ids("<|vision_end|>"))
        self.assertNotIn("mm_token_type_ids", processor.model_input_names)

    def test_image_placeholders_match_the_patch_grid(self):
        processor = self.get_processor()
        images = self.prepare_images_inputs(batch_size=2)
        inputs = processor(text=self.prepare_text_inputs(batch_size=2, modalities="image"), images=images)

        merge_length = processor.image_processor.merge_size**2
        for input_ids, grid in zip(inputs["input_ids"], inputs["image_grid_thw"]):
            self.assertEqual(input_ids.count(processor.image_token_id), math.prod(grid) // merge_length)
        self.assertEqual(inputs["pixel_values"].shape[0], sum(math.prod(grid) for grid in inputs["image_grid_thw"]))
        self.assertNotIn("mm_token_type_ids", inputs)

    def test_video_placeholders_carry_frame_timestamps(self):
        processor = self.get_processor()
        video = self.prepare_videos_inputs(batch_size=1)
        inputs = processor(text=self.prepare_text_inputs(batch_size=1, modalities="video"), videos=video)

        grid = inputs["video_grid_thw"][0]
        tokens_per_frame = int(grid[1] * grid[2]) // processor.video_processor.merge_size**2
        input_ids = inputs["input_ids"][0]
        self.assertEqual(input_ids.count(processor.video_token_id), int(grid[0]) * tokens_per_frame)
        # every temporal patch is wrapped in vision start/end tokens and preceded by a timestamp
        self.assertEqual(input_ids.count(processor.vision_start_token_id), int(grid[0]))
        self.assertEqual(input_ids.count(processor.vision_end_token_id), int(grid[0]))

    def test_video_frames_are_prefixed_with_their_timestamp(self):
        processor = self.get_processor()
        video_inputs = {
            "video_grid_thw": np.asarray([[2, 4, 4]]),
            "video_metadata": [SimpleNamespace(frames_indices=[0, 2, 4, 6], fps=2)],
        }
        placeholder = processor.replace_video_token(video_inputs, video_idx=0)
        frame = "<|vision_start|>" + "<|video|>" * 4 + "<|vision_end|>"
        self.assertEqual(placeholder, "<0.5 seconds>" + frame + "<2.5 seconds>" + frame)

    def test_video_metadata_is_returned_for_both_kwarg_forms(self):
        processor = self.get_processor()
        video = self.prepare_videos_inputs(batch_size=1)
        for kwargs in ({"return_metadata": True}, {"videos_kwargs": {"return_metadata": True}}):
            with self.subTest(kwargs=kwargs):
                inputs = processor(text="<|video|>", videos=video, **kwargs)
                self.assertEqual(len(inputs["video_metadata"]), 1)
        self.assertNotIn("video_metadata", processor(text="<|video|>", videos=video))

    def test_get_num_multimodal_tokens(self):
        processor = self.get_processor()
        image_sizes = [(100, 100), (300, 100), (500, 30)]
        output = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        self.assertEqual(len(output["num_image_tokens"]), 3)
        self.assertEqual(len(output["num_image_patches"]), 3)
        for num_tokens, num_patches in zip(output["num_image_tokens"], output["num_image_patches"]):
            self.assertEqual(num_tokens, num_patches // processor.image_processor.merge_size**2)

    @unittest.skip("The word-level test tokenizer cannot round-trip the frame timestamps")
    def test_replacement_offsets(self):
        pass

    @unittest.skip("LFM3-VL cannot sample frames from decoded frames without their metadata, like Qwen3-VL")
    def test_apply_chat_template_video_2(self):
        pass
