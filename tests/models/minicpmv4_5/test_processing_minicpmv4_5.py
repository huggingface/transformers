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

import unittest

import numpy as np
from parameterized import parameterized

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torchvision_available, is_vision_available
from transformers.video_utils import VideoMetadata

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import MiniCPMV2_5ImageProcessor, MiniCPMV4_5Processor, PreTrainedTokenizerFast

if is_torchvision_available():
    from transformers import MiniCPMV4_5VideoProcessor


@require_vision
@require_torch
@require_torchvision
class MiniCPMV4_5ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MiniCPMV4_5Processor

    @staticmethod
    def prepare_processor_dict():
        # The converted checkpoint's chat template, without the thinking and tool parts
        chat_template = (
            "{%- set ns = namespace(image_count=0) -%}"
            "{%- for message in messages -%}"
            "{%- if message['content'] is string -%}{%- set content = message['content'] -%}"
            "{%- else -%}{%- set parts = namespace(items=[]) -%}"
            "{%- for item in message['content'] -%}"
            "{%- if item['type'] == 'image' -%}"
            "{%- set parts.items = parts.items + ['<image_id>' ~ ns.image_count ~ '</image_id><unk>'] -%}"
            "{%- set ns.image_count = ns.image_count + 1 -%}"
            "{%- elif item['type'] == 'video' -%}{%- set parts.items = parts.items + ['<|video_pad|>'] -%}"
            "{%- elif item['type'] == 'text' -%}{%- set parts.items = parts.items + [item['text']] -%}{%- endif -%}"
            "{%- endfor -%}{%- set content = parts.items | join('\n') -%}{%- endif -%}"
            "{{- '<|im_start|>' + message['role'] + '\n' + content + '<|im_end|>\n' -}}"
            "{%- endfor -%}"
            "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
        )
        return {"chat_template": chat_template, "num_image_tokens": 4}

    @classmethod
    def _setup_tokenizer(cls):
        return PreTrainedTokenizerFast.from_pretrained("openbmb/MiniCPM-V-4_5")

    @classmethod
    def _setup_image_processor(cls):
        # One overview image and no slices keeps the tests fast.
        return MiniCPMV2_5ImageProcessor(scale_resolution=64, max_slice_nums=1)

    @classmethod
    def _setup_video_processor(cls):
        return MiniCPMV4_5VideoProcessor(scale_resolution=64, do_sample_frames=False)

    @property
    def video_sampling_expectations(self):
        # tiny_video_320x240.mp4 has 11 frames; every frame is one slice
        return [
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 11},
            {"do_sample_frames": True, "num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 3},
        ]

    @parameterized.expand([(1, "pt")])
    @unittest.skip("MiniCPM can't sample already decoded videos, have to turn off sampling!")
    def test_apply_chat_template_decoded_video(self, batch_size: int, return_tensors: str):
        pass

    @parameterized.expand([(1, "pt"), (2, "pt")])
    @unittest.skip("The frames of all videos are concatenated: `pixel_values_videos` has one row per frame slice")
    def test_apply_chat_template_video(self, batch_size: int, return_tensors: str):
        pass

    def test_video_placeholder(self):
        processor = self.get_processor()
        # 7 frames of a 100 s video: packed 3 per group -> groups of 3, 3 and 1 frames
        video = np.random.randint(255, size=(7, 64, 64, 3), dtype=np.uint8)
        metadata = VideoMetadata(total_num_frames=1000, fps=10.0, duration=100.0, frames_indices=list(range(7)))
        inputs = processor(text=processor.video_token, videos=[video], video_metadata=[metadata], return_tensors="pt")

        self.assertEqual(inputs["pixel_values_videos"].shape[0], 7)
        self.assertListEqual(inputs["temporal_ids"].tolist(), [[0, 1, 2], [3, 4, 5], [6, -1, -1]])
        group = f"<image>{processor.video_token * 4}</image>"
        expected = processor.tokenizer("\n".join([group] * 3))
        self.assertListEqual(inputs["input_ids"][0].tolist(), expected["input_ids"])

        # Frames twice as wide as `scale_resolution` get an overview and 1 x 2 slices each
        video = np.random.randint(255, size=(7, 64, 128, 3), dtype=np.uint8)
        inputs = processor(
            text=processor.video_token,
            videos=[video],
            video_metadata=[metadata],
            max_slice_nums=2,
            return_tensors="pt",
        )

        self.assertEqual(inputs["pixel_values_videos"].shape[0], 7 * 3)
        self.assertListEqual(inputs["temporal_ids"].tolist(), [[0, 1, 2]] * 3 + [[3, 4, 5]] * 3 + [[6, -1, -1]] * 3)
        slices = f"<slice>{processor.video_token * 4}</slice>" * 2
        expected = processor.tokenizer("\n".join([group + slices] * 3))
        self.assertListEqual(inputs["input_ids"][0].tolist(), expected["input_ids"])
