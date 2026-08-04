# Copyright 2026 NAVER Corp. and The HuggingFace Team. All rights reserved.
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

import re
import unittest

from transformers.testing_utils import require_av, require_torch, require_torchvision, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_vision_available():
    from transformers import GPT2TokenizerFast, HyperCLOVAXVisionV2Processor, Qwen2VLVideoProcessor

VIDEO_URL = "https://huggingface.co/datasets/hf-internal-testing/test-videos/resolve/main/tiny_video_320x240.mp4"


@require_vision
@require_torch
@require_torchvision
class HyperCLOVAXVisionV2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = HyperCLOVAXVisionV2Processor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer = GPT2TokenizerFast.from_pretrained("hf-internal-testing/tiny-random-GPT2Model")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|image_pad|>", "<|video_pad|>"]})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @classmethod
    def _setup_video_processor(cls):
        return Qwen2VLVideoProcessor(min_pixels=3136, max_pixels=12845056)

    @classmethod
    def _get_real_processor(cls):
        # `AutoProcessor` would resolve `processor_class` from the hub's `processor_config.json`,
        # which still points at `Exaone4_5_Processor` until the hub PR above lands; loading directly
        # through this class sidesteps that and pulls in the real chat_template regardless.
        return HyperCLOVAXVisionV2Processor.from_pretrained(
            "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B", revision="refs/pr/14"
        )

    @require_av
    def test_apply_chat_template_video_duration_filled_when_missing(self):
        processor = self._get_real_processor()
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "url": url_to_local_path(VIDEO_URL)},
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                }
            ]
        ]

        out_dict = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, num_frames=2
        )
        decoded = processor.tokenizer.decode(out_dict["input_ids"][0])

        self.assertNotIn("<|video_duration|>", decoded)
        self.assertRegex(decoded, r'"video_duration": \d+(\.\d+)?')
        self.assertIn(self.videos_input_name, out_dict)

    @require_av
    def test_apply_chat_template_video_duration_kept_when_provided(self):
        processor = self._get_real_processor()
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "url": url_to_local_path(VIDEO_URL), "video_duration": 3.5},
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                }
            ]
        ]

        out_dict = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, num_frames=2
        )
        decoded = processor.tokenizer.decode(out_dict["input_ids"][0])

        self.assertIn('"video_duration": 3.5', decoded)

    @require_av
    def test_apply_chat_template_video_duration_mixed(self):
        processor = self._get_real_processor()
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "url": url_to_local_path(VIDEO_URL)},
                        {"type": "video", "url": url_to_local_path(VIDEO_URL), "video_duration": 9.99},
                        {"type": "text", "text": "Compare these videos."},
                    ],
                }
            ]
        ]

        out_dict = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, num_frames=2
        )
        decoded = processor.tokenizer.decode(out_dict["input_ids"][0])

        durations = re.findall(r'"video_duration": (\d+(?:\.\d+)?)', decoded)
        self.assertEqual(len(durations), 2)
        self.assertNotEqual(durations[0], "<|video_duration|>")
        self.assertEqual(durations[1], "9.99")
