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

from transformers import (
    Dots3NoteProcessor,
    is_torch_available,
)
from transformers.testing_utils import require_torch, require_torchvision

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


@require_torch
@require_torchvision
class Dots3NoteProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Dots3NoteProcessor
    tiny_model_id = "hf-internal-testing/dummy-dots3"

    @classmethod
    def _setup_test_attributes(cls, processor):
        super()._setup_test_attributes(processor)
        messages = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "describe"}]}]
        cls.audio_prompt = processor.apply_chat_template(messages, tokenize=False)
        input_ids = processor.tokenizer(cls.audio_prompt, add_special_tokens=False).input_ids
        audio_index = input_ids.index(processor.audio_token_id)
        cls.audio_start_id = input_ids[audio_index - 1]
        cls.audio_end_id = input_ids[audio_index + 1]

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        # Keep the common tests' short audio clips in one chunk; chunking has dedicated feature extractor tests.
        return super()._setup_from_pretrained(model_id, chunk_length=30, **kwargs)

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 12},
            {"fps": 10, "expected_dim": 0, "output_length": 16},
            {"do_sample_frames": False, "fps": 10, "expected_dim": 0, "output_length": 44},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 44},
            {"expected_dim": 0, "output_length": 16},
        ]

    def test_expands_audio_placeholder(self):
        processor = self.get_processor()
        output = processor(
            text=self.audio_prompt,
            audio=[torch.zeros(1281)],
            sampling_rate=16000,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = output.input_ids[0].tolist()
        self.assertEqual(input_ids.count(self.audio_start_id), 1)
        self.assertEqual(input_ids.count(processor.audio_token_id), 2)
        self.assertEqual(input_ids.count(self.audio_end_id), 1)
        self.assertEqual(output.chunk_sample_lengths.tolist(), [1281])
        self.assertNotIn("num_audio_tokens", output)

    def test_apply_chat_template_tool_calls_no_content(self):
        # The Dots template requires tool arguments as dictionaries, as documented in chat_extras.md.
        processor = self.get_processor()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "What is the weather?"}]},
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {}}}],
            },
        ]
        result = processor.apply_chat_template(messages, tokenize=True)
        self.assertIsInstance(result, list)
