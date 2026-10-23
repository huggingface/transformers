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
import tempfile
import unittest

from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoProcessor,
    AutoVideoProcessor,
    Dots3NoteFeatureExtractor,
    Dots3NoteProcessor,
    Qwen2VLImageProcessor,
    Qwen2VLImageProcessorPil,
    Qwen3VLVideoProcessor,
    is_torch_available,
)
from transformers.testing_utils import require_torch, require_torchvision

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


AUDIO_START = "<|audio_comp_start|>"
AUDIO_END = "<|audio_comp_end|>"


@require_torch
@require_torchvision
class Dots3NoteProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Dots3NoteProcessor
    tiny_model_id = "miraclezqc/TestForDots3"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        # Keep the common tests' short audio clips in one chunk; chunking has dedicated feature extractor tests.
        return super()._setup_from_pretrained(
            model_id, revision="14258a1269bdeba3b9d1ef69ea72673470464300", chunk_length=30, **kwargs
        )

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 12},
            {"fps": 10, "expected_dim": 0, "output_length": 16},
            {"do_sample_frames": False, "fps": 10, "expected_dim": 0, "output_length": 44},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 44},
            {"expected_dim": 0, "output_length": 16},
        ]

    def test_component_config_roundtrip(self):
        processor = Dots3NoteProcessor(
            image_processor=Qwen2VLImageProcessorPil(
                size={"shortest_edge": 3136, "longest_edge": 1016064}, temporal_patch_size=1
            ),
            tokenizer=self.get_component("tokenizer"),
            video_processor=Qwen3VLVideoProcessor(
                size={"shortest_edge": 3136, "longest_edge": 1016064},
                temporal_patch_size=1,
                patch_size=14,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
                fps=1,
            ),
            feature_extractor=Dots3NoteFeatureExtractor(),
        )

        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            processor = AutoProcessor.from_pretrained(directory)
            image_processor = AutoImageProcessor.from_pretrained(directory)
            video_processor = AutoVideoProcessor.from_pretrained(directory)
            feature_extractor = AutoFeatureExtractor.from_pretrained(directory)
        expected_size = {"shortest_edge": 3136, "longest_edge": 1016064}
        for component in (processor.image_processor, image_processor):
            self.assertIsInstance(component, (Qwen2VLImageProcessorPil, Qwen2VLImageProcessor))
        for component in (processor.video_processor, video_processor):
            self.assertIsInstance(component, Qwen3VLVideoProcessor)
        for component in (processor.image_processor, image_processor, processor.video_processor, video_processor):
            self.assertEqual(dict(component.size), expected_size)
            self.assertEqual(component.temporal_patch_size, 1)
        for component in (processor.feature_extractor, feature_extractor):
            self.assertIsInstance(component, Dots3NoteFeatureExtractor)
            self.assertEqual(component.sampling_rate, 16000)
            self.assertEqual(component.feature_size, 128)

    def test_expands_audio_placeholder(self):
        processor = self.get_processor()
        output = processor(
            text=f"{AUDIO_START}{processor.audio_token}{AUDIO_END} describe",
            audio=[torch.zeros(1281)],
            sampling_rate=16000,
            add_special_tokens=False,
            return_tensors="pt",
        )

        input_ids = output.input_ids[0].tolist()
        self.assertEqual(input_ids.count(processor.tokenizer.convert_tokens_to_ids(AUDIO_START)), 1)
        self.assertEqual(input_ids.count(processor.audio_token_id), 2)
        self.assertEqual(input_ids.count(processor.tokenizer.convert_tokens_to_ids(AUDIO_END)), 1)
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
