# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from transformers import MuseGlimmerProcessor
from transformers.testing_utils import require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


VOCAB = {
    "<|begin_of_text|>": 0,
    "<|end_of_text|>": 1,
    "<|finetune_right_pad|>": 2,
    "<|unk|>": 3,
    "<|patch|>": 4,
    "<|video|>": 5,
    "<|vid_start|>": 6,
    "<|vid_end|>": 7,
    "<|vid_frame_separator|>": 8,
    "<|image_start|>": 9,
    "<|image_end|>": 10,
    "lower": 11,
    "newer": 12,
    "upper": 13,
    "older": 14,
    "longer": 15,
    "string": 16,
}


@require_vision
@require_torch
class MuseGlimmerProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MuseGlimmerProcessor
    model_id = "meta-models/Muse-Glimmer-30B"

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(max_image_tokens=40)

    @classmethod
    def _setup_video_processor(cls):
        video_processor_class = cls._get_component_class_from_processor("video_processor")
        # `replace_video_token` needs the metadata to write one timestamp per temporal group
        return video_processor_class(max_video_frame_tokens=40, do_sample_frames=False, return_metadata=True)

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 140},
            {"num_frames": None, "fps": 2, "expected_dim": 0, "output_length": 140},
            {"do_sample_frames": False, "fps": 10, "expected_dim": 0, "output_length": 840},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 840},
            {"expected_dim": 0, "output_length": 840},
        ]

    def test_image_boundary_tokens(self):
        processor = self.get_processor()
        images = self.prepare_images_inputs(batch_size=2)
        text = f"{processor.image_token}lower{processor.image_token}upper"

        inputs = processor(text=text, images=images)
        num_tokens = [int(grid.prod()) // processor.image_processor.merge_size**2 for grid in inputs.image_grid_thw]
        expanded_text = (
            processor.image_start_token
            + processor.image_token * num_tokens[0]
            + processor.image_end_token
            + "lower"
            + processor.image_start_token
            + processor.image_token * num_tokens[1]
            + processor.image_end_token
            + "upper"
        )

        self.assertEqual(inputs.input_ids[0], processor.tokenizer(expanded_text).input_ids)
        self.assertEqual(inputs.input_ids[0].count(processor.image_start_token_id), 2)
        self.assertEqual(inputs.input_ids[0].count(processor.image_end_token_id), 2)
        self.assertEqual(inputs.input_ids[0].count(processor.image_token_id), sum(num_tokens))

    @unittest.skip("Doesn't work with model's jinja templte. Let know Quentin and maybe ask Meta if needs to be fixed")
    def test_apply_chat_template_tool_calls_no_content(self):
        pass
