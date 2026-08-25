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

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

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

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        tokenizer = Tokenizer(WordLevel(vocab=VOCAB, unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = Whitespace()
        return tokenizer_class(
            tokenizer_object=tokenizer,
            unk_token="<|unk|>",
            pad_token="<|finetune_right_pad|>",
            bos_token="<|begin_of_text|>",
            eos_token="<|end_of_text|>",
            # adjacent runs of these carry no whitespace, so they must split as added tokens
            additional_special_tokens=[
                "<|patch|>",
                "<|video|>",
                "<|vid_start|>",
                "<|vid_end|>",
                "<|vid_frame_separator|>",
                "<|image_start|>",
                "<|image_end|>",
            ],
        )

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(max_image_tokens=40)

    @classmethod
    def _setup_video_processor(cls):
        video_processor_class = cls._get_component_class_from_processor("video_processor")
        # `replace_video_token` needs the metadata to write one timestamp per temporal group
        return video_processor_class(max_video_frame_tokens=40, do_sample_frames=False, return_metadata=True)

    @unittest.skip("The processor consumes `video_metadata`, so its output cannot be equal to the raw one")
    def test_video_processor_defaults(self):
        pass

    def test_image_boundary_tokens(self):
        processor = self.get_processor()
        images = self.prepare_image_inputs(batch_size=2)
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
