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
    "lower": 9,
    "newer": 10,
    "upper": 11,
    "older": 12,
    "longer": 13,
    "string": 14,
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
