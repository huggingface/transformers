# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import json
import os
import unittest

import pytest

from transformers.models.clip.tokenization_clip import VOCAB_FILES_NAMES
from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import CLIPSegProcessor


@require_vision
class CLIPSegProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = CLIPSegProcessor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        vocab = ["l", "o", "w", "e", "r", "s", "t", "i", "d", "n", "lo", "l</w>", "w</w>", "r</w>", "t</w>", "low</w>", "er</w>", "lowest</w>", "newer</w>", "wider", "<unk>", "<|startoftext|>", "<|endoftext|>"]  # fmt: skip
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "l o", "lo w</w>", "e r</w>", ""]

        vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        merges_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))
        return tokenizer_class.from_pretrained(cls.tmpdirname)

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        image_processor_map = {
            "do_resize": True,
            "size": 20,
            "do_center_crop": True,
            "crop_size": 18,
            "do_normalize": True,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
        }
        return image_processor_class(**image_processor_map)

    def test_processor_text(self):
        processor = self.get_processor()

        input_str = "lower newer"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(list(inputs.keys()), ["input_ids", "attention_mask", "pixel_values"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_processor_visual_prompt(self):
        processor = self.get_processor()

        image_input = self.prepare_image_inputs()
        visual_prompt_input = self.prepare_image_inputs()

        inputs = processor(images=image_input, visual_prompt=visual_prompt_input)

        self.assertListEqual(list(inputs.keys()), ["pixel_values", "conditional_pixel_values"])

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()
