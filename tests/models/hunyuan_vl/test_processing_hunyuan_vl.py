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

from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from transformers import PreTrainedTokenizerFast
from transformers.models.hunyuan_vl.image_processing_hunyuan_vl import HunYuanVLImageProcessor
from transformers.models.hunyuan_vl.processing_hunyuan_vl import HunYuanVLProcessor
from transformers.testing_utils import require_torch


class HunYuanVLProcessorTest(unittest.TestCase):
    def get_tokenizer(self):
        vocab = {
            "<unk>": 0,
            "<pad>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<image_start>": 4,
            "<image>": 5,
            "<image_end>": 6,
            "hello": 7,
            "<placeholder>": 8,
        }
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            additional_special_tokens=["<image_start>", "<image>", "<image_end>", "<placeholder>"],
        )
        fast_tokenizer.image_start_token = "<image_start>"
        fast_tokenizer.image_token = "<image>"
        fast_tokenizer.image_end_token = "<image_end>"
        return fast_tokenizer

    def get_processor(self):
        image_processor = HunYuanVLImageProcessor(
            min_pixels=32 * 32,
            max_pixels=32 * 32,
            patch_size=16,
            temporal_patch_size=1,
            merge_size=1,
        )
        return HunYuanVLProcessor(image_processor=image_processor, tokenizer=self.get_tokenizer())

    @require_torch
    def test_processor_outputs_image_only_inputs(self):
        processor = self.get_processor()
        image = Image.new("RGB", (32, 32), color="white")

        inputs = processor(text=["<image> hello"], images=[image], padding=True, return_tensors="pt")

        self.assertSetEqual(
            set(inputs.keys()),
            {"input_ids", "attention_mask", "position_ids", "imgs_pos", "pixel_values", "image_grid_thw"},
        )
        self.assertEqual(inputs["position_ids"].shape[1], 4)
        self.assertGreater(inputs["pixel_values"].shape[0], 0)
        self.assertEqual(inputs["image_grid_thw"].shape[-1], 3)

    def test_get_num_multimodal_tokens(self):
        processor = self.get_processor()
        output = processor._get_num_multimodal_tokens(image_sizes=[(32, 32)])

        self.assertEqual(len(output["num_image_tokens"]), 1)
        self.assertEqual(len(output["num_image_patches"]), 1)
        self.assertGreater(output["num_image_tokens"][0], 0)

    def test_model_input_names(self):
        processor = self.get_processor()
        self.assertListEqual(
            processor.model_input_names,
            ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "position_ids", "imgs_pos"],
        )
