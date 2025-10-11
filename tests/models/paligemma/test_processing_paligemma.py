# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest

from transformers import GemmaTokenizer, PaliGemmaProcessor
from transformers.testing_utils import get_tests_dir, require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import SiglipImageProcessor

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_vision
class PaliGemmaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = PaliGemmaProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        image_processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        image_processor.image_seq_length = 0  # TODO: raushan fix me in #37342
        tokenizer = GemmaTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        processor = PaliGemmaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        processor.save_pretrained(cls.tmpdirname)
        cls.image_token = processor.image_token

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    # Copied from tests.models.llava.test_processing_llava.LlavaProcessorTest.test_get_num_vision_tokens
    def test_get_num_vision_tokens(self):
        "Tests general functionality of the helper used internally in vLLM"

        processor = self.get_processor()

        output = processor._get_num_multimodal_tokens(image_sizes=[(100, 100), (300, 100), (500, 30)])
        self.assertTrue("num_image_tokens" in output)
        self.assertEqual(len(output["num_image_tokens"]), 3)

        self.assertTrue("num_image_patches" in output)
        self.assertEqual(len(output["num_image_patches"]), 3)

    @require_torch
    @require_vision
    def test_image_seq_length(self):
        input_str = "lower newer"
        image_input = self.prepare_image_inputs()
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=112, padding="max_length")
        image_processor.image_seq_length = 14
        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        inputs = processor(
            text=input_str, images=image_input, return_tensors="pt", max_length=112, padding="max_length"
        )
        self.assertEqual(len(inputs["input_ids"][0]), 112)

    @require_torch
    def test_call_with_suffix(self):
        input_str = "lower newer"
        suffix = "upper older longer string"
        image_input = self.prepare_image_inputs()
        processor = self.get_processor()
        inputs = processor(text=input_str, images=image_input, suffix=suffix)
        self.assertTrue("labels" in inputs)
        self.assertEqual(len(inputs["labels"][0]), len(inputs["input_ids"][0]))

        inputs = processor(text=input_str, images=image_input, suffix=suffix, return_tensors="pt")
        self.assertTrue("labels" in inputs)
        self.assertEqual(len(inputs["labels"][0]), len(inputs["input_ids"][0]))

    def test_text_with_image_tokens(self):
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        text_multi_images = "<image><image>Dummy text!"
        text_single_image = "<image>Dummy text!"
        text_no_image = "Dummy text!"

        image = self.prepare_image_inputs()

        out_noimage = processor(text=text_no_image, images=image, return_tensors="np")
        out_singlimage = processor(text=text_single_image, images=image, return_tensors="np")
        for k in out_noimage:
            self.assertTrue(out_noimage[k].tolist() == out_singlimage[k].tolist())

        out_multiimages = processor(text=text_multi_images, images=[image, image], return_tensors="np")
        out_noimage = processor(text=text_no_image, images=[[image, image]], return_tensors="np")

        # We can't be sure what is users intention, whether user want "one text + two images" or user forgot to add the second text
        with self.assertRaises(ValueError):
            out_noimage = processor(text=text_no_image, images=[image, image], return_tensors="np")

        for k in out_noimage:
            self.assertTrue(out_noimage[k].tolist() == out_multiimages[k].tolist())

        text_batched = ["Dummy text!", "Dummy text!"]
        text_batched_with_image = ["<image>Dummy text!", "<image>Dummy text!"]
        out_images = processor(text=text_batched_with_image, images=[image, image], return_tensors="np")
        out_noimage_nested = processor(text=text_batched, images=[[image], [image]], return_tensors="np")
        out_noimage = processor(text=text_batched, images=[image, image], return_tensors="np")
        for k in out_noimage:
            self.assertTrue(out_noimage[k].tolist() == out_images[k].tolist() == out_noimage_nested[k].tolist())
