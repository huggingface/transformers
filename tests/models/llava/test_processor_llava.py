# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import AutoProcessor, AutoTokenizer, CLIPImageProcessor, LlamaTokenizerFast, LlavaProcessor


@require_vision
class LlavaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = LlavaProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = CLIPImageProcessor(do_center_crop=False)
        tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b")

        processor = LlavaProcessor(image_processor, tokenizer)

        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_can_load_various_tokenizers(self):
        for checkpoint in ["Intel/llava-gemma-2b", "llava-hf/llava-1.5-7b-hf"]:
            processor = LlavaProcessor.from_pretrained(checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    def test_chat_template(self):
        processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        expected_prompt = "USER: <image>\nWhat is shown in this image? ASSISTANT:"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = ["lower newer", "upper older longer string"]
        image_input = self.prepare_image_inputs() * 2
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            size={"height": 214, "width": 214},
            padding="longest",
            max_length=76,
        )

        self.assertEqual(inputs["pixel_values"].shape[2], 214)

        self.assertEqual(len(inputs["input_ids"][0]), 5)
