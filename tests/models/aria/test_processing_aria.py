# Copyright 2024 HuggingFace Inc.
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
from io import BytesIO

import numpy as np
import requests

from transformers import AriaProcessor
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
class AriaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = AriaProcessor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor = AriaProcessor.from_pretrained("m-ric/Aria_hf_2", size_conversion={490: 2, 980: 2})
        processor.save_pretrained(cls.tmpdirname)
        cls.image1 = Image.open(
            BytesIO(
                requests.get(
                    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
                ).content
            )
        )
        cls.image2 = Image.open(
            BytesIO(requests.get("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg").content)
        )
        cls.image3 = Image.open(
            BytesIO(
                requests.get(
                    "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"
                ).content
            )
        )
        cls.bos_token = "<|im_start|>"
        cls.eos_token = "<|im_end|>"

        cls.image_token = processor.tokenizer.image_token
        cls.fake_image_token = "o"
        cls.global_img_token = "<|img|>"

        cls.bos_token_id = processor.tokenizer.convert_tokens_to_ids(cls.bos_token)
        cls.eos_token_id = processor.tokenizer.convert_tokens_to_ids(cls.eos_token)

        cls.image_token_id = processor.tokenizer.convert_tokens_to_ids(cls.image_token)
        cls.fake_image_token_id = processor.tokenizer.convert_tokens_to_ids(cls.fake_image_token)
        cls.global_img_tokens_id = processor.tokenizer(cls.global_img_token, add_special_tokens=False)["input_ids"]
        cls.padding_token_id = processor.tokenizer.pad_token_id
        cls.image_seq_len = 2

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}{% elif message['content'] is iterable %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<fim_prefix><|img|><fim_suffix>{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
            "size_conversion": {490: 2, 980: 2},
        }  # fmt: skip

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

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

    def test_process_interleaved_images_prompts_image_splitting(self):
        processor = self.get_processor()
        processor.image_processor.split_image = True

        # Test that a single image is processed correctly
        inputs = processor(images=self.image1, text="Ok<|img|>", images_kwargs={"split_image": True})
        self.assertEqual(np.array(inputs["pixel_values"]).shape, (2, 3, 980, 980))
        self.assertEqual(np.array(inputs["pixel_mask"]).shape, (2, 980, 980))

    def test_process_interleaved_images_prompts_no_image_splitting(self):
        processor = self.get_processor()
        processor.image_processor.split_image = False

        # Test that a single image is processed correctly
        inputs = processor(images=self.image1, text="Ok<|img|>")
        image1_expected_size = (980, 980)
        self.assertEqual(np.array(inputs["pixel_values"]).shape, (1, 3, *image1_expected_size))
        self.assertEqual(np.array(inputs["pixel_mask"]).shape, (1, *image1_expected_size))
        # fmt: on

        # Test a single sample with image and text
        image_str = "<|img|>"
        text_str = "In this image, we see"
        text = image_str + text_str
        inputs = processor(text=text, images=self.image1)

        # fmt: off
        tokenized_sentence = processor.tokenizer(text_str, add_special_tokens=False)

        expected_input_ids = [[self.image_token_id] * self.image_seq_len + tokenized_sentence["input_ids"]]
        # self.assertEqual(len(inputs["input_ids"]), len(expected_input_ids))

        self.assertEqual(inputs["input_ids"], expected_input_ids)
        self.assertEqual(inputs["attention_mask"], [[1] * len(expected_input_ids[0])])
        self.assertEqual(np.array(inputs["pixel_values"]).shape, (1, 3, *image1_expected_size))
        self.assertEqual(np.array(inputs["pixel_mask"]).shape, (1, *image1_expected_size))
        # fmt: on

        # Test that batch is correctly processed
        image_str = "<|img|>"
        text_str_1 = "In this image, we see"
        text_str_2 = "In this image, we see"

        text = [
            image_str + text_str_1,
            image_str + image_str + text_str_2,
        ]
        images = [[self.image1], [self.image2, self.image3]]

        inputs = processor(text=text, images=images, padding=True)

        # fmt: off
        tokenized_sentence_1 = processor.tokenizer(text_str_1, add_special_tokens=False)
        tokenized_sentence_2 = processor.tokenizer(text_str_2, add_special_tokens=False)

        image_tokens = [self.image_token_id] * self.image_seq_len
        expected_input_ids_1 = image_tokens + tokenized_sentence_1["input_ids"]
        expected_input_ids_2 = 2 * image_tokens + tokenized_sentence_2["input_ids"]

        # Pad the first input to match the second input
        pad_len = len(expected_input_ids_2) - len(expected_input_ids_1)

        expected_attention_mask = [[0] * pad_len + [1] * len(expected_input_ids_1), [1] * (len(expected_input_ids_2))]

        self.assertEqual(
            inputs["attention_mask"],
            expected_attention_mask
        )
        self.assertEqual(np.array(inputs['pixel_values']).shape, (3, 3, 980, 980))
        self.assertEqual(np.array(inputs['pixel_mask']).shape, (3, 980, 980))
        # fmt: on

    def test_non_nested_images_with_batched_text(self):
        processor = self.get_processor()
        processor.image_processor.do_image_splitting = False

        image_str = "<|img|>"
        text_str_1 = "In this image, we see"
        text_str_2 = "In this image, we see"

        text = [
            image_str + text_str_1,
            image_str + image_str + text_str_2,
        ]
        images = [self.image1, self.image2, self.image3]

        inputs = processor(text=text, images=images, padding=True)

        self.assertEqual(np.array(inputs["pixel_values"]).shape, (3, 3, 980, 980))
        self.assertEqual(np.array(inputs["pixel_mask"]).shape, (3, 980, 980))

    def test_apply_chat_template(self):
        # Message contains content which a mix of lists with images and image urls and string
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do these images show?"},
                    {"type": "image"},
                    {"type": "image"},
                    "What do these images show?",
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "And who is that?"}]},
        ]
        processor = self.get_processor()
        # Make short sequence length to test that the fake tokens are added correctly
        rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
        print(rendered)

        expected_rendered = """<|im_start|>user
What do these images show?<fim_prefix><|img|><fim_suffix><fim_prefix><|img|><fim_suffix><|im_end|>
<|im_start|>assistant
The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.<|im_end|>
<|im_start|>user
And who is that?<|im_end|>
<|im_start|>assistant
"""
        self.assertEqual(rendered, expected_rendered)

    def test_image_chat_template_accepts_processing_kwargs(self):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                    ],
                },
            ]
        ]

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            padding="max_length",
            max_length=50,
        )
        self.assertEqual(len(formatted_prompt_tokenized[0]), 50)

        formatted_prompt_tokenized = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            truncation=True,
            max_length=5,
        )
        self.assertEqual(len(formatted_prompt_tokenized[0]), 5)

        # Now test the ability to return dict
        messages[0][0]["content"].append(
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
        )
        out_dict = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            max_image_size=980,
            return_tensors="np",
        )
        self.assertListEqual(list(out_dict[self.images_input_name].shape), [1, 3, 980, 980])

    def test_special_mm_token_truncation(self):
        """Tests that special vision tokens do not get truncated when `truncation=True` is set."""

        processor = self.get_processor()

        input_str = self.prepare_text_inputs(batch_size=2, modality="image")
        image_input = self.prepare_image_inputs(batch_size=2)

        _ = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            truncation=None,
            padding=True,
        )

        with self.assertRaises(ValueError):
            _ = processor(
                text=input_str,
                images=image_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=3,
            )
