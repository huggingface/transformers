# coding=utf-8
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
from typing import Optional

import numpy as np
import requests

from transformers import Phi3VProcessor
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
class Phi3VProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Phi3VProcessor
    videos_input_name = "pixel_values"

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor = Phi3VProcessor.from_pretrained("microsoft/Phi-3.5-vision-instruct", num_crops=4)
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
        cls.bos_token = processor.tokenizer.bos_token

        cls.bos_token_id = processor.tokenizer.convert_tokens_to_ids(cls.bos_token)
        cls.padding_token_id = processor.tokenizer.pad_token_id

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def prepare_processor_dict(self):
        return {
            "chat_template": "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}",
        }

    def get_expected_input_ids(self, processor, image):
        num_crops = processor.image_processor.num_crops
        imgsize = list(image.size)
        imgsize.sort()
        height, width = imgsize
        ratio = width / height
        scale = 1
        while scale * np.ceil(scale / ratio) <= num_crops:
            scale += 1
        scale -= 1
        num_img_tokens = (scale * scale + 1) * 144 + 1 + (scale + 1) * 12
        img_tokens = [-1 for num in range(num_img_tokens)]
        img_tokens.append(1)
        return img_tokens

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname)

    def test_image_token_count(self):
        processor = self.get_processor()

        image_str = "<|image_1|>"
        text_str = "In this image, we see\n"
        text = text_str + image_str

        inputs = processor(text=text, images=self.image1, return_tensors='pt')
        tokenized_sentence = processor.tokenizer(text_str)
        split_image1_tokens = self.get_expected_input_ids(processor, self.image1)
        expected_input_ids = [tokenized_sentence["input_ids"] + split_image1_tokens]
        inputs = inputs["input_ids"].tolist()
        self.assertEqual(inputs, expected_input_ids)

    @unittest.skip(reason="from @molbap @zucchini-nlp, passing non-nested images is error-prone and not recommended")
    def test_non_nested_images_with_batched_text(self):
        processor = self.get_processor()
        processor.image_processor.do_image_splitting = False

        image_str = "<image>"
        text_str_1 = "In this image, we see"
        text_str_2 = "In this image, we see"

        text = [
            image_str + text_str_1,
            image_str + image_str + text_str_2,
        ]
        images = [[self.image1], [self.image2, self.image3]]

        inputs = processor(text=text, images=images, padding=True)

        self.assertEqual(np.array(inputs["pixel_values"]).shape, (2, 2, 3, 512, 512))
        self.assertEqual(np.array(inputs["pixel_attention_mask"]).shape, (2, 2, 512, 512))

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

        expected_rendered = (
            "<|im_start|>User: What do these images show?<image><image><end_of_utterance>\n"
            "Assistant: The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.<end_of_utterance>\n"
            "User: And who is that?<end_of_utterance>\n"
            "Assistant:"
        )
        self.assertEqual(rendered, expected_rendered)

    @unittest.skip(reason="Broken from common. Fixing TODO @zucchini-nlp @molbap")
    def test_chat_template_video_special_processing(self):
        pass

    # Override as SmolVLMProcessor needs image tokens in prompts
    def prepare_text_inputs(self, batch_size: Optional[int] = None):
        if batch_size is None:
            return "lower newer <image>"

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return ["lower newer <image>"]
        return ["lower newer <image>", "<image> upper older longer string"] + ["<image> lower newer"] * (
            batch_size - 2
        )

    # Override tests as inputs_ids padded dimension is the second one but not the last one
    @require_vision
    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=30)

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt", max_length=30)
        self.assertEqual(len(inputs["input_ids"][0]), 30)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        inputs = processor(
            text=input_str,
            images=image_input,
            common_kwargs={"return_tensors": "pt"},
            images_kwargs={"max_image_size": {"longest_edge": 32}},
            text_kwargs={"padding": "max_length", "max_length": 120, "truncation": "longest_first"},
        )
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs["pixel_values"].shape[3], 32)

        self.assertEqual(len(inputs["input_ids"][0]), 120)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"max_image_size": {"longest_edge": 32}},
            "text_kwargs": {"padding": "max_length", "max_length": 120, "truncation": "longest_first"},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs["pixel_values"].shape[3], 32)
        self.assertEqual(len(inputs["input_ids"][0]), 120)

    @require_vision
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=30)

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(len(inputs["input_ids"][0]), 30)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2)
        image_input = self.prepare_image_inputs(batch_size=2)
        image_input = [[image_input[0]], [image_input[1]]]
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            padding="longest",
            max_length=76,
            truncation=True,
            max_image_size={"longest_edge": 30},
        )

        self.assertEqual(inputs["pixel_values"].shape[2], 3)
        self.assertEqual(inputs["pixel_values"].shape[3], 30)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            max_image_size={"longest_edge": 32},
            padding="max_length",
            max_length=120,
            truncation="longest_first",
        )

        self.assertEqual(inputs["pixel_values"].shape[3], 32)
        self.assertEqual(len(inputs["input_ids"][0]), 120)

    @require_torch
    @require_vision
    def test_text_only_inference(self):
        """Test that the processor works correctly with text-only input."""
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding_side="left")
        processor_kwargs = self.prepare_processor_dict()

        processor = self.processor_class(**processor_components, **processor_kwargs)

        text = "This is a simple text without images."
        inputs = processor(text=text, add_special_tokens=False, return_tensors=None)

        tokenized_sentence = processor.tokenizer(text, add_special_tokens=False)
        expected_input_ids = tokenized_sentence["input_ids"]

        self.assertEqual(inputs["input_ids"], expected_input_ids)
        self.assertEqual(inputs["attention_mask"], [[1] * len(expected_input_ids)][0])
        self.assertTrue("pixel_values" not in inputs)
        self.assertTrue("pixel_attention_mask" not in inputs)

        # Test batch of texts without image tokens
        texts = ["First text.", "Second piece of text."]
        batch_inputs = processor(text=texts, padding=True, add_special_tokens=False, return_tensors=None)

        tokenized_1 = processor.tokenizer(texts[0], add_special_tokens=False)
        tokenized_2 = processor.tokenizer(texts[1], add_special_tokens=False)

        expected_1 = tokenized_1["input_ids"]
        expected_2 = tokenized_2["input_ids"]

        # Pad the shorter sequence
        pad_len = len(expected_2) - len(expected_1)
        if pad_len > 0:
            padded_expected_1 = [self.padding_token_id] * pad_len + expected_1
            expected_attention_1 = [0] * pad_len + [1] * len(expected_1)
            self.assertEqual(batch_inputs["input_ids"], [padded_expected_1, expected_2])
            self.assertEqual(batch_inputs["attention_mask"], [expected_attention_1, [1] * len(expected_2)])
        else:
            pad_len = -pad_len
            padded_expected_2 = [self.padding_token_id] * pad_len + expected_2
            expected_attention_2 = [0] * pad_len + [1] * len(expected_2)
            self.assertEqual(batch_inputs["input_ids"], [expected_1, padded_expected_2])
            self.assertEqual(batch_inputs["attention_mask"], [[1] * len(expected_1), expected_attention_2])
