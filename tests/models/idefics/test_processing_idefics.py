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

import unittest

import numpy as np

from transformers import (
    IdeficsProcessor,
)
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    pass

if is_vision_available():
    from PIL import Image


@require_torch
@require_vision
class IdeficsProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = IdeficsProcessor
    input_keys = ["pixel_values", "input_ids", "attention_mask", "image_attention_mask"]

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(return_tensors="pt")

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        return tokenizer_class.from_pretrained("HuggingFaceM4/tiny-random-idefics")

    def prepare_prompts(self):
        """This function prepares a list of PIL images"""

        num_images = 2
        images = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8) for x in range(num_images)]
        images = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in images]

        # print([type(x) for x in images])
        # die

        prompts = [
            # text and 1 image
            [
                "User:",
                images[0],
                "Describe this image.\nAssistant:",
            ],
            # text and images
            [
                "User:",
                images[0],
                "Describe this image.\nAssistant: An image of two dogs.\n",
                "User:",
                images[1],
                "Describe this image.\nAssistant:",
            ],
            # only text
            [
                "User:",
                "Describe this image.\nAssistant: An image of two kittens.\n",
                "User:",
                "Describe this image.\nAssistant:",
            ],
            # only images
            [
                images[0],
                images[1],
            ],
        ]

        return prompts

    def test_save_load_pretrained_additional_features(self):
        tokenizer_add_kwargs = self.get_component("tokenizer", bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_component("image_processor", do_normalize=False, padding_value=1.0)
        processor = IdeficsProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, self._get_component_class_from_processor("tokenizer"))

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, self._get_component_class_from_processor("image_processor"))

    def test_tokenizer_padding(self):
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", padding_side="right")

        processor = IdeficsProcessor(tokenizer=tokenizer, image_processor=image_processor, return_tensors="pt")

        predicted_tokens = [
            "<s>Describe this image.\nAssistant:<unk><unk><unk><unk><unk><unk><unk><unk><unk>",
            "<s>Describe this image.\nAssistant:<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>",
        ]
        predicted_attention_masks = [
            ([1] * 10) + ([0] * 9),
            ([1] * 10) + ([0] * 10),
        ]
        prompts = [[prompt] for prompt in self.prepare_prompts()[2]]

        max_length = processor(text=prompts, padding="max_length", truncation=True, max_length=20, return_tensors="pt")
        longest = processor(text=prompts, padding="longest", truncation=True, max_length=30, return_tensors="pt")

        decoded_max_length = processor.tokenizer.decode(max_length["input_ids"][-1])
        decoded_longest = processor.tokenizer.decode(longest["input_ids"][-1])

        self.assertEqual(decoded_max_length, predicted_tokens[1])
        self.assertEqual(decoded_longest, predicted_tokens[0])

        self.assertListEqual(max_length["attention_mask"][-1].tolist(), predicted_attention_masks[1])
        self.assertListEqual(longest["attention_mask"][-1].tolist(), predicted_attention_masks[0])

    def test_tokenizer_left_padding(self):
        """Identical to test_tokenizer_padding, but with padding_side not explicitly set."""
        processor = self.get_processor()

        predicted_tokens = [
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><s>Describe this image.\nAssistant:",
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><s>Describe this image.\nAssistant:",
        ]
        predicted_attention_masks = [
            ([0] * 9) + ([1] * 10),
            ([0] * 10) + ([1] * 10),
        ]
        prompts = [[prompt] for prompt in self.prepare_prompts()[2]]
        max_length = processor(text=prompts, padding="max_length", truncation=True, max_length=20)
        longest = processor(text=prompts, padding="longest", truncation=True, max_length=30)

        decoded_max_length = processor.tokenizer.decode(max_length["input_ids"][-1])
        decoded_longest = processor.tokenizer.decode(longest["input_ids"][-1])

        self.assertEqual(decoded_max_length, predicted_tokens[1])
        self.assertEqual(decoded_longest, predicted_tokens[0])

        self.assertListEqual(max_length["attention_mask"][-1].tolist(), predicted_attention_masks[1])
        self.assertListEqual(longest["attention_mask"][-1].tolist(), predicted_attention_masks[0])

    def test_tokenizer_defaults(self):
        # Override to account for the processor prefixing the BOS token to prompts.
        components = {attribute: self.get_component(attribute) for attribute in self.processor_class.get_attributes()}
        processor = self.processor_class(**components)
        tokenizer = components["tokenizer"]

        input_str = ["lower newer"]
        encoded_processor = processor(text=input_str, padding=False, return_tensors="pt")
        encoded_tok = tokenizer(
            [f"{tokenizer.bos_token}{input_str[0]}"], padding=False, add_special_tokens=False, return_tensors="pt"
        )

        for key in encoded_tok:
            if key in encoded_processor:
                self.assertListEqual(encoded_tok[key].tolist(), encoded_processor[key].tolist())
