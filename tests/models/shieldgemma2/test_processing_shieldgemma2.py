# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import shutil
import tempfile
import unittest
from collections.abc import Mapping

from parameterized import parameterized

from transformers import GemmaTokenizer, ShieldGemma2Processor
from transformers.testing_utils import get_tests_dir, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Gemma3ImageProcessor

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")

# Copied from _CHAT_TEMPLATE in src/transformers/models/shieldgemma2/convert_shieldgemma2_weights_orbax_to_hf.py
_CHAT_TEMPLATE = """{{ bos_token }}
{%- for message in messages -%}
    {%- if message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {{ '<start_of_turn>user\n' }}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ 'You are a policy expert trying to help determine whether a generated or real image is in' }}
                {{ ' violation of the safety policy below. Your decision must very clearly follow the specific'}}
                {{ ' language used in the safety policy and should not include any interpretation outside of the' }}
                {{ ' safety policy.\n\n' }}
                {{ '<BEGIN_SAFETY_POLICY>\n' }}
                {{ item['text'] + '\n' }}
                {{ '</END_SAFETY_POLICY>\n\n' }}
                {{ 'Does the image violate the above policy? Your Answer must start with \\'Yes\\' or \\'No\\'.' }}
                {{ '<end_of_turn>\n' }}
            {%- endif -%}
        {%- endfor -%}
        {{'<start_of_turn>model\n'}}
    {%- else -%}
        {{ raise_exception("Conversation messages must contain iterable content containing images and policy definitions in text.") }}
    {%- endif -%}
{%- endfor -%}
"""

# Simplified from _SHIELDGEMMA2_POLICIES in src/transformers/models/shieldgemma2/convert_shieldgemma2_weights_orbax_to_hf.py
_SHIELDGEMMA2_POLICIES: Mapping[str, str] = {
    "dangerous": "Test policy related to dangerous content.",
    "sexual": "Test policy related to sexually explicit content.",
    "violence": "Test policy related to violent content.",
}


@require_vision
class ShieldGemma2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ShieldGemma2Processor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        image_processor = Gemma3ImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

        extra_special_tokens = {
            "image_token": "<image_soft_token>",
            "boi_token": "<start_of_image>",
            "eoi_token": "<end_of_image>",
        }
        tokenizer = GemmaTokenizer(SAMPLE_VOCAB, keep_accents=True, extra_special_tokens=extra_special_tokens)

        processor_kwargs = cls.prepare_processor_dict()
        processor = ShieldGemma2Processor(image_processor=image_processor, tokenizer=tokenizer, **processor_kwargs)
        processor.save_pretrained(cls.tmpdirname)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @classmethod
    def prepare_processor_dict(cls):
        return {
            "chat_template": _CHAT_TEMPLATE,
            "policy_definitions": _SHIELDGEMMA2_POLICIES,
        }

    def test_policy_definitions_saved_in_config(self):
        processor_config_path = os.path.join(self.tmpdirname, "processor_config.json")

        with open(processor_config_path, "rb") as processor_config_file:
            json_dict = json.load(processor_config_file)

        self.assertIsInstance(json_dict, dict)
        self.assertIn("policy_definitions", json_dict)
        self.assertIs(len(json_dict["policy_definitions"]), 3)

    @parameterized.expand(
        [
            ("all_policies", None, 3),
            ("selected_policies", ["dangerous", "violence"], 2),
            ("single_policy", ["sexual"], 1),
        ]
    )
    def test_with_default_policies(self, name, policies, expected_batch_size):
        processor = self.get_processor()

        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        images = self.prepare_image_inputs()
        processed_inputs = processor(images=images, policies=policies)
        self.assertEqual(len(processed_inputs[self.text_input_name]), expected_batch_size)
        self.assertEqual(len(processed_inputs[self.images_input_name]), expected_batch_size)

    @parameterized.expand(
        [
            ("all_policies", None, 6),
            ("selected_policies_from_both", ["cbrne", "dangerous", "specialized_advice", "violence"], 4),
            ("selected_policies_from_custom", ["cbrne", "specialized_advice"], 2),
            ("selected_policies_from_default", ["dangerous", "violence"], 2),
            ("single_policy_from_custom", ["ip"], 1),
            ("single_policy_from_default", ["sexual"], 1),
        ]
    )
    def test_with_custom_policies(self, name, policies, expected_batch_size):
        processor = self.get_processor()

        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        # Test policies adapted from https://ailuminate.mlcommons.org/benchmarks/ hazard categories
        custom_policies = {
            "cbrne": "Test policy related to indiscriminate weapons.",
            "ip": "Test policy related to intellectual property.",
            "specialized_advice": "Test policy related to specialized advice.",
        }

        images = self.prepare_image_inputs()
        processed_inputs = processor(images=images, custom_policies=custom_policies, policies=policies)
        self.assertEqual(len(processed_inputs[self.text_input_name]), expected_batch_size)
        self.assertEqual(len(processed_inputs[self.images_input_name]), expected_batch_size)

    def test_with_multiple_images(self):
        processor = self.get_processor()

        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        images = self.prepare_image_inputs(batch_size=2)
        processed_inputs = processor(images=images)
        self.assertEqual(len(processed_inputs[self.text_input_name]), 6)
        self.assertEqual(len(processed_inputs[self.images_input_name]), 6)

    # TODO(ryanmullins): Adapt this test for ShieldGemma 2
    @parameterized.expand([(1, "np"), (1, "pt"), (2, "np"), (2, "pt")])
    @unittest.skip("ShieldGemma 2 chat template requires different message structure from parent.")
    def test_apply_chat_template_image(self, batch_size: int, return_tensors: str):
        pass

    # TODO(ryanmullins): Adapt this test for ShieldGemma 2
    @unittest.skip("Parent test needs to be adapted for ShieldGemma 2.")
    def test_unstructured_kwargs_batched(self):
        pass

    # TODO(ryanmullins): Adapt this test for ShieldGemma 2
    @unittest.skip("Parent test needs to be adapted for ShieldGemma 2.")
    def test_unstructured_kwargs(self):
        pass

    # TODO(ryanmullins): Adapt this test for ShieldGemma 2
    @unittest.skip("Parent test needs to be adapted for ShieldGemma 2.")
    def test_tokenizer_defaults_preserved_by_kwargs(self):
        pass

    # TODO(ryanmullins): Adapt this test for ShieldGemma 2
    @unittest.skip("Parent test needs to be adapted for ShieldGemma 2.")
    def test_structured_kwargs_nested_from_dict(self):
        pass

    # TODO(ryanmullins): Adapt this test for ShieldGemma 2
    @unittest.skip("Parent test needs to be adapted for ShieldGemma 2.")
    def test_structured_kwargs_nested(self):
        pass

    # TODO(ryanmullins): Adapt this test for ShieldGemma 2
    @unittest.skip("Parent test needs to be adapted for ShieldGemma 2.")
    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        pass

    # TODO(ryanmullins): Adapt this test for ShieldGemma 2
    @unittest.skip("Parent test needs to be adapted for ShieldGemma 2.")
    def test_kwargs_overrides_default_image_processor_kwargs(self):
        pass

    @unittest.skip("ShieldGemma requires images in input, and fails in text-only processing")
    def test_apply_chat_template_assistant_mask(self):
        pass

    def test_processor_text_has_no_visual(self):
        # Overwritten: Shieldgemma has a complicated processing so we don't check id values
        processor = self.get_processor()

        text = self.prepare_text_inputs(batch_size=3, modalities="image")
        image_inputs = self.prepare_image_inputs(batch_size=3)
        processing_kwargs = {"return_tensors": "pt", "padding": True, "multi_page": True}

        # Call with nested list of vision inputs
        image_inputs_nested = [[image] if not isinstance(image, list) else image for image in image_inputs]
        inputs_dict_nested = {"text": text, "images": image_inputs_nested}
        inputs = processor(**inputs_dict_nested, **processing_kwargs)
        self.assertTrue(self.text_input_name in inputs)

        # Call with one of the samples with no associated vision input
        plain_text = "lower newer"
        image_inputs_nested[0] = []
        text[0] = plain_text
        inputs_dict_no_vision = {"text": text, "images": image_inputs_nested}
        inputs_nested = processor(**inputs_dict_no_vision, **processing_kwargs)
        self.assertTrue(self.text_input_name in inputs_nested)
