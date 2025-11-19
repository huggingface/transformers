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

import tempfile
import unittest

import numpy as np
import pytest

from transformers import HiggsAudioV2Processor, HiggsAudioV2TokenizerModel
from transformers.models.auto.processing_auto import processor_class_from_name
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


DEFAULT_CHAT_TEMPLATE = """{{- bos_token }}
{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- if messages[0]['content'] is string %}
        {%- set system_message = messages[0]['content']|trim %}
    {%- elif messages[0]['content'] is iterable and messages[0]['content'][0]['type'] == 'text' %}
        {%- set system_message = messages[0]['content'][0]['text']|trim %}
    {%- else %}
        {{- raise_exception("System message content must be a string or contain text type!") }}
    {%- endif %}
    {%- set messages = messages[1:] %}
{%- else %}
    {{- raise_exception("A system message is required but not provided!") }}
{%- endif %}
{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{{- system_message }}
{#- Check for scene message and handle it specially #}
{%- if messages and messages[0]['role'] == 'scene' %}
    {{- "\n\n<|scene_desc_start|>\n" }}
    {%- if messages[0]['content'] is string %}
        {{- messages[0]['content'] | trim }}
    {%- elif messages[0]['content'] is iterable %}
        {%- for content_item in messages[0]['content'] %}
            {%- if content_item['type'] == 'text' %}
                {%- set text_content = content_item['text'] | trim %}
                {{- text_content }}
                {%- if loop.first and not loop.last %}
                    {{- "\n\n" }}
                {%- endif %}
                {%- if not loop.first and not loop.last and messages[0]['content'][loop.index]['type'] != 'audio' %}
                    {{- "\n" }}
                {%- endif %}
            {%- elif content_item['type'] == 'audio' %}
                {{- ' <|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>' }}
                {%- if not loop.last %}
                    {{- "\n" }}
                {%- endif %}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- "\n<|scene_desc_end|>" }}
    {%- set messages = messages[1:] %}
{%- endif %}
{{- "<|eot_id|>" }}
{#- Loop through all messages #}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
    {%- if message['role'] == 'assistant' %}
        {%- if message['content'] is not iterable or message['content'][0]['type'] != 'audio' %}
            {{- raise_exception("Assistant messages must contain audio content only!") }}
        {%- endif %}
        {{- '<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>' }}
    {%- else %}
        {%- if message['content'] is string %}
            {{- message['content'] | trim }}
        {%- elif message['content'] is iterable %}
            {%- for content_item in message['content'] %}
                {%- if content_item['type'] == 'text' %}
                    {{- content_item['text'] | trim }}
                {%- elif content_item['type'] == 'audio' and message['role'] == 'user' %}
                    {{- raise_exception("User messages cannot contain audio content!") }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
    {%- endif %}
    {{- '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>' }}
{%- endif %}
"""


@require_torch
class HiggsAudioV2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = HiggsAudioV2Processor
    audio_input_name = "audio_input_ids"

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = "eustlb/higgs-v2"
        cls.audio_tokenizer_checkpoint = "eustlb/higgs-audio-v2-tokenizer"

        processor = HiggsAudioV2Processor.from_pretrained(cls.checkpoint)
        audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(cls.audio_tokenizer_checkpoint)

        cls.tmpdirname = tempfile.mkdtemp()
        cls.audio_tokenizer_tmpdirname = tempfile.mkdtemp()

        cls.audio_token = processor.audio_token

        processor.save_pretrained(cls.tmpdirname)
        audio_tokenizer.save_pretrained(cls.audio_tokenizer_tmpdirname)

    def prepare_processor_dict(self):
        audio_tokenizer_class = processor_class_from_name("HiggsAudioV2TokenizerModel")
        audio_tokenizer = audio_tokenizer_class.from_pretrained(self.audio_tokenizer_tmpdirname)
        return {
            "audio_tokenizer": audio_tokenizer,
            "chat_template": DEFAULT_CHAT_TEMPLATE,
        }

    @pytest.mark.skip(reason='HiggsAudioV2Processor does not support return_tensors="np"')
    def test_apply_chat_template_audio_0(self):
        pass

    @pytest.mark.skip(reason='HiggsAudioV2Processor does not support return_tensors="np"')
    def test_apply_chat_template_audio_2(self):
        pass

    @pytest.mark.skip(
        reason="This test does not apply to HiggsAudioV2Processor because `decode` method calls the audio_tokenizer.decode"
    )
    def test_apply_chat_template_assistant_mask(self):
        pass

    # TODO: @eustlb, should be fixed at some point
    @pytest.mark.skip()
    def test_processor_from_and_save_pretrained_as_nested_dict(self):
        pass

    # TODO: @eustlb, should be fixed at some point
    @pytest.mark.skip()
    def test_processor_to_json_string(self):
        pass

    @require_torch
    def _test_apply_chat_template(
        self,
        modality: str,
        batch_size: int,
        return_tensors: str,
        input_name: str,
        processor_name: str,
        input_data: list[str],
    ):
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        if processor_name not in self.processor_class.attributes:
            self.skipTest(f"{processor_name} attribute not present in {self.processor_class}")

        # some models have only Fast image processor
        if getattr(processor, processor_name).__class__.__name__.endswith("Fast"):
            return_tensors = "pt"

        batch_messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": "Describe this."}]},
            ]
        ] * batch_size

        # Test that jinja can be applied
        formatted_prompt = processor.apply_chat_template(batch_messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), batch_size)

        # Test that tokenizing with template and directly with `self.tokenizer` gives same output
        formatted_prompt_tokenized = processor.apply_chat_template(
            batch_messages, add_generation_prompt=True, tokenize=True, return_tensors=return_tensors
        )
        add_special_tokens = True
        if processor.tokenizer.bos_token is not None and formatted_prompt[0].startswith(processor.tokenizer.bos_token):
            add_special_tokens = False
        tok_output = processor.tokenizer(
            formatted_prompt, return_tensors=return_tensors, add_special_tokens=add_special_tokens
        )
        expected_output = tok_output.input_ids
        self.assertListEqual(expected_output.tolist(), formatted_prompt_tokenized.tolist())

        # Test that kwargs passed to processor's `__call__` are actually used
        tokenized_prompt_100 = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
            max_length=100,
        )
        self.assertEqual(len(tokenized_prompt_100[0]), 100)

        # Test that `return_dict=True` returns text related inputs in the dict
        out_dict_text = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
        )
        self.assertTrue(all(key in out_dict_text for key in ["input_ids", "attention_mask"]))
        self.assertEqual(len(out_dict_text["input_ids"]), batch_size)
        self.assertEqual(len(out_dict_text["attention_mask"]), batch_size)

        # Test that with modality URLs and `return_dict=True`, we get modality inputs in the dict
        for idx, url in enumerate(input_data[:batch_size]):
            batch_messages[idx] = batch_messages[idx] + [
                {"role": "assistant", "content": [{"type": modality, "url": url}]},
                {"role": "user", "content": [{"type": "text", "text": "Hey, how you doing?"}]},
            ]

        out_dict = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
            num_frames=2,  # by default no more than 2 frames, otherwise too slow
        )
        input_name = getattr(self, input_name)
        self.assertTrue(input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)
        self.assertEqual(len(out_dict[input_name]), batch_size)

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for k in out_dict:
            self.assertIsInstance(out_dict[k], return_tensor_to_type[return_tensors])
