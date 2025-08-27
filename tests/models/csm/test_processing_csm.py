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

import json
import shutil
import tempfile
import unittest

import jinja2
import numpy as np

from transformers import CsmProcessor
from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch


@require_torch
class CsmProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = CsmProcessor
    audio_input_name = "input_values"

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = "hf-internal-testing/namespace-sesame-repo_name_csm-1b"
        processor = CsmProcessor.from_pretrained(cls.checkpoint)
        cls.audio_token = processor.audio_token
        cls.audio_token_id = processor.audio_token_id
        cls.pad_token_id = processor.tokenizer.pad_token_id
        cls.bos_token_id = processor.tokenizer.bos_token_id
        cls.tmpdirname = tempfile.mkdtemp()
        processor.save_pretrained(cls.tmpdirname)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def prepare_processor_dict(self):
        return {"chat_template": "\n{%- for message in messages %}\n    {#-- Validate role is a stringified integer --#}\n    {%- if not message['role'] is string or not message['role'].isdigit() %}\n        {{- raise_exception(\"The role must be an integer or a stringified integer (e.g. '0') designating the speaker id\") }}\n    {%- endif %}\n\n    {#-- Validate content is a list --#}\n    {%- set content = message['content'] %}\n    {%- if content is not iterable or content is string %}\n        {{- raise_exception(\"The content must be a list\") }}\n    {%- endif %}\n\n    {#-- Collect content types --#}\n    {%- set content_types = content | map(attribute='type') | list %}\n    {%- set is_last = loop.last %}\n\n    {#-- Last message validation --#}\n    {%- if is_last %}\n        {%- if 'text' not in content_types %}\n            {{- raise_exception(\"The last message must include one item of type 'text'\") }}\n        {%- elif (content_types | select('equalto', 'text') | list | length > 1) or (content_types | select('equalto', 'audio') | list | length > 1) %}\n            {{- raise_exception(\"At most two items are allowed in the last message: one 'text' and one 'audio'\") }}\n        {%- endif %}\n\n    {#-- All other messages validation --#}\n    {%- else %}\n        {%- if content_types | select('equalto', 'text') | list | length != 1\n              or content_types | select('equalto', 'audio') | list | length != 1 %}\n            {{- raise_exception(\"Each message (except the last) must contain exactly one 'text' and one 'audio' item\") }}\n        {%- elif content_types | reject('in', ['text', 'audio']) | list | length > 0 %}\n            {{- raise_exception(\"Only 'text' and 'audio' types are allowed in content\") }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n\n{%- for message in messages %}\n    {{- bos_token }}\n    {{- '[' + message['role'] + ']' }}\n    {{- message['content'][0]['text'] }}\n    {{- eos_token }}\n    {%- if message['content']|length > 1 %}\n        {{- '<|AUDIO|><|audio_eos|>' }}\n    {%- endif %}\n{%- endfor %}\n"}  # fmt: skip

    def test_chat_template_is_saved(self):
        processor_loaded = self.processor_class.from_pretrained(self.tmpdirname)
        processor_dict_loaded = json.loads(processor_loaded.to_json_string())
        # chat templates aren't serialized to json in processors
        self.assertFalse("chat_template" in processor_dict_loaded)

        # they have to be saved as separate file and loaded back from that file
        # so we check if the same template is loaded
        processor_dict = self.prepare_processor_dict()
        self.assertTrue(processor_loaded.chat_template == processor_dict.get("chat_template", None))

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
        if return_tensors != "pt":
            self.skipTest("CSM only supports PyTorch tensors")
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
                {
                    "role": "0",
                    "content": [{"type": "text", "text": "Describe this."}],
                },
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
            batch_messages[idx][0]["content"] = [batch_messages[idx][0]["content"][0], {"type": modality, "url": url}]

        out_dict = processor.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
            num_frames=2,  # by default no more than 2 frames, otherwise too slow
        )
        input_name = getattr(self, input_name)
        print(f"================ input_name={input_name} =================")
        print(f"out_dict={out_dict.keys()}")
        self.assertTrue(input_name in out_dict)
        self.assertEqual(len(out_dict["input_ids"]), batch_size)
        self.assertEqual(len(out_dict["attention_mask"]), batch_size)
        self.assertEqual(len(out_dict[input_name]), batch_size)

        return_tensor_to_type = {"pt": torch.Tensor, "np": np.ndarray, None: list}
        for k in out_dict:
            self.assertIsInstance(out_dict[k], return_tensor_to_type[return_tensors])

        # Test continue from final message
        assistant_message = {
            "role": "1",
            "content": [{"type": "text", "text": "It is the sound of"}],
        }
        for idx, url in enumerate(input_data[:batch_size]):
            batch_messages[idx] = batch_messages[idx] + [assistant_message]
        continue_prompt = processor.apply_chat_template(batch_messages, continue_final_message=True, tokenize=False)
        for prompt in continue_prompt:
            self.assertTrue(prompt.endswith("It is the sound of"))  # no `eos` token at the end

    def test_apply_chat_template(self):
        # Message contains content which a mix of lists with images and image urls and string
        messages = [
            {
                "role": "0",
                "content": [
                    {"type": "text", "text": "This is a test sentence 0."},
                    {"type": "audio"},
                ],
            },
            {
                "role": "1",
                "content": [
                    {"type": "text", "text": "This is a test sentence 1."},
                    {"type": "audio"},
                ],
            },
            {
                "role": "0",
                "content": [
                    {"type": "text", "text": "This is a prompt."},
                ],
            },
        ]
        processor = CsmProcessor.from_pretrained(self.tmpdirname)
        rendered = processor.apply_chat_template(messages, tokenize=False)

        expected_rendered = (
            "<|begin_of_text|>[0]This is a test sentence 0.<|end_of_text|>"
            "<|AUDIO|><|audio_eos|>"
            "<|begin_of_text|>[1]This is a test sentence 1.<|end_of_text|>"
            "<|AUDIO|><|audio_eos|>"
            "<|begin_of_text|>[0]This is a prompt.<|end_of_text|>"
        )
        self.assertEqual(rendered, expected_rendered)

        messages = [
            {
                "role": "0",
                "content": [
                    {"type": "text", "text": "This is a test sentence."},
                ],
            },
            {
                "role": "1",
                "content": [
                    {"type": "text", "text": "This is a test sentence."},
                ],
            },
        ]

        # this should raise an error because the CSM processor requires audio content in the messages expect the last one
        with self.assertRaises(jinja2.exceptions.TemplateError):
            input_ids = processor.apply_chat_template(messages, tokenize=False)

        # now let's very that it expands audio tokens correctly
        messages = [
            {
                "role": "0",
                "content": [
                    {"type": "text", "text": "This is a test sentence."},
                    {"type": "audio", "audio": np.zeros(4096)},
                ],
            },
        ]

        input_ids = processor.apply_chat_template(messages, tokenize=True)

        # 4096 audio input values should give 3 audio tokens
        expected_ids = torch.tensor(
            [[128000, 58, 15, 60, 2028, 374, 264, 1296, 11914, 13, 128001, 128002, 128002, 128002, 128003]]
        )
        torch.testing.assert_close(input_ids, expected_ids)

    @require_torch
    @unittest.skip("CSM doesn't need assistant masks as an audio generation model")
    def test_apply_chat_template_assistant_mask(self):
        pass
