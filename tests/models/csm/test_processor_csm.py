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
