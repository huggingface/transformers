# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Regression tests for https://github.com/huggingface/transformers/issues/40913.

Verify that a caller-supplied ``chat_template`` kwarg is not silently
overwritten by the model's default template in ProcessorMixin.get_processor_dict.
"""

import json
import os
import shutil
import tempfile
import unittest

from transformers.processing_utils import ProcessorMixin
from transformers.utils import CHAT_TEMPLATE_FILE


class TestChatTemplateKwargOverride(unittest.TestCase):
    MODEL_TEMPLATE = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        with open(os.path.join(self.tmpdir, CHAT_TEMPLATE_FILE), "w", encoding="utf-8") as fh:
            fh.write(self.MODEL_TEMPLATE)
        with open(os.path.join(self.tmpdir, "processor_config.json"), "w", encoding="utf-8") as fh:
            json.dump({}, fh)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_user_chat_template_preserved_in_returned_kwargs(self):
        """A caller-supplied chat_template must survive get_processor_dict in returned_kwargs."""
        user_template = "{{ messages | tojson }}"
        _processor_dict, returned_kwargs = ProcessorMixin.get_processor_dict(self.tmpdir, chat_template=user_template)
        self.assertEqual(
            returned_kwargs.get("chat_template"),
            user_template,
            "User-supplied chat_template was silently overwritten by the model's default template "
            "(regression of https://github.com/huggingface/transformers/issues/40913).",
        )

    def test_model_chat_template_loaded_into_processor_dict(self):
        """Without a caller-supplied kwarg, the model's on-disk template must appear in processor_dict."""
        processor_dict, _kwargs = ProcessorMixin.get_processor_dict(self.tmpdir)
        self.assertIn(
            "chat_template",
            processor_dict,
            "chat_template was not loaded from the model directory into processor_dict.",
        )
        self.assertEqual(
            processor_dict["chat_template"],
            self.MODEL_TEMPLATE,
            "Loaded chat_template does not match the model's chat_template.jinja content.",
        )
