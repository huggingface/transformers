# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import ast
import unittest
from pathlib import Path

from transformers.utils.chat_parsing import ResponseParser, parse_response
from transformers.utils.chat_template_utils import _compile_jinja_template


class Gemma4ResponseTemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read the conversion template without importing Orbax/JAX or fetching Hub files.
        source = Path(__file__).resolve().parents[3] / "src/transformers/models/gemma4/convert_gemma4_weights.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls.response_template = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_RESPONSE_TEMPLATE" for target in node.targets)
        )

    def test_tool_call_reasoning_survives_rendering(self):
        text = (
            "<|channel>thought\nI should multiply.\n<channel|>"
            "<|tool_call>call:multiply{a:3,b:4}<tool_call|><|tool_response>"
        )
        # These are the reasoning fields consumed by the published Gemma 4 chat template.
        template = _compile_jinja_template(
            "{% set thought = message.get('reasoning') or message.get('reasoning_content') %}"
            "{% if thought %}<|channel>thought\n{{ thought }}<channel|>{% endif %}"
        )
        for prefix in ("<|turn>model\n", "<tool_response|>"):
            with self.subTest(prefix=prefix):
                message = parse_response(text, self.response_template, prefix=prefix)
                self.assertEqual(message["role"], "assistant")
                self.assertEqual(message["reasoning_content"].strip(), "I should multiply.")
                self.assertIn("I should multiply.", template.render(message=message))
                self.assertEqual(
                    message["tool_calls"],
                    [{"type": "function", "function": {"name": "multiply", "arguments": {"a": 3, "b": 4}}}],
                )

    def test_streamed_reasoning_matches_complete_response(self):
        text = "<|channel>thought\nLet me check.\n<channel|>The answer is 12.<turn|>"
        expected = parse_response(text, self.response_template, prefix="<|turn>model\n")
        self.assertEqual(expected["reasoning_content"].strip(), "Let me check.")
        self.assertEqual(expected["content"], "The answer is 12.")
        parser = ResponseParser(self.response_template, prefix="<|turn>model\n")
        for char in text:
            parser.feed(char)
        message, _ = parser.finalize()
        self.assertEqual(message, expected)

    def test_response_without_reasoning(self):
        message = parse_response("12<turn|>", self.response_template, prefix="<|turn>model\n")
        self.assertEqual(message, {"role": "assistant", "content": "12"})
