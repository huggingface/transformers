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

import unittest
from types import SimpleNamespace

from transformers import GPT2Tokenizer
from transformers.models.smolvlm.processing_smolvlm import SmolVLMProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.testing_utils import CaptureLogger, require_jinja, require_tokenizers
from transformers.utils import logging


logger = logging.get_logger("transformers.processing_utils")

CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['content'][0]['text'] }}{% endfor %}"
    "{% if add_generation_prompt %}ASSISTANT:{% endif %}"
)
THINKING_TEMPLATE = (
    "{% for message in messages %}{{ message['content'][0]['text'] }}{% endfor %}"
    "{% if enable_thinking is defined and enable_thinking %}THINK{% endif %}"
    "{% if add_generation_prompt %}ASSISTANT:{% endif %}"
)
MISPLACED_KWARGS_WARNING = (
    "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`"
)


class RecordingProcessor(ProcessorMixin):
    def __init__(self, tokenizer, chat_template=None):
        super().__init__(tokenizer, chat_template=chat_template)
        self.captured = None

    def __call__(self, text=None, images=None, videos=None, audio=None, **kwargs):
        self.captured = kwargs
        ids = list(range(20))
        max_length = kwargs.get("max_length")
        if kwargs.get("truncation") and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": [ids]}


class RecordingSmolVLMProcessor(SmolVLMProcessor):
    def __init__(self, tokenizer, chat_template=None):
        # Skip SmolVLMProcessor.__init__: it requires num2words and real sub-processors.
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.video_processor = SimpleNamespace(num_frames=8, fps=1)
        self.captured = None

    def __call__(self, text=None, images=None, videos=None, audio=None, **kwargs):
        self.captured = kwargs
        return {"input_ids": [[0]]}


def _tokenizer():
    vocab = {"<|endoftext|>": 0, "H": 1, "e": 2, "l": 3, "o": 4}
    return GPT2Tokenizer(vocab=vocab, merges=[])


def _messages():
    return [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]


@require_jinja
@require_tokenizers
class ApplyChatTemplateProcessorKwargsTest(unittest.TestCase):
    def test_explicit_processor_kwargs_kept_when_extra_kwarg_passed(self):
        processor = RecordingProcessor(_tokenizer(), chat_template=CHAT_TEMPLATE)
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "processor_kwargs": {"max_length": 5, "truncation": True},
        }
        with CaptureLogger(logger) as logs:
            without_extra = processor.apply_chat_template(_messages(), **kwargs)
            with_extra = processor.apply_chat_template(_messages(), **kwargs, enable_thinking=False)
        self.assertEqual(len(without_extra["input_ids"][0]), 5)
        self.assertEqual(len(with_extra["input_ids"][0]), 5)
        self.assertNotIn("enable_thinking", processor.captured)
        self.assertNotIn(MISPLACED_KWARGS_WARNING, logs.out)
        self.assertNotIn("not a valid argument", logs.out)

    def test_unused_template_kwarg_is_ignored(self):
        processor = RecordingProcessor(_tokenizer(), chat_template=CHAT_TEMPLATE)
        with CaptureLogger(logger) as logs:
            out = processor.apply_chat_template(
                _messages(),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                enable_thinking=False,
            )
        self.assertIn("input_ids", out)
        self.assertNotIn("enable_thinking", processor.captured)
        self.assertNotIn(MISPLACED_KWARGS_WARNING, logs.out)
        self.assertNotIn("not a valid argument", logs.out)

    def test_template_kwarg_used_by_template_is_not_a_processor_kwarg(self):
        processor = RecordingProcessor(_tokenizer(), chat_template=THINKING_TEMPLATE)
        with CaptureLogger(logger) as logs:
            rendered = processor.apply_chat_template(
                _messages(),
                add_generation_prompt=False,
                tokenize=False,
                enable_thinking=True,
                processor_kwargs={"max_length": 5, "truncation": True},
            )
        self.assertIn("THINK", rendered)
        self.assertNotIn(MISPLACED_KWARGS_WARNING, logs.out)

    def test_real_processor_kwargs_in_kwargs_are_forwarded(self):
        processor = RecordingProcessor(_tokenizer(), chat_template=CHAT_TEMPLATE)
        with CaptureLogger(logger) as logs:
            out = processor.apply_chat_template(
                _messages(),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                truncation=True,
                max_length=5,
            )
        self.assertEqual(len(out["input_ids"][0]), 5)
        self.assertEqual(processor.captured["max_length"], 5)
        self.assertTrue(processor.captured["truncation"])
        self.assertIn(MISPLACED_KWARGS_WARNING, logs.out)

    def test_direct_processor_kwargs_do_not_drop_explicit_dict(self):
        processor = RecordingProcessor(_tokenizer(), chat_template=CHAT_TEMPLATE)
        with CaptureLogger(logger) as logs:
            processor.apply_chat_template(
                _messages(),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                padding=True,
                processor_kwargs={"max_length": 5, "truncation": True},
            )
        self.assertEqual(processor.captured["max_length"], 5)
        self.assertTrue(processor.captured["truncation"])
        self.assertTrue(processor.captured["padding"])
        self.assertIn(MISPLACED_KWARGS_WARNING, logs.out)

    def test_smolvlm_frame_defaults_do_not_warn(self):
        processor = RecordingSmolVLMProcessor(_tokenizer(), chat_template=CHAT_TEMPLATE)
        with CaptureLogger(logger) as logs:
            processor.apply_chat_template(_messages(), add_generation_prompt=True, tokenize=True, return_dict=True)
        self.assertEqual(processor.captured["num_frames"], 8)
        self.assertEqual(processor.captured["fps"], 1)
        self.assertNotIn(MISPLACED_KWARGS_WARNING, logs.out)

    def test_smolvlm_keeps_explicit_processor_kwargs_and_user_fps(self):
        processor = RecordingSmolVLMProcessor(_tokenizer(), chat_template=CHAT_TEMPLATE)
        with CaptureLogger(logger) as logs:
            processor.apply_chat_template(
                _messages(),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                processor_kwargs={"max_length": 5, "truncation": True, "fps": 4},
            )
        self.assertEqual(processor.captured["max_length"], 5)
        self.assertTrue(processor.captured["truncation"])
        self.assertEqual(processor.captured["fps"], 4)
        self.assertEqual(processor.captured["num_frames"], 8)
        self.assertNotIn(MISPLACED_KWARGS_WARNING, logs.out)
