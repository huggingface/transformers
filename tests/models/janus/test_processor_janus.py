# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Janus model."""

import tempfile
import unittest

from transformers import JanusProcessor, LlamaTokenizer
from transformers.testing_utils import get_tests_dir
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import JanusImageProcessor


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")

CHAT_TEMPLATE = (
    "{%set seps=['\n\n','<\uFF5Cend▁of▁sentence\uFF5C>']%}"
    "{%set i=0%}"
    "{%for message in messages%}"
        "{%if message['role']=='user'%}"
            "<|User|>: "
        "{%elif message['role']=='assistant'%}"
            "<|Assistant|>:{%if not (loop.last and not add_generation_prompt and message['content'][0]['type']=='text' and message['content'][0]['text']=='')%} {%endif%}"
        "{%else%}"
            "{{message['role'].capitalize()}}: "
        "{%endif%}"
        "{%for content in message['content']%}"
            "{%if content['type']=='image'%}"
                "{%if not loop.first%}{{'\n'}}{%endif%}"
                "<image_placeholder>"
                "{%if not loop.last%}{{'\n'}}{%endif%}"
            "{%elif content['type']=='text'%}"
                "{%set text=content['text']%}"
                "{%if loop.first%}{%set text=text.lstrip()%}{%endif%}"
                "{%if loop.last%}{%set text=text.rstrip()%}{%endif%}"
                "{%if not loop.first and message['content'][loop.index0-1]['type']=='text'%}"
                    "{{' '+text}}"
                "{%else%}"
                    "{{text}}"
                "{%endif%}"
            "{%endif%}"
        "{%endfor%}"
        "{%if not loop.last or add_generation_prompt%}"
            "{%if message['role']=='user'%}"
                "{{seps[0]}}"
            "{%else%}"
                "{{seps[1]}}"
            "{%endif%}"
        "{%endif%}"
    "{%endfor%}"
    "{%if add_generation_prompt%}<|Assistant|>:{%endif%}"
)


class JanusProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = JanusProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        image_processor = JanusImageProcessor()
        tokenizer = LlamaTokenizer(vocab_file=SAMPLE_VOCAB)
        tokenizer.pad_token_id = 0
        tokenizer.sep_token_id = 1
        processor = self.processor_class(image_processor=image_processor, tokenizer=tokenizer, chat_template=CHAT_TEMPLATE)
        processor.save_pretrained(self.tmpdirname)

    def prepare_processor_dict(self):
        return {
            "chat_template": CHAT_TEMPLATE,
        }

    def test_chat_template_matches_original_single(self):
        """
        Tests that the chat template matches the original implementation when applied to a single message.
        """
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        # Single image message
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                        {"type": "image"},
                    ],
                },
            ]
        ]

        correct_prompt = ['<|User|>: What is shown in this image?\n<image_placeholder>\n\n<|Assistant|>:']
        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompt, correct_prompt)

        # Passing generation prompt explicitly
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                    ],
                }
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        self.assertEqual(formatted_prompt, correct_prompt)

        # Single prompt with multiple images
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare this image"},
                        {"type": "image"},
                        {"type": "text", "text": "with this image"},
                        {"type": "image"},
                    ],
                },
            ]
        ]

        correct_prompt = ['<|User|>: Compare this image\n<image_placeholder>\nwith this image\n<image_placeholder>\n\n<|Assistant|>:']
        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompt, correct_prompt)

        # Multiple turns and multiple images
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare this image"},
                        {"type": "image"},
                        {"type": "text", "text": "with this image"},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The first image is an equation, the second is a pie chart."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text",
                         "text": "What about this third image? To which of the previous to is it more similar?"},
                    ],
                },
            ]
        ]

        correct_prompt = ['<|User|>: Compare this image\n<image_placeholder>\nwith this image\n<image_placeholder>\n\n<|Assistant|>: The first image is an equation, the second is a pie chart.<\uFF5Cend▁of▁sentence\uFF5C><|User|>: <image_placeholder>\nWhat about this third image? To which of the previous to is it more similar?\n\n<|Assistant|>:']
        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompt, correct_prompt)

    def test_chat_template_matches_original_batched(self):
        """
        Tests that the chat template matches the original implementation when applied to a batch of messages.
        """
        processor = self.get_processor()
        if processor.chat_template is None:
            self.skipTest("Processor has no chat template")

        # Test 1: Simple single image per message batch
        batched_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                        {"type": "image"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown in this image?"},
                        {"type": "image"},
                    ],
                },
            ]
        ]

        correct_prompts = ['<|User|>: What is shown in this image?\n<image_placeholder>\n\n<|Assistant|>:', '<|User|>: What is shown in this image?\n<image_placeholder>\n\n<|Assistant|>:']
        
        formatted_prompts = processor.apply_chat_template(batched_messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompts, correct_prompts)

        # Test 2: Two images per message batch with different prompts
        batched_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare this image"},
                        {"type": "image"},
                        {"type": "text", "text": "with this image"},
                        {"type": "image"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe how the previous image compares to the following"},
                        {"type": "image"},
                    ],
                },
            ]
        ]

        correct_prompts = ['<|User|>: Compare this image\n<image_placeholder>\nwith this image\n<image_placeholder>\n\n<|Assistant|>:', '<|User|>: <image_placeholder>\nDescribe how the previous image compares to the following\n<image_placeholder>\n\n<|Assistant|>:']
        formatted_prompts = processor.apply_chat_template(batched_messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompts, correct_prompts)

        # Test 3: Multi-turn conversations with multiple images
        batched_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare this image"},
                        {"type": "image"},
                        {"type": "text", "text": "with this image"},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The first image is an equation, the second is a pie chart."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What about this third image? To which of the previous to is it more similar?"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe how the previous image compares to the following"},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The first image is a formula, the second is a plot."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Which of them is closer to the following?"},
                        {"type": "image"},
                    ],
                },
            ]
        ]

        correct_prompts = ['<|User|>: Compare this image\n<image_placeholder>\nwith this image\n<image_placeholder>\n\n<|Assistant|>: The first image is an equation, the second is a pie chart.<｜end▁of▁sentence｜><|User|>: <image_placeholder>\nWhat about this third image? To which of the previous to is it more similar?\n\n<|Assistant|>:', '<|User|>: <image_placeholder>\nDescribe how the previous image compares to the following\n<image_placeholder>\n\n<|Assistant|>: The first image is a formula, the second is a plot.<｜end▁of▁sentence｜><|User|>: Which of them is closer to the following?\n<image_placeholder>\n\n<|Assistant|>:']
        formatted_prompts = processor.apply_chat_template(batched_messages, add_generation_prompt=True)
        self.assertEqual(formatted_prompts, correct_prompts)

