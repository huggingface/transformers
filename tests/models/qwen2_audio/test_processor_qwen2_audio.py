# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Optional

from transformers import AutoProcessor, AutoTokenizer, Qwen2AudioProcessor, WhisperFeatureExtractor
from transformers.testing_utils import require_torch, require_torchaudio
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available:
    pass


@require_torch
@require_torchaudio
class Qwen2AudioProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen2AudioProcessor

    def setUp(self):
        self.checkpoint = "Qwen/Qwen2-Audio-7B-Instruct"
        self.tmpdirname = tempfile.mkdtemp()

        processor_kwargs = self.prepare_processor_dict()
        processor = Qwen2AudioProcessor.from_pretrained(self.checkpoint, **processor_kwargs)
        processor.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_audio_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).audio_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_processor_dict(self):
        return {
            "chat_template": "{% set audio_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if 'audio' in content or 'audio_url' in content or message['type'] == 'audio' %}{% set audio_count.value = audio_count.value + 1 %}Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        }

    # Override as Qwen2AudioProcessor needs audio tokens in prompts
    def prepare_text_inputs(self, batch_size: Optional[int] = None):
        if batch_size is None:
            return "lower newer <|AUDIO|>"

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return ["lower newer <|AUDIO|>"]
        return ["lower newer <|AUDIO|>", "<|AUDIO|> upper older longer string"] + ["<|AUDIO|> lower newer"] * (
            batch_size - 2
        )

    def test_can_load_various_tokenizers(self):
        processor = Qwen2AudioProcessor.from_pretrained(self.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    def test_save_load_pretrained_default(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        processor = Qwen2AudioProcessor.from_pretrained(self.checkpoint)
        feature_extractor = processor.feature_extractor

        processor = Qwen2AudioProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

        processor.save_pretrained(self.tmpdirname)
        processor = Qwen2AudioProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, WhisperFeatureExtractor)

    def test_tokenizer_integration(self):
        slow_tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=False)
        fast_tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, from_slow=True, legacy=False)

        prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>\nWhat is it in this audio?<|im_end|><|im_start|>assistant\n"
        EXPECTED_OUTPUT = [
            "<|im_start|>",
            "system",
            "Ċ",
            "Answer",
            "Ġthe",
            "Ġquestions",
            ".",
            "<|im_end|>",
            "<|im_start|>",
            "user",
            "Ċ",
            "<|audio_bos|>",
            "<|AUDIO|>",
            "<|audio_eos|>",
            "Ċ",
            "What",
            "Ġis",
            "Ġit",
            "Ġin",
            "Ġthis",
            "Ġaudio",
            "?",
            "<|im_end|>",
            "<|im_start|>",
            "assistant",
            "Ċ",
        ]

        self.assertEqual(slow_tokenizer.tokenize(prompt), EXPECTED_OUTPUT)
        self.assertEqual(fast_tokenizer.tokenize(prompt), EXPECTED_OUTPUT)

    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)
        expected_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\nWhat's that sound?<|im_end|>\n<|im_start|>assistant\nIt is the sound of glass shattering.<|im_end|>\n<|im_start|>user\nAudio 2: <|audio_bos|><|AUDIO|><|audio_eos|>\nHow about this one?<|im_end|>\n<|im_start|>assistant\n"

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
                    },
                    {"type": "text", "text": "What's that sound?"},
                ],
            },
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav",
                    },
                    {"type": "text", "text": "How about this one?"},
                ],
            },
        ]

        formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)

    def test_chat_template_with_continue_final_message(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)
        expected_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\nWhat's that sound?<|im_end|>\n<|im_start|>assistant\nIt is the sound of "  # fmt: skip
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
                    },
                    {"type": "text", "text": "What's that sound?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "It is the sound of "}],
            },
        ]
        prompt = processor.apply_chat_template(messages, continue_final_message=True)
        self.assertEqual(expected_prompt, prompt)
