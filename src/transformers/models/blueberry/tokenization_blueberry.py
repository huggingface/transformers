# coding=utf-8
# Copyright 2025 Dustin Loring
# Website: dustinwloring1988.github.io
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

from typing import Optional

from ..gpt2.tokenization_gpt2 import GPT2Tokenizer
from ...tokenization_utils_fast import PreTrainedTokenizerFast


HARMONY_SPECIAL_TOKENS = [
    "<|start|>",
    "<|end|>",
    "<|call|>",
    "<|system|>",
    "<|developer|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|message|>",
    "<|channel|>",
]


HARMONY_CHAT_TEMPLATE = (
    "{%- for message in messages -%}\n"
    "{%- if message['role'] == 'system' -%}\n"
    "<|start|><|system|><|message|>{{ message['content'] }}<|end|>\n"
    "{%- elif message['role'] == 'developer' -%}\n"
    "<|start|><|developer|><|message|>{{ message['content'] }}<|end|>\n"
    "{%- elif message['role'] == 'user' -%}\n"
    "<|start|><|user|><|message|>{{ message['content'] }}<|end|>\n"
    "{%- elif message['role'] == 'assistant' -%}\n"
    "<|start|><|assistant|><|channel|>{{ message['channel'] }}<|message|>{{ message['content'] }}<|end|>\n"
    "{%- elif message['role'] == 'tool' -%}\n"
    "<|start|><|tool|><|message|>{{ message['content'] }}<|end|>\n"
    "{%- endif -%}\n"
    "{%- endfor -%}"
)


class BlueberryTokenizer(GPT2Tokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add Harmony special tokens if not present
        self.add_special_tokens({"additional_special_tokens": [tok for tok in HARMONY_SPECIAL_TOKENS if tok not in self.get_vocab()]})
        # Set chat template
        self.chat_template = HARMONY_CHAT_TEMPLATE


class BlueberryTokenizerFast(PreTrainedTokenizerFast):
    slow_tokenizer_class = BlueberryTokenizer

    def __init__(self, tokenizer_file: Optional[str] = None, **kwargs):
        super().__init__(tokenizer_file=tokenizer_file, **kwargs)
        # Add Harmony special tokens if not present
        self.add_special_tokens({"additional_special_tokens": [tok for tok in HARMONY_SPECIAL_TOKENS if tok not in self.get_vocab()]})
        # Set chat template
        self.chat_template = HARMONY_CHAT_TEMPLATE


__all__ = ["BlueberryTokenizer", "BlueberryTokenizerFast"]

