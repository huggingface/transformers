# coding=utf-8
# Copyright 2025 Dustin Loring
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

from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from ..gpt2.tokenization_gpt2 import GPT2Tokenizer


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}


HARMONY_SPECIAL_TOKENS = [
    "<|start|>",
    "<|end|>",
    "<|call|>",
    "<|system|>",
    "<|developer|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|channel|>",
    "<|message|>",
]


HARMONY_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|start|><|system|><|message|>{{ message['content'] }}<|end|>"
    "{% elif message['role'] == 'developer' %}"
    "<|start|><|developer|><|message|>{{ message['content'] }}<|end|>"
    "{% elif message['role'] == 'user' %}"
    "<|start|><|user|><|message|>{{ message['content'] }}<|end|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|start|><|assistant|><|channel|>{{ message['channel'] }}<|message|>{{ message['content'] }}<|end|>"
    "{% elif message['role'] == 'tool' %}"
    "<|start|><|tool|><|message|>{{ message['content'] }}<|end|>"
    "{% endif %}"
    "{% endfor %}"
)


class BlueberryTokenizer(GPT2Tokenizer):
    vocab_files_names = {k: v for k, v in VOCAB_FILES_NAMES.items() if k != "tokenizer_file"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        add_bos_token=False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            **kwargs,
        )

        # Register Harmony special tokens if not already present
        to_add = []
        for tok in HARMONY_SPECIAL_TOKENS:
            if tok not in self.get_vocab():
                to_add.append(tok)
        if to_add:
            self.add_tokens(to_add, special_tokens=True)

        # Set chat template
        try:
            self.chat_template = HARMONY_CHAT_TEMPLATE
        except Exception as e:
            logger.warning(f"Failed to set chat_template for BlueberryTokenizer: {e}")


class BlueberryTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = BlueberryTokenizer

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        add_prefix_space: bool = False,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        # Ensure Harmony tokens exist; if a tokenizer.json is loaded these should exist already, otherwise add.
        try:
            added = 0
            for tok in HARMONY_SPECIAL_TOKENS:
                if self.convert_tokens_to_ids(tok) == self.unk_token_id:
                    self.add_tokens([tok], special_tokens=True)
                    added += 1
            if added:
                logger.info(f"Added {added} Harmony special tokens to BlueberryTokenizerFast")
        except Exception:
            pass

        try:
            self.chat_template = HARMONY_CHAT_TEMPLATE
        except Exception as e:
            logger.warning(f"Failed to set chat_template for BlueberryTokenizerFast: {e}")

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )
        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )
        return super()._encode_plus(*args, **kwargs)


__all__ = ["BlueberryTokenizer", "BlueberryTokenizerFast"]

