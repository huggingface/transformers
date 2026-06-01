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

from tokenizers import AddedToken, Tokenizer
from tokenizers.models import BPE, WordLevel

from transformers.models.conformer import ConformerTokenizer
from transformers.testing_utils import require_tokenizers


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


@require_tokenizers
class ConformerTokenizationTest(unittest.TestCase):
    @staticmethod
    def get_bpe_tokenizer() -> ConformerTokenizer:
        vocabulary = [UNK_TOKEN, "d", "e", "r", "▁he", "o", "ll", "or", "l", "w", "▁", "▁w", "h", "ld", "he"]
        merges = [("▁", "he"), ("l", "l"), ("o", "r"), ("▁", "w"), ("l", "d"), ("h", "e")]

        tokenizer_object = Tokenizer(
            BPE(
                vocab={token: index for index, token in enumerate(vocabulary)},
                merges=merges,
                unk_token=UNK_TOKEN,
                fuse_unk=True,
            )
        )
        tokenizer_object.add_special_tokens([AddedToken(UNK_TOKEN), AddedToken(PAD_TOKEN)])

        return ConformerTokenizer(tokenizer_object=tokenizer_object, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN)

    @staticmethod
    def get_word_level_tokenizer() -> ConformerTokenizer:
        vocabulary = list(" abcdefghijklmnopqrstuvwxyz")

        tokenizer_object = Tokenizer(WordLevel(vocab={word: index for index, word in enumerate(vocabulary)}))
        tokenizer_object.add_special_tokens([AddedToken(PAD_TOKEN)])

        return ConformerTokenizer(tokenizer_object=tokenizer_object, pad_token=PAD_TOKEN)

    @staticmethod
    def decode_tokens(tokenizer: ConformerTokenizer, tokens: list[str]) -> str:
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        decoded = tokenizer.decode(token_ids)
        assert isinstance(decoded, str)

        return decoded

    def test_bpe_tokenization(self):
        tokenizer = self.get_bpe_tokenizer()

        self.assertEqual(self.decode_tokens(tokenizer, ["▁he", "ll", "o"]), "hello")
        self.assertEqual(self.decode_tokens(tokenizer, ["▁he", "l", "l", "o"]), "helo")
        self.assertEqual(self.decode_tokens(tokenizer, ["▁he", "l", PAD_TOKEN, "l", "o"]), "hello")
        self.assertEqual(self.decode_tokens(tokenizer, ["▁he", "l", "l", PAD_TOKEN, "l", "l", "o"]), "hello")
        self.assertEqual(self.decode_tokens(tokenizer, ["▁he", "ll", "o", "▁w", "or", "ld"]), "hello world")
        self.assertEqual(self.decode_tokens(tokenizer, ["▁he", "ll", "o", "▁w", "▁w", "or", "ld"]), "hello world")
        self.assertEqual(self.decode_tokens(tokenizer, ["▁he", "ll", "o", "▁w", "or", PAD_TOKEN, "ld"]), "hello world")

    def test_word_level_tokenization(self):
        tokenizer = self.get_word_level_tokenizer()

        self.assertEqual(self.decode_tokens(tokenizer, ["h", "e", "l", "l", "o"]), "helo")
        self.assertEqual(self.decode_tokens(tokenizer, ["h", "e", "l", PAD_TOKEN, "l", "o"]), "hello")

        self.assertEqual(
            self.decode_tokens(
                tokenizer,
                ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"],
            ),
            "helo world",
        )
        self.assertEqual(
            self.decode_tokens(
                tokenizer,
                ["h", "e", "l", PAD_TOKEN, "l", "o", " ", "w", "o", "r", "l", "d"],
            ),
            "hello world",
        )
        self.assertEqual(
            self.decode_tokens(
                tokenizer,
                ["h", "e", "l", PAD_TOKEN, "l", "o", " ", " ", "w", "o", "r", "l", "d"],
            ),
            "hello world",
        )
        self.assertEqual(
            self.decode_tokens(
                tokenizer,
                ["h", "e", "l", PAD_TOKEN, "l", "o", " ", "w", "o", "r", PAD_TOKEN, "l", "d"],
            ),
            "hello world",
        )
