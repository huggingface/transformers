# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
Tests for `BertTokenizerLegacy`, the pure-Python WordPiece tokenizer.

It is not reachable through `AutoTokenizer` any more, but it is far from unused: `pipelines/token_classification.py`
imports `BasicTokenizer` from this module, `data/processors/squad.py` imports `whitespace_tokenize`, and the
tokenizers of tapas, roc_bert, prophetnet, bert_japanese and openai are all built on its helpers.

The `BasicTokenizer` / `WordpieceTokenizer` casing and accent matrix is exercised in
`tests/models/prophetnet/test_tokenization_prophetnet.py`, which imports those helpers from this module. What is
covered here is the part nothing else reaches: the `BertTokenizerLegacy` class itself, and `whitespace_tokenize`.
"""

import os
import tempfile
import unittest

from transformers.models.bert.tokenization_bert_legacy import (
    VOCAB_FILES_NAMES,
    BertTokenizerLegacy,
    whitespace_tokenize,
)


VOCAB_TOKENS = [
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[MASK]",
    "want",
    "##want",
    "##ed",
    "wa",
    "un",
    "runn",
    "##ing",
    ",",
    "low",
    "lowest",
]


class BertTokenizerLegacyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(cls.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join(token + "\n" for token in VOCAB_TOKENS))

    def get_tokenizer(self, **kwargs):
        return BertTokenizerLegacy(self.vocab_file, **kwargs)

    def test_full_tokenizer(self):
        tokenizer = self.get_tokenizer()

        tokens = tokenizer.tokenize("UNwantéd,running")

        self.assertListEqual(tokens, ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), [9, 6, 7, 12, 10, 11])

    def test_vocab_size_and_get_vocab(self):
        tokenizer = self.get_tokenizer()

        self.assertEqual(tokenizer.vocab_size, len(VOCAB_TOKENS))
        self.assertEqual(tokenizer.get_vocab()["##want"], 6)

    def test_missing_vocab_file_raises(self):
        with self.assertRaises(ValueError):
            BertTokenizerLegacy(os.path.join(self.tmpdirname, "does-not-exist.txt"))

    def test_do_basic_tokenize_false_goes_straight_to_wordpiece(self):
        # Without basic tokenization the input is only split on whitespace, so punctuation stays glued to the word
        # and no longer matches the vocabulary.
        tokenizer = self.get_tokenizer(do_basic_tokenize=False)

        self.assertListEqual(tokenizer.tokenize("unwanted , running"), ["un", "##want", "##ed", ",", "runn", "##ing"])
        self.assertListEqual(tokenizer.tokenize("unwanted, running"), ["[UNK]", "runn", "##ing"])

    def test_never_split_is_honored(self):
        tokenizer = self.get_tokenizer(never_split=["[UNK]"])

        self.assertListEqual(tokenizer.tokenize("lowest [UNK]"), ["lowest", "[UNK]"])

    def test_tokenize_chinese_chars_can_be_disabled(self):
        text = "want博推want"

        spaced = self.get_tokenizer(tokenize_chinese_chars=True).tokenize(text)
        unspaced = self.get_tokenizer(tokenize_chinese_chars=False).tokenize(text)

        # With CJK spacing on, the two han characters are isolated and the surrounding word pieces survive.
        self.assertListEqual(spaced, ["want", "[UNK]", "[UNK]", "want"])
        # With it off, the whole run is one token and falls out of the vocabulary.
        self.assertListEqual(unspaced, ["[UNK]"])

    def test_convert_tokens_to_string(self):
        tokenizer = self.get_tokenizer()

        self.assertEqual(tokenizer.convert_tokens_to_string(["un", "##want", "##ed"]), "unwanted")

    def test_build_inputs_with_special_tokens(self):
        tokenizer = self.get_tokenizer()
        cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id

        text = tokenizer.encode("want", add_special_tokens=False)
        text_pair = tokenizer.encode("lowest", add_special_tokens=False)

        self.assertEqual(tokenizer.build_inputs_with_special_tokens(text), [cls_id] + text + [sep_id])
        self.assertEqual(
            tokenizer.build_inputs_with_special_tokens(text, text_pair),
            [cls_id] + text + [sep_id] + text_pair + [sep_id],
        )

    def test_create_token_type_ids_from_sequences(self):
        tokenizer = self.get_tokenizer()

        text = tokenizer.encode("want", add_special_tokens=False)
        text_pair = tokenizer.encode("lowest", add_special_tokens=False)

        token_type_ids = tokenizer.create_token_type_ids_from_sequences(text, text_pair)

        # 0 for [CLS] + first segment + [SEP], 1 for the second segment + its [SEP].
        self.assertEqual(token_type_ids, [0] * (len(text) + 2) + [1] * (len(text_pair) + 1))

    def test_get_special_tokens_mask(self):
        tokenizer = self.get_tokenizer()

        ids = tokenizer.encode("want")
        mask = tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)

        self.assertEqual(mask[0], 1)
        self.assertEqual(mask[-1], 1)
        self.assertEqual(sum(mask), 2)

    def test_save_and_reload_vocabulary(self):
        tokenizer = self.get_tokenizer()
        sequence = "UNwantéd,running"

        with tempfile.TemporaryDirectory() as tmpdirname:
            tokenizer.save_pretrained(tmpdirname)
            reloaded = BertTokenizerLegacy.from_pretrained(tmpdirname)

        self.assertEqual(reloaded.get_vocab(), tokenizer.get_vocab())
        self.assertListEqual(reloaded.tokenize(sequence), tokenizer.tokenize(sequence))


class WhitespaceTokenizeTest(unittest.TestCase):
    """`whitespace_tokenize` is the helper `data/processors/squad.py` relies on to align answer spans."""

    def test_splits_on_any_whitespace(self):
        self.assertListEqual(whitespace_tokenize("a  b\tc\nd"), ["a", "b", "c", "d"])

    def test_strips_surrounding_whitespace(self):
        self.assertListEqual(whitespace_tokenize("  padded  "), ["padded"])

    def test_empty_text_gives_no_tokens(self):
        self.assertListEqual(whitespace_tokenize("   "), [])
