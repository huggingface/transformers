# Copyright 2025 Arc Institute and The HuggingFace Inc. team. All rights reserved.
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

import re
import shutil
import tempfile
import unittest

from transformers import Evo2Tokenizer

from ...test_tokenization_common import TokenizerTesterMixin


class Evo2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = Evo2Tokenizer
    test_rust_tokenizer = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tokenizer = Evo2Tokenizer()
        tokenizer.save_pretrained(cls.tmpdirname)

    @classmethod
    def get_tokenizer(cls, pretrained_name=None, **kwargs) -> Evo2Tokenizer:
        pretrained_name = pretrained_name or cls.tmpdirname
        return cls.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5) -> tuple[str, list]:
        toks = []
        for i in range(len(tokenizer)):
            tok = tokenizer.decode([i], clean_up_tokenization_spaces=False)
            toks.append((i, tok))

        toks = list(filter(lambda t: re.match(r"^[ a-zA-Z]+$", t[1]), toks))
        toks = list(filter(lambda t: [t[0]] == tokenizer.encode(t[1], add_special_tokens=False), toks))
        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        if min_length is not None and len(toks) < min_length and len(toks) > 0:
            while len(toks) < min_length:
                toks = toks + toks
        toks_ids = [t[0] for t in toks]

        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        if " " not in output_txt and len(toks_ids) > 1:
            output_txt = (
                tokenizer.decode([toks_ids[0]], clean_up_tokenization_spaces=False)
                + " "
                + tokenizer.decode(toks_ids[1:], clean_up_tokenization_spaces=False)
            )
        if with_prefix_space:
            output_txt = " " + output_txt
        output_ids = tokenizer.encode(output_txt, add_special_tokens=False)
        return output_txt, output_ids

    def test_save_and_load_tokenizer(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                self.assertNotEqual(tokenizer.model_max_length, 42)

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tmpdirname = tempfile.mkdtemp()
                sample_text = "ACGTacgtNn"
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                self.assertListEqual(before_tokens, after_tokens)
                shutil.rmtree(tmpdirname)

        tokenizers = self.get_tokenizers(model_max_length=42)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tmpdirname = tempfile.mkdtemp()
                sample_text = "ACGTacgtNn"
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                self.assertListEqual(before_tokens, after_tokens)
                self.assertEqual(after_tokenizer.model_max_length, 42)

                tokenizer = tokenizer.__class__.from_pretrained(tmpdirname, model_max_length=43)
                self.assertEqual(tokenizer.model_max_length, 43)
                shutil.rmtree(tmpdirname)

    @unittest.skip(reason="inputs cannot be pretokenized as ids depend on single-byte characters")
    def test_pretokenized_inputs(self):
        pass

    @unittest.skip(reason="Evo2Tokenizer maps single bytes and clamps out-of-vocabulary strings")
    def test_conversion_reversible(self):
        pass

    def test_convert_tokens_to_string_format(self):
        tokenizers = self.get_tokenizers(fast=True, do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokens = ["A", "C", "G", "T", " ", "a", "c", "g", "t"]
                string = tokenizer.convert_tokens_to_string(tokens)
                self.assertIsInstance(string, str)
                self.assertEqual(string, "ACGT acgt")

    def test_round_trip_ascii(self):
        tokenizer = Evo2Tokenizer()
        text = "Hello, Evo2!"
        encoded = tokenizer(text)["input_ids"]
        expected = list(text.encode("utf-8"))
        self.assertEqual(encoded, expected)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_clamp_behavior(self):
        tokenizer = Evo2Tokenizer()
        tokens = [0, 1, 255, 300, -5]
        decoded = tokenizer.decode(tokens)
        expected = "".join(chr(max(0, min(255, token))) for token in tokens)
        self.assertEqual(decoded, expected)

    def test_tokenize_returns_byte_chars(self):
        tokenizer = Evo2Tokenizer()
        text = "ACGT"
        tokens = tokenizer.tokenize(text)
        self.assertEqual(tokens, ["A", "C", "G", "T"])
        self.assertEqual(tokenizer.convert_tokens_to_ids(tokens), [65, 67, 71, 84])
