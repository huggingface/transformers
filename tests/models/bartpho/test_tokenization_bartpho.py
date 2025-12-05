# Copyright 2021 HuggingFace Inc. team.
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


import os
import tempfile
import unittest

from transformers.models.bartpho.tokenization_bartpho import VOCAB_FILES_NAMES, BartphoTokenizer
from transformers.testing_utils import get_tests_dir

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece_bpe.model")


class BartphoTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "vinai/bartpho-syllable"
    tokenizer_class = BartphoTokenizer
    test_rust_tokenizer = False
    test_sentencepiece = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.special_tokens_map = {"unk_token": "<unk>"}

    @classmethod
    def get_tokenizer(cls, pretrained_name=None, **kwargs):
        """Create a fresh tokenizer for each test instead of loading from saved."""
        kwargs.update(cls.special_tokens_map)

        # Create a temporary directory for this tokenizer
        tmpdir = tempfile.mkdtemp()
        vocab = ["▁This", "▁is", "▁a", "▁t", "est"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        monolingual_vocab_file = os.path.join(tmpdir, VOCAB_FILES_NAMES["monolingual_vocab_file"])
        with open(monolingual_vocab_file, "w", encoding="utf-8") as fp:
            fp.writelines(f"{token} {vocab_tokens[token]}\n" for token in vocab_tokens)

        return BartphoTokenizer(SAMPLE_VOCAB, monolingual_vocab_file, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "This is a là test"
        output_text = "This is a<unk><unk> test"
        return input_text, output_text

    def test_full_tokenizer(self):
        vocab = ["▁This", "▁is", "▁a", "▁t", "est"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        special_tokens_map = {"unk_token": "<unk>"}

        with tempfile.TemporaryDirectory() as tmpdir:
            monolingual_vocab_file = os.path.join(tmpdir, VOCAB_FILES_NAMES["monolingual_vocab_file"])
            with open(monolingual_vocab_file, "w", encoding="utf-8") as fp:
                fp.writelines(f"{token} {vocab_tokens[token]}\n" for token in vocab_tokens)
            tokenizer = BartphoTokenizer(SAMPLE_VOCAB, monolingual_vocab_file, **special_tokens_map)

            text = "This is a là test"
            bpe_tokens = "▁This ▁is ▁a ▁l à ▁t est".split()
            tokens = tokenizer.tokenize(text)
            self.assertListEqual(tokens, bpe_tokens)

            input_tokens = tokens + [tokenizer.unk_token]
            input_bpe_tokens = [4, 5, 6, 3, 3, 7, 8, 3]
            self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)
