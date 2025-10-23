# Copyright 2019 Hugging Face inc.
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

from transformers import DebertaV2Tokenizer, AutoTokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers, slow
from transformers.tokenization_sentencepiece import SentencePieceExtractor
from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/spiece.model")


@require_sentencepiece
@require_tokenizers
class DebertaV2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "microsoft/deberta-v2-xlarge"
    tokenizer_class = DebertaV2Tokenizer
    rust_tokenizer_class = DebertaV2Tokenizer
    test_sentencepiece = True
    test_sentencepiece_ignore_case = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # We have a SentencePiece fixture for testing
        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tok_from_vocab = DebertaV2Tokenizer(vocab=vocab_scores)
        tok_from_vocab.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tok_from_vocab]

    def get_input_output_texts(self, tokenizer):
        input_text = "this is a test"
        output_text = "this is a test"
        return input_text, output_text

    def test_vocab_size(self):
        self.assertEqual(self.get_tokenizer().vocab_size, 30_000)

    def test_do_lower_case(self):
        # fmt: off
        sequence = " \tHeLLo!how  \n Are yoU?  "
        tokens_target = ["▁hello", "!", "how", "▁are", "▁you", "?"]
        # fmt: on

        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tokenizer = DebertaV2Tokenizer(vocab=vocab_scores, unk_token="<unk>", do_lower_case=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False))

        self.assertListEqual(tokens, tokens_target)


    @unittest.skip(reason="There is an inconsistency between slow and fast tokenizer due to a bug in the fast one.")
    def test_sentencepiece_tokenize_and_convert_tokens_to_string(self):
        pass

    @unittest.skip(reason="There is an inconsistency between slow and fast tokenizer due to a bug in the fast one.")
    def test_sentencepiece_tokenize_and_decode(self):
        pass

    def test_split_by_punct(self):
        # fmt: off
        sequence = "I was born in 92000, and this is falsé!"
        tokens_target = ["▁", "<unk>", "▁was", "▁born", "▁in", "▁9", "2000", "▁", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "▁", "!", ]
        # fmt: on

        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tokenizer = DebertaV2Tokenizer(vocab=vocab_scores, merges=merges, unk_token="<unk>", split_by_punct=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False))

        self.assertListEqual(tokens, tokens_target)

    def test_do_lower_case_split_by_punct(self):
        # fmt: off
        sequence = "I was born in 92000, and this is falsé!"
        tokens_target = ["▁i", "▁was", "▁born", "▁in", "▁9", "2000", "▁", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "▁", "!", ]
        # fmt: on

        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tokenizer = DebertaV2Tokenizer(vocab=vocab_scores, merges=merges, unk_token="<unk>", do_lower_case=True, split_by_punct=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False))
        self.assertListEqual(tokens, tokens_target)


    def test_do_lower_case_split_by_punct_false(self):
        # fmt: off
        sequence = "I was born in 92000, and this is falsé!"
        tokens_target = ["▁i", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "!", ]
        # fmt: on

        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tokenizer = DebertaV2Tokenizer(vocab=vocab_scores, merges=merges, unk_token="<unk>", do_lower_case=True, split_by_punct=False)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False))

        self.assertListEqual(tokens, tokens_target)

    def test_do_lower_case_false_split_by_punct(self):
        # fmt: off
        sequence = "I was born in 92000, and this is falsé!"
        tokens_target = ["▁", "<unk>", "▁was", "▁born", "▁in", "▁9", "2000", "▁", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "▁", "!", ]
        # fmt: on
        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tokenizer = DebertaV2Tokenizer(vocab=vocab_scores, merges=merges, unk_token="<unk>", do_lower_case=False, split_by_punct=True)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False))

        self.assertListEqual(tokens, tokens_target)

    def test_do_lower_case_false_split_by_punct_false(self):
        # fmt: off
        sequence = " \tHeLLo!how  \n Are yoU?  "
        tokens_target = ["▁", "<unk>", "e", "<unk>", "o", "!", "how", "▁", "<unk>", "re", "▁yo", "<unk>", "?"]
        # fmt: on
        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tokenizer = DebertaV2Tokenizer(vocab=vocab_scores, merges=merges, unk_token="<unk>", do_lower_case=False, split_by_punct=False)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False))

        self.assertListEqual(tokens, tokens_target)


    def test_full_tokenizer(self):
        sequence = "This is a test"
        ids_target = [13, 1, 4398, 25, 21, 1289]
        tokens_target = ["▁", "T", "his", "▁is", "▁a", "▁test"]
        back_tokens_target = ["▁", "<unk>", "his", "▁is", "▁a", "▁test"]
        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tokenizer = DebertaV2Tokenizer(vocab=vocab_scores, merges=merges, unk_token="<unk>", keep_accents=True)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, ids_target)
        tokens = tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, tokens_target)
        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(back_tokens, back_tokens_target)

        # fmt: off
        sequence = "I was born in 92000, and this is falsé!"
        ids_target = [13, 1, 23, 386, 19, 561, 3050, 15, 17, 48, 25, 8256, 18, 1, 187]
        tokens_target = ["▁", "I", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "é", "!", ]
        back_tokens_target = ["▁", "<unk>", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "!", ]
        # fmt: on

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, ids_target)
        tokens = tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, tokens_target)
        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(back_tokens, back_tokens_target)


    @slow
    def test_tokenizer_integration(self):
        expected_encoding = {'input_ids': [[1, 39867, 36, 19390, 486, 27, 35052, 81436, 18, 60685, 1225, 7, 35052, 81436, 18, 9367, 16899, 18, 15937, 53, 594, 773, 18, 16287, 30465, 36, 15937, 6, 41139, 38, 36979, 60763, 191, 6, 34132, 99, 6, 50538, 390, 43230, 6, 34132, 2779, 20850, 14, 699, 1072, 1194, 36, 382, 10901, 53, 7, 699, 1072, 2084, 36, 20422, 630, 53, 19, 105, 3049, 1896, 1053, 16899, 1506, 11, 37978, 4243, 7, 1237, 31869, 200, 16566, 654, 6, 35052, 81436, 7, 55630, 13593, 4, 2], [1, 26, 15011, 13, 667, 8, 1053, 18, 23611, 1237, 72356, 12820, 34, 104134, 1209, 35, 13313, 6627, 21, 202, 347, 7, 164, 2399, 11, 46, 4485, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 5, 1232, 2864, 15785, 14951, 105, 5, 8581, 1250, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}  # fmt: skip

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="microsoft/deberta-v2-xlarge",
            revision="ad6e42c1532ddf3a15c39246b63f5559d558b670",
        )
