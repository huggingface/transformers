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

from transformers import DebertaV2Tokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers
from transformers.tokenization_utils_sentencepiece import SentencePieceExtractor

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/spiece.model")


@require_sentencepiece
@require_tokenizers
class DebertaV2TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "microsoft/deberta-v2-xlarge"
    tokenizer_class = DebertaV2Tokenizer

    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 's', 'é', '.', '▁', '生', '活', '的', '真', '谛', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁<', 's', '>', '▁hi', '<', 's', '>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁Hello', '.', '▁But', '▁i', 'rd', '▁and', '▁', 'ป', 'ี', '▁i', 'rd', '▁', 'ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [69, 13, 10, 711, 112100, 16, 28, 1022, 11, 728, 16135, 6, 7, 32, 13, 46426, 12, 5155, 4, 250, 40289, 102080, 8593, 98226, 3, 29213, 2302, 4800, 2302, 4800, 4800, 2318, 12, 2259, 8133, 9475, 12, 2259, 7493, 23, 524, 3664, 146, 26, 2141, 23085, 43, 4800, 4, 167, 306, 1893, 7, 250, 86501, 70429, 306, 1893, 250, 51857, 4839, 100, 24, 17, 381]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 's', 'é', '.', '▁', '生', '活', '的', '真', '[UNK]', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁<', 's', '>', '▁hi', '<', 's', '>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁Hello', '.', '▁But', '▁i', 'rd', '▁and', '▁', 'ป', 'ี', '▁i', 'rd', '▁', 'ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊 I was born in 92000, and this is falsé. 生活的真[UNK]是 Hi Hello Hi Hello Hello <s> hi<s>there The following string should be properly encoded: Hello. But ird and ปี ird ด Hey how are you doing"

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
        tokenizer = DebertaV2Tokenizer(
            vocab=vocab_scores, merges=merges, unk_token="<unk>", do_lower_case=True, split_by_punct=True
        )
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False))
        self.assertListEqual(tokens, tokens_target)

    def test_do_lower_case_split_by_punct_false(self):
        # fmt: off
        sequence = "I was born in 92000, and this is falsé!"
        tokens_target = ["▁i", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "!", ]
        # fmt: on

        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tokenizer = DebertaV2Tokenizer(
            vocab=vocab_scores, merges=merges, unk_token="<unk>", do_lower_case=True, split_by_punct=False
        )
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False))

        self.assertListEqual(tokens, tokens_target)

    def test_do_lower_case_false_split_by_punct(self):
        # fmt: off
        sequence = "I was born in 92000, and this is falsé!"
        tokens_target = ["▁", "<unk>", "▁was", "▁born", "▁in", "▁9", "2000", "▁", ",", "▁and", "▁this", "▁is", "▁fal", "s", "<unk>", "▁", "!", ]
        # fmt: on
        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tokenizer = DebertaV2Tokenizer(
            vocab=vocab_scores, merges=merges, unk_token="<unk>", do_lower_case=False, split_by_punct=True
        )
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False))

        self.assertListEqual(tokens, tokens_target)

    def test_do_lower_case_false_split_by_punct_false(self):
        # fmt: off
        sequence = " \tHeLLo!how  \n Are yoU?  "
        tokens_target = ["▁", "<unk>", "e", "<unk>", "o", "!", "how", "▁", "<unk>", "re", "▁yo", "<unk>", "?"]
        # fmt: on
        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tokenizer = DebertaV2Tokenizer(
            vocab=vocab_scores, merges=merges, unk_token="<unk>", do_lower_case=False, split_by_punct=False
        )
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sequence, add_special_tokens=False))

        self.assertListEqual(tokens, tokens_target)

    def test_add_special_tokens_regression_issue_44568(self):
        """
        Regression test for GitHub issue #44568.

        Bug: After a save/load round-trip (which simulates from_pretrained()), the
        post_processor was being overwritten by update_post_processor() inside _post_init().
        Because DebertaV2Tokenizer does not use add_bos_token/add_eos_token, that call
        produced a no-op TemplateProcessing, silently stripping [CLS]/[SEP] from outputs.

        Fix: The TemplateProcessing post_processor is now built *before* super().__init__()
        and passed as a kwarg so the base class installs it and marks
        _should_update_post_processor=False, preventing any subsequent clobber.
        """
        import tempfile

        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tokenizer = DebertaV2Tokenizer(vocab=vocab_scores, unk_token="<unk>")

        # --- Direct instantiation ---
        encoding = tokenizer("Hello world", add_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        self.assertEqual(tokens[0], "[CLS]", "Direct init: first token should be [CLS]")
        self.assertEqual(tokens[-1], "[SEP]", "Direct init: last token should be [SEP]")

        # --- Save / load round-trip (mimics from_pretrained) ---
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            reloaded = DebertaV2Tokenizer.from_pretrained(tmp_dir)

        encoding_rt = reloaded("Hello world", add_special_tokens=True)
        tokens_rt = reloaded.convert_ids_to_tokens(encoding_rt["input_ids"])
        self.assertEqual(
            tokens_rt[0],
            "[CLS]",
            "After save/load: first token should be [CLS] (regression: issue #44568)",
        )
        self.assertEqual(
            tokens_rt[-1],
            "[SEP]",
            "After save/load: last token should be [SEP] (regression: issue #44568)",
        )

        # --- Pair encoding after round-trip ---
        encoding_pair = reloaded("Hello", "World", add_special_tokens=True)
        tokens_pair = reloaded.convert_ids_to_tokens(encoding_pair["input_ids"])
        self.assertEqual(tokens_pair[0], "[CLS]")
        sep_indices = [i for i, t in enumerate(tokens_pair) if t == "[SEP]"]
        self.assertEqual(len(sep_indices), 2, "Pair encoding should have two [SEP] tokens")
        self.assertEqual(sep_indices[-1], len(tokens_pair) - 1, "Last token should be [SEP]")

    def test_post_processor_adds_special_tokens(self):
        extractor = SentencePieceExtractor(SAMPLE_VOCAB)
        vocab, vocab_scores, merges = extractor.extract()
        tokenizer = DebertaV2Tokenizer(vocab=vocab_scores, unk_token="<unk>")

        encoding = tokenizer("Hello world")
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        self.assertEqual(tokens[0], "[CLS]")
        self.assertEqual(tokens[-1], "[SEP]")

        encoding_pair = tokenizer("Hello", "World")
        tokens_pair = tokenizer.convert_ids_to_tokens(encoding_pair["input_ids"])
        self.assertEqual(tokens_pair[0], "[CLS]")
        sep_indices = [i for i, t in enumerate(tokens_pair) if t == "[SEP]"]
        self.assertEqual(len(sep_indices), 2)
        self.assertEqual(sep_indices[-1], len(tokens_pair) - 1)
