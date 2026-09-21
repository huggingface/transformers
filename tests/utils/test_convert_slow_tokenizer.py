import unittest
import warnings
from dataclasses import dataclass

from transformers.convert_slow_tokenizer import SpmConverter
from transformers.testing_utils import get_tests_dir


@dataclass
class FakeOriginalTokenizer:
    vocab_file: str


class ConvertSlowTokenizerTest(unittest.TestCase):
    def test_spm_converter_bytefallback_warning(self):
        spm_model_file_without_bytefallback = get_tests_dir("fixtures/test_sentencepiece.model")
        spm_model_file_with_bytefallback = get_tests_dir("fixtures/test_sentencepiece_with_bytefallback.model")

        original_tokenizer_without_bytefallback = FakeOriginalTokenizer(vocab_file=spm_model_file_without_bytefallback)

        with warnings.catch_warnings(record=True) as w:
            _ = SpmConverter(original_tokenizer_without_bytefallback)
        # We are looking for if there is any `UserWarning` with
        # `The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers.`
        w = [x for x in w if x.category.__name__ != "DeprecationWarning"]
        self.assertEqual(len(w), 0)

        original_tokenizer_with_bytefallback = FakeOriginalTokenizer(vocab_file=spm_model_file_with_bytefallback)

        with warnings.catch_warnings(record=True) as w:
            _ = SpmConverter(original_tokenizer_with_bytefallback)
        w = [x for x in w if x.category.__name__ != "DeprecationWarning"]
        self.assertEqual(len(w), 1)

        self.assertIn(
            "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
            " which is not implemented in the fast tokenizers.",
            str(w[0].message),
        )

    def test_spm_extractor_empty_precompiled_charsmap(self):
        from transformers.convert_slow_tokenizer import SentencePieceExtractor
        from tokenizers.models import Unigram
        spm_model_file = get_tests_dir("fixtures/test_sentencepiece.model")
        extractor = SentencePieceExtractor(spm_model_file)
        extractor.proto.normalizer_spec.precompiled_charsmap = b""
        kwargs = extractor.extract(model_type=Unigram)
        self.assertIsNone(kwargs.get("_spm_precompiled_charsmap"))



