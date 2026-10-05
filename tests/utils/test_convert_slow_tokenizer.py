from unittest.mock import MagicMock
import unittest
import warnings
from dataclasses import dataclass

from transformers.convert_slow_tokenizer import SpmConverter
from transformers.testing_utils import get_tests_dir, require_sentencepiece


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

    @require_sentencepiece
    def test_spm_extractor_empty_precompiled_charsmap_returns_none(self):
        """SentencePieceExtractor must return None when protobuf yields b'' for charsmap.
        Regression for issue #48942.
        """
        from transformers.convert_slow_tokenizer import SentencePieceExtractor
        from tokenizers.models import Unigram

        spm_file = get_tests_dir("fixtures/test_sentencepiece.model")
        extractor = SentencePieceExtractor(model=spm_file)

        mock_proto = MagicMock()
        mock_proto.normalizer_spec.precompiled_charsmap = b""
        mock_proto.pieces = extractor.proto.pieces
        mock_proto.trainer_spec.unk_id = extractor.proto.trainer_spec.unk_id
        mock_proto.trainer_spec.model_type = extractor.proto.trainer_spec.model_type

        original_proto = extractor.proto
        extractor.proto = mock_proto
        try:
            result = extractor.extract(model_type=Unigram)
        finally:
            extractor.proto = original_proto

        self.assertIsNone(
            result.get("_spm_precompiled_charsmap"),
            "Empty protobuf bytes b'' must be normalised to None, not passed to Precompiled().",
        )

    @require_sentencepiece
    def test_spm_extractor_nonempty_precompiled_charsmap_is_preserved(self):
        """A genuine charsmap must be forwarded unchanged."""
        from transformers.convert_slow_tokenizer import SentencePieceExtractor
        from tokenizers.models import Unigram

        spm_file = get_tests_dir("fixtures/test_sentencepiece.model")
        extractor = SentencePieceExtractor(model=spm_file)

        fake_charsmap = b"\x00\x01\x02\x03"
        mock_proto = MagicMock()
        mock_proto.normalizer_spec.precompiled_charsmap = fake_charsmap
        mock_proto.pieces = extractor.proto.pieces
        mock_proto.trainer_spec.unk_id = extractor.proto.trainer_spec.unk_id
        mock_proto.trainer_spec.model_type = extractor.proto.trainer_spec.model_type

        original_proto = extractor.proto
        extractor.proto = mock_proto
        try:
            result = extractor.extract(model_type=Unigram)
        finally:
            extractor.proto = original_proto

        self.assertEqual(result.get("_spm_precompiled_charsmap"), fake_charsmap)