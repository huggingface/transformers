import json
import tempfile
import unittest
import warnings
from dataclasses import dataclass

from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import SentencePieceExtractor, SpmConverter
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

    def test_spm_converter_bpe_model(self):
        # Regression test: converting a BPE-trained SentencePiece model through
        # `SpmConverter.tokenizer` used to crash with
        # `AttributeError: 'list' object has no attribute '__name__'` because the
        # BPE branch called `SentencePieceExtractor.extract` with a stale signature.
        import sentencepiece as spm

        spm_model_file = get_tests_dir("fixtures/test_sentencepiece_bpe.model")

        original_tokenizer = FakeOriginalTokenizer(vocab_file=spm_model_file)
        tokenizer = SpmConverter(original_tokenizer).converted()

        # The converted vocab must carry every piece from the SentencePiece model
        # at its exact id, and merges must be generated from the piece scores.
        extractor = SentencePieceExtractor(spm_model_file)
        for piece_id, piece in enumerate(extractor.proto.pieces):
            self.assertEqual(tokenizer.id_to_token(piece_id), piece.piece)
        merges = json.loads(tokenizer.to_str())["model"]["merges"]
        self.assertGreater(len(merges), 0)

        # The converted tokenizer must encode exactly like SentencePiece itself.
        sp = spm.SentencePieceProcessor(model_file=spm_model_file)
        for text in [
            "the quick brown fox jumps over the lazy dog",
            "sentence piece bpe conversion test",
            "hello world this is a longer representative input for tokenization",
            "tokenization",
        ]:
            self.assertEqual(tokenizer.encode(text, add_special_tokens=False).ids, sp.encode(text))

    def test_spm_precompiled_charsmap_empty_is_none(self):
        # If the `precompiled_charsmap` is empty (`b""`), it should be converted to `None` and complete conversion successfully.
        spm_model_file = get_tests_dir("fixtures/test_sentencepiece.model")
        extractor = SentencePieceExtractor(spm_model_file)
        extractor.proto.normalizer_spec.precompiled_charsmap = b""
        kwargs = extractor.extract(model_type=None)
        self.assertIsNone(kwargs["_spm_precompiled_charsmap"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(f"{tmp_dir}/spiece.model", "wb") as f:
                f.write(extractor.proto.SerializeToString())
            with open(f"{tmp_dir}/config.json", "w", encoding="utf-8") as f:
                json.dump({"model_type": "t5"}, f)

            tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            self.assertGreater(len(tokenizer("Hello, world!")["input_ids"]), 1)
