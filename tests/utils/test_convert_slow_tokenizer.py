import json
import tempfile
import unittest
import warnings
from dataclasses import dataclass

from transformers import AutoTokenizer, TokenizersBackend
from transformers.convert_slow_tokenizer import SentencePieceExtractor, SpmConverter
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers
from transformers.utils import is_sentencepiece_available


if is_sentencepiece_available():
    import sentencepiece as spm


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


@require_sentencepiece
@require_tokenizers
class SentencePieceModelConversionTest(unittest.TestCase):
    """`TokenizersBackend` built from a bare `tokenizer.model` should encode and decode like SentencePiece."""

    corpus = [
        "Hello world, this is a test of the tokenizer.",
        "    def f(x):\n        return x + 1  # comment",
        "数字 12345 和中文文本，还有更多的中文。",
        "foo_bar__baz qux",
    ] * 50
    texts = [
        "Hello  world\n__x 12<reserve_0>3",
        " leading space",
        "trailing   ",
        "\n\n    indented\tcode",
        "中文 text 123",
        "__init__",
        "🤗 unseen characters",
        "",
    ]

    def _train(self, tmp_dir, **kwargs):
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(self.corpus),
            model_prefix=f"{tmp_dir}/tokenizer",
            vocab_size=400,
            model_type="bpe",
            byte_fallback=True,
            normalization_rule_name="identity",
            minloglevel=2,
            **kwargs,
        )
        return spm.SentencePieceProcessor(model_file=f"{tmp_dir}/tokenizer.model")

    def test_matches_sentencepiece(self):
        configs = {
            # User-defined pieces are never merged, and with a dummy prefix every chunk starts with "▁"
            "dummy_prefix_user_defined": {
                "add_dummy_prefix": True,
                "remove_extra_whitespaces": False,
                "allow_whitespace_only_pieces": True,
                "split_digits": True,
                "user_defined_symbols": ["__", "<reserve_0>", "\n"],
            },
            "no_dummy_prefix_user_defined": {
                "add_dummy_prefix": False,
                "remove_extra_whitespaces": False,
                "user_defined_symbols": ["__", "<reserve_0>", "\n"],
            },
            "remove_extra_whitespaces": {"add_dummy_prefix": True, "remove_extra_whitespaces": True},
        }
        for name, kwargs in configs.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp_dir:
                sp = self._train(tmp_dir, **kwargs)
                tokenizer = TokenizersBackend.from_pretrained(tmp_dir)
                for text in self.texts:
                    ids = sp.encode(text)
                    self.assertEqual(tokenizer.encode(text, add_special_tokens=False), ids, text)
                    self.assertEqual(tokenizer.decode(ids), sp.decode(ids), text)

    def test_added_tokens_decoder_can_swap_pieces(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            sp = self._train(tmp_dir, control_symbols=["<a>", "<b>"])
            a_id, b_id = sp.piece_to_id("<a>"), sp.piece_to_id("<b>")
            added_tokens_decoder = {
                str(a_id): {"content": "<b>", "special": True},
                str(b_id): {"content": "<a>", "special": True},
            }
            with open(f"{tmp_dir}/tokenizer_config.json", "w", encoding="utf-8") as f:
                json.dump({"added_tokens_decoder": added_tokens_decoder}, f)

            tokenizer = TokenizersBackend.from_pretrained(tmp_dir)
            self.assertEqual(tokenizer.convert_tokens_to_ids(["<a>", "<b>"]), [b_id, a_id])
            self.assertEqual(tokenizer.encode("<a>x<b>", add_special_tokens=False)[::2], [b_id, a_id])
