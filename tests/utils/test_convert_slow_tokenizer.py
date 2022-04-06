import unittest
import warnings
from dataclasses import dataclass

from transformers import logging
from transformers.convert_slow_tokenizer import SpmConverter
from transformers.testing_utils import CaptureLogger, get_tests_dir


@dataclass
class FakeOriginalTokenizer:
    vocab_file: str


class ConvertSlowTokenizerTest(unittest.TestCase):
    def test_spm_converter_bytefallback_warning(self):
        spm_model_file_without_bytefallback = f"{get_tests_dir()}/fixtures/test_sentencepiece.model"
        spm_model_file_with_bytefallback = f"{get_tests_dir()}/fixtures/test_sentencepiece_with_bytefallback.model"

        original_tokenizer_without_bytefallback = FakeOriginalTokenizer(vocab_file=spm_model_file_without_bytefallback)

        logging.set_verbosity_warning()
        logger = logging.get_logger("transformers.convert_slow_tokenizer")

        with CaptureLogger(logger) as cl:
            _ = SpmConverter(original_tokenizer_without_bytefallback)
        self.assertNotIn(
            (
                "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
                " which is not implemented in the fast tokenizers."
            ),
            cl.out,
        )

        original_tokenizer_with_bytefallback = FakeOriginalTokenizer(vocab_file=spm_model_file_with_bytefallback)

        with CaptureLogger(logger) as cl:
            _ = SpmConverter(original_tokenizer_with_bytefallback)
        self.assertIn(
            (
                "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
                " which is not implemented in the fast tokenizers."
            ),
            cl.out,
        )
