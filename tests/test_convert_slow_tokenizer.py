import unittest
from dataclasses import dataclass

from transformers.convert_slow_tokenizer import SpmConverter
from transformers.testing_utils import get_tests_dir


@dataclass
class FakeOriginalTokenizer:
    vocab_file: str


class ConvertSlowTokenizerTest(unittest.TestCase):
    def test_spm_converter_bytefallback_warning(self):
        spm_model_file_without_bytefallback = f"{get_tests_dir()}/fixtures/test_sentencepiece.model"
        spm_model_file_with_bytefallback = f"{get_tests_dir()}/fixtures/test_sentencepiece_with_bytefallback.model"

        original_tokenizer_without_bytefallback = FakeOriginalTokenizer(vocab_file=spm_model_file_without_bytefallback)

        with self.assertLogs(level="WARNING") as captured:
            _ = SpmConverter(original_tokenizer_without_bytefallback)
        print(captured)

        original_tokenizer_with_bytefallback = FakeOriginalTokenizer(vocab_file=spm_model_file_with_bytefallback)

        with self.assertLogs(level="WARNING") as captured:
            _ = SpmConverter(original_tokenizer_with_bytefallback)
        print(captured)
