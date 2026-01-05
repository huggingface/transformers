import unittest

from transformers import GemmaTokenizerFast


class GemmaTokenizerFastTest(unittest.TestCase):
    def test_backend_has_no_pre_tokenizer(self):
        tok = GemmaTokenizerFast.from_pretrained("hf-internal-testing/dummy-gemma")
        self.assertIsNone(tok.backend_tokenizer.pre_tokenizer)
