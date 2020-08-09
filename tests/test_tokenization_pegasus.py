import unittest
from pathlib import Path

from transformers.file_utils import cached_property
from transformers.tokenization_pegasus import PegasusTokenizer

from .test_tokenization_common import TokenizerTesterMixin


class PegasusTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = PegasusTokenizer

    def setUp(self):
        super().setUp()

        save_dir = Path(self.tmpdirname)
        spm_file = PegasusTokenizer.vocab_files_names["vocab_file"]
        if not (save_dir / spm_file).exists():
            tokenizer = self.default_tokenizer
            tokenizer.save_pretrained(self.tmpdirname)

    @cached_property
    def default_tokenizer(self):
        return PegasusTokenizer.from_pretrained("sshleifer/pegasus")

    @unittest.skip("add_tokens does not work yet")
    def test_swap_special_token(self):
        pass

    def get_tokenizer(self, **kwargs) -> PegasusTokenizer:
        if not kwargs:
            return self.default_tokenizer
        else:
            return PegasusTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        return ("This is a test", "This is a test")

    def test_full_tokenizer(self):
        tokenizer = PegasusTokenizer.from_pretrained("sshleifer/pegasus")
        assert tokenizer.vocab_size == 96103
        raw_input_str = "To ensure a smooth flow of bank resolutions."
        desired_result = [413, 615, 114, 2291, 1971, 113, 1679, 10710, 107, 1]
        ids = tokenizer([raw_input_str], return_tensors=None).input_ids[0]
        self.assertListEqual(desired_result, ids)
