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
        return PegasusTokenizer.from_pretrained("google/pegasus-large")

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

    def test_pegasus_large_tokenizer_settings(self):
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
        # The tracebacks for the following asserts are **better** without messages or self.assertEqual
        assert tokenizer.vocab_size == 96103
        assert tokenizer.pad_token_id == 0
        assert tokenizer.eos_token_id == 1
        assert tokenizer.offset == 103
        assert tokenizer.unk_token_id == tokenizer.offset + 2 == 105
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.mask_token is None
        assert tokenizer.mask_token_id is None
        raw_input_str = "To ensure a smooth flow of bank resolutions."
        desired_result = [413, 615, 114, 2291, 1971, 113, 1679, 10710, 107, 1]
        ids = tokenizer([raw_input_str], return_tensors=None).input_ids[0]
        self.assertListEqual(desired_result, ids)
        assert tokenizer.convert_ids_to_tokens([0, 1, 2]) == ["<pad>", "</s>", "unk_2"]
