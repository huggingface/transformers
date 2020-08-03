from .test_tokenization_common import TokenizerTesterMixin
import unittest

from .test_tokenization_marian import SAMPLE_SP

from pathlib import Path
from shutil import copyfile
FRAMEWORK = 'tf'
from transformers.tokenization_pegasus import PegasusTokenizer
class PegasusTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = PegasusTokenizer

    def setUp(self):
        super().setUp()
        #vocab = ["</s>", "<unk>", "▁This", "▁is", "▁a", "▁t", "est", "\u0120", "<pad>"]
        #vocab_tokens = dict(zip(vocab, range(len(vocab))))
        save_dir = Path(self.tmpdirname)
        spm_file = PegasusTokenizer.vocab_files_names["vocab_file"]
        #if not (save_dir / spm_file).exists():
        #copyfile(SAMPLE_SP, save_dir / spm_file)
        tokenizer = PegasusTokenizer(SAMPLE_SP)

        #
        #
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizer(self, **kwargs) -> PegasusTokenizer:
        return PegasusTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        return (
            "This is a test",
            "This is a test",
        )


    def test_full_tokenizer(self):
        tokenizer = PegasusTokenizer.from_pretrained('sshleifer/pegasus')
        assert tokenizer.vocab_size == 96103
