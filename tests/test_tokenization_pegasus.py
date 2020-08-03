import unittest
from pathlib import Path
from shutil import copyfile

from transformers.tokenization_pegasus import PegasusTokenizer

from .test_tokenization_common import TokenizerTesterMixin
from .test_tokenization_marian import SAMPLE_SP


# SAMPLE_SP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/pegasus_sentencepiece.model")
FRAMEWORK = "tf"

from transformers.file_utils import cached_property

class PegasusTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = PegasusTokenizer

    def setUp(self):
        super().setUp()
        # vocab = ["</s>", "<unk>", "▁This", "▁is", "▁a", "▁t", "est", "\u0120", "<pad>"]
        # vocab_tokens = dict(zip(vocab, range(len(vocab))))
        save_dir = Path(self.tmpdirname)
        spm_file = PegasusTokenizer.vocab_files_names["vocab_file"]
        if not (save_dir / spm_file).exists():
            tokenizer = self.default_tokenizer
            tokenizer.save_pretrained(self.tmpdirname)
        # copyfile(SAMPLE_SP, save_dir / spm_file)
        # tokenizer = PegasusTokenizer(SAMPLE_SP)



        #
        #


    @cached_property
    def default_tokenizer(self):
        return PegasusTokenizer.from_pretrained('sshleifer/pegasus')

    @unittest.skip("add_tokens does not work yet")
    def test_swap_special_token(self):
        pass

    def get_tokenizer(self, **kwargs) -> PegasusTokenizer:
        if not kwargs:
            return self.default_tokenizer
        else:
            return PegasusTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        return (
            "This is a test",
            "This is a test",
        )

    def test_full_tokenizer(self):
        tokenizer = PegasusTokenizer.from_pretrained("sshleifer/pegasus")
        assert tokenizer.vocab_size == 96103
