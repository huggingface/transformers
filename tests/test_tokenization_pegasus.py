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
        raw_input_str = (
            "To ensure a smooth flow of bank resolutions to the necessary signatories, "
            "I am requesting that Enron Treasury first route the bank resolutions to Angela Davis "
            "(EWS Legal) to be initialed before being routed to John Lavorato or Louise Kitchen.\n"
            "If you have any questions please call me at 3-6544."
            "Thank you for your attention to this matter."
        )
        desired_result = [413, 615, 114, 2291, 1971, 113, 1679, 10710, 112,
                          109, 993, 67158, 108, 125, 346, 11518, 120, 81991,
                          12596, 211, 2610, 109, 1679, 10710, 112, 14058, 5503,
                          143, 76757, 6797, 158, 112, 129, 2061, 316, 269,
                          270, 33259, 112, 1084, 33485, 11661, 497, 132, 17542,
                          3549, 107, 240, 119, 133, 189, 574, 528, 443,
                          213, 134, 33625, 43573, 107, 2556, 119, 118, 128,
                          1090, 112, 136, 841, 107, 1]
        ids = tokenizer([raw_input_str], return_tensors=None).input_ids[0]
        self.assertListEqual(desired_result, ids)
