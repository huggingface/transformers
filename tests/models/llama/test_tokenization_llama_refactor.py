import pytest
import unittest
import tempfile
import shutil

from transformers import AutoTokenizer, AddedToken, PreTrainedTokenizerFast
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.create_fast_tokenizer import SentencePieceExtractor
from transformers.testing_utils import (
    require_sentencepiece, 
    require_tokenizers, 
    require_tiktoken, 
    require_read_token,
    get_tests_dir
)
from tests.test_tokenization_common import TokenizerTesterMixin


# Master input string of combined test cases
input_string = """This is a test
I was born in 92000, and this is falsé.
生活的真谛是
Hi  Hello
Hi   Hello

 
  
 Hello
<s>
hi<s>there
The following string should be properly encoded: Hello.
But ird and ปี   ird   ด
Hey how are you doing"""


expected_tokens = ['▁This', '▁is', '▁a', '▁test', '<0x0A>', 'I', '▁was', '▁born', '▁in', '▁', '9', '2', '0', '0', '0', ',', '▁and', '▁this', '▁is', '▁f', 'als', 'é', '.', '<0x0A>', '生', '活', '的', '真', '<0xE8>', '<0xB0>', '<0x9B>', '是', '<0x0A>', 'Hi', '▁', '▁Hello', '<0x0A>', 'Hi', '▁▁', '▁Hello', '<0x0A>', '<0x0A>', '▁', '<0x0A>', '▁▁', '<0x0A>', '▁Hello', '<0x0A>', '<s>', '<0x0A>', 'hi', '<s>', 'there', '<0x0A>', 'The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁Hello', '.', '<0x0A>', 'But', '▁', 'ird', '▁and', '▁', 'ป', 'ี', '▁▁▁', 'ird', '▁▁▁', 'ด', '<0x0A>', 'H', 'ey', '▁how', '▁are', '▁you', '▁doing']
expected_token_ids = [1, 910, 338, 263, 1243, 13, 29902, 471, 6345, 297, 29871, 29929, 29906, 29900, 29900, 29900, 29892, 322, 445, 338, 285, 1338, 29948, 29889, 13, 30486, 31704, 30210, 30848, 235, 179, 158, 30392, 13, 18567, 29871, 15043, 13, 18567, 259, 15043, 13, 13, 29871, 13, 259, 13, 15043, 13, 1, 13, 2918, 1, 12711, 13, 1576, 1494, 1347, 881, 367, 6284, 18511, 29901, 15043, 29889, 13, 6246, 29871, 1823, 322, 29871, 31010, 30691, 1678, 1823, 1678, 30718, 13, 29950, 1032, 920, 526, 366, 2599]


@require_sentencepiece
@require_tokenizers
class LlamaTokenizationRefactorTest(TokenizerTesterMixin, unittest.TestCase):
    # TokenizerTesterMixin configuration
    from_pretrained_id = ["hf-internal-testing/llama-tokenizer"]
    tokenizer_class = LlamaTokenizerFast  # We'll set this dynamically
    rust_tokenizer_class = LlamaTokenizerFast
    test_rust_tokenizer = False
    test_sentencepiece = True
    from_pretrained_kwargs = {}
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "hf-internal-testing/llama-tokenizer"
        
        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.pad_token = tok_auto.eos_token
        tok_auto.save_pretrained(cls.tmpdirname)

        # Build backend for slow tokenizer from the fast tokenizer's SentencePiece model
        vocab_file = getattr(tok_auto, "vocab_file", None)

        extractor = SentencePieceExtractor(vocab_file)
        vocab, merges = extractor.extract()
        tok_from_vocab = LlamaTokenizerFast(vocab=vocab, merges=merges)
        tok_from_vocab.pad_token = tok_from_vocab.eos_token

        cls.tokenizers = [tok_auto, tok_from_vocab]

    def get_tokenizers(self, **kwargs):
        kwargs.update({"pad_token": "<PAD>"})
        return super().get_tokenizers(**kwargs)

    def test_llama_tokenizers_match_expected_tokens(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.tokenize(input_string), expected_tokens)

    def test_llama_tokenizers_match_expected_ids(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.encode(input_string), expected_token_ids)

    def test_save_and_reload(self):
        for tok in self.tokenizers:
            with self.subTest(f"{tok.__class__.__name__}"):
                original_tokens = tok.tokenize(input_string)
                original_ids = tok.encode(input_string)
                
                # Save tokenizer to temporary directory
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tok.save_pretrained(tmp_dir)
                    
                    # Reload tokenizer from saved directory
                    reloaded_tok = tok.__class__.from_pretrained(tmp_dir)
                    
                    # Test that reloaded tokenizer produces same results
                    reloaded_tokens = reloaded_tok.tokenize(input_string)
                    reloaded_ids = reloaded_tok.encode(input_string)
                    
                    self.assertEqual(original_tokens, reloaded_tokens)
                    self.assertEqual(original_ids, reloaded_ids)
