import pytest
import unittest

from transformers import AutoTokenizer
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from transformers.create_fast_tokenizer import SentencePieceExtractor
from transformers.testing_utils import require_sentencepiece, require_tokenizers


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
class LlamaTokenizationRefactorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tok_auto = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

        # Build backend for slow tokenizer from the fast tokenizer's SentencePiece model
        vocab_file = getattr(tok_auto, "vocab_file", None)

        extractor = SentencePieceExtractor(vocab_file)
        vocab, merges = extractor.extract()

        tok_from_vocab = LlamaTokenizer(vocab=vocab, merges=merges)

        cls.tokenizers = [tok_auto, tok_from_vocab]

    def test_llama_tokenizers_match_expected_tokens(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.tokenize(input_string), expected_tokens)

    def test_llama_tokenizers_match_expected_ids(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.encode(input_string), expected_token_ids)
