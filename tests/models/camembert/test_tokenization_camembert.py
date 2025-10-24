import unittest
import tempfile
from transformers import AutoTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin
from transformers.models.camembert.tokenization_camembert import CamembertTokenizer
from transformers.tokenization_sentencepiece import SentencePieceExtractor




@require_tokenizers
class CamembertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = ["almanach/camembert-base"]
    tokenizer_class = CamembertTokenizer
    test_sentencepiece = True
    test_sentencepiece_ignore_case = True


    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁I', '▁was', '▁', 'born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 'sé', '.', '▁', '生活的真谛是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', '▁the', 're', '▁The', '▁', 'follow', 'ing', '▁string', '▁s', 'h', 'ould', '▁be', '▁pro', 'per', 'ly', '▁en', 'code', 'd', ':', '▁Hello', '.', '▁But', '▁i', 'rd', '▁and', '▁', 'ปี', '▁i', 'rd', '▁', 'ด', '▁Hey', '▁h', 'ow', '▁are', '▁you', '▁do', 'ing']
    integration_expected_token_ids = [5, 17526, 2856, 33, 2006, 551, 15760, 21, 24900, 378, 419, 13233, 7, 1168, 9098, 2856, 19289, 5100, 9, 21, 3, 5108, 9774, 5108, 9774, 9774, 5, 7874, 5, 808, 346, 908, 21, 31189, 402, 20468, 52, 133, 19306, 2446, 909, 1399, 1107, 22, 14420, 204, 92, 9774, 9, 10503, 1723, 6682, 1168, 21, 3, 1723, 6682, 21, 3, 20128, 616, 3168, 9581, 4835, 7503, 402, 6]
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from_pretrained_id = "almanach/camembert-base"
        tokenizer = CamembertTokenizer.from_pretrained(from_pretrained_id)
        tokenizer.save_pretrained(cls.tmpdirname)
        

        #Build backend for slow tokenizer from the fast tokenizer's SentencePiece model
        vocab_file = getattr(tokenizer, "vocab_file", None)

        extractor = SentencePieceExtractor(vocab_file)
        vocab, vocab_scores, merges = extractor.extract()
        
        tokenizer_from_vocab = CamembertTokenizer(vocab=vocab_scores)
        tokenizer_from_vocab.pad_token = tokenizer_from_vocab.eos_token

        cls.tokenizers = [tokenizer, tokenizer_from_vocab]
