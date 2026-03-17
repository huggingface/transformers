import unittest

from transformers.models.camembert.tokenization_camembert import CamembertTokenizer
from transformers.testing_utils import require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class CamembertTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = ["almanach/camembert-base"]
    tokenizer_class = CamembertTokenizer

    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁', 'born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 'sé', '.', '▁', '生活的真谛是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', '▁the', 're', '▁The', '▁', 'follow', 'ing', '▁string', '▁s', 'h', 'ould', '▁be', '▁pro', 'per', 'ly', '▁en', 'code', 'd', ':', '▁Hello', '.', '▁But', '▁i', 'rd', '▁and', '▁', 'ปี', '▁i', 'rd', '▁', 'ด', '▁Hey', '▁h', 'ow', '▁are', '▁you', '▁do', 'ing']  # fmt: skip
    integration_expected_token_ids = [17526, 2856, 33, 2006, 21, 3, 551, 15760, 21, 24900, 378, 419, 13233, 7, 1168, 9098, 2856, 19289, 5100, 9, 21, 3, 5108, 9774, 5108, 9774, 9774, 5, 7874, 5, 808, 346, 908, 21, 31189, 402, 20468, 52, 133, 19306, 2446, 909, 1399, 1107, 22, 14420, 204, 92, 9774, 9, 10503, 1723, 6682, 1168, 21, 3, 1723, 6682, 21, 3, 20128, 616, 3168, 9581, 4835, 7503, 402]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁', '<unk>', '▁I', '▁was', '▁', 'born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 'sé', '.', '▁', '<unk>', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', '▁the', 're', '▁The', '▁', 'follow', 'ing', '▁string', '▁s', 'h', 'ould', '▁be', '▁pro', 'per', 'ly', '▁en', 'code', 'd', ':', '▁Hello', '.', '▁But', '▁i', 'rd', '▁and', '▁', '<unk>', '▁i', 'rd', '▁', '<unk>', '▁Hey', '▁h', 'ow', '▁are', '▁you', '▁do', 'ing']  # fmt: skip
    integration_expected_decoded_text = "This is a test <unk> I was born in 92000, and this is falsé. <unk> Hi Hello Hi Hello Hello<s> hi<s> there The following string should be properly encoded: Hello. But ird and <unk> ird <unk> Hey how are you doing"


@require_tokenizers
class CamembertDictVocabTest(unittest.TestCase):
    """Test CamembertTokenizer with dict vocab (from tokenizer.json BPE format)."""

    def test_camembert_tokenizer_with_dict_vocab(self):
        """Vocab from tokenizer.json is dict {token: id}; Camembert expects list of (token, score)."""
        vocab = {
            "<s>NOTUSED": 0,
            "<pad>": 1,
            "</s>NOTUSED": 2,
            "<unk>": 3,
            "<unk>NOTUSED": 4,
            "<mask>": 5,
            "▁hello": 6,
            "▁world": 7,
        }
        tokenizer = CamembertTokenizer(vocab=vocab)
        self.assertEqual(tokenizer.vocab_size, 8)
        result = tokenizer("hello world", add_special_tokens=False)
        self.assertGreater(len(result["input_ids"]), 0)
