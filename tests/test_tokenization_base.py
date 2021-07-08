import unittest

from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert_japanese.tokenization_bert_japanese import BertJapaneseTokenizer


class ClassMismatchTest(unittest.TestCase):
    def test_mismatch_error(self):
        PRETRAINED_MODEL = "cl-tohoku/bert-base-japanese"
        with self.assertRaises(ValueError):
            BertTokenizer.from_pretrained(PRETRAINED_MODEL)

    def test_limit_of_match_validation(self):
        # Can't detect mismatch because this model's config
        # doesn't have information about the tokenizer model.
        PRETRAINED_MODEL = "bert-base-uncased"
        BertJapaneseTokenizer.from_pretrained(PRETRAINED_MODEL)
