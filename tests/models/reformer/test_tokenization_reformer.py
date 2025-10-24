import tempfile
import unittest

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers import AutoTokenizer
from transformers.models.reformer.tokenization_reformer import ReformerTokenizer
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
)
from transformers.tokenization_sentencepiece import SentencePieceExtractor


@require_sentencepiece
@require_tokenizers
class ReformerTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    # TokenizerTesterMixin configuration
    from_pretrained_id = ["google/reformer-crime-and-punishment"]
    tokenizer_class = ReformerTokenizer  # We'll set this dynamically
    test_sentencepiece = True
    from_pretrained_kwargs = {}

    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = ['▁T', 'h', 'is', '▁is', '▁a', '▁t', 'est', '▁I', '▁was', '▁b', 'or', 'n', '▁in', '▁', '<unk>', ',', '▁and', '▁this', '▁is', '▁f', 'al', 's', '<unk>', '.', '▁', '<unk>', '▁H', 'i', '▁He', 'll', 'o', '▁H', 'i', '▁He', 'll', 'o', '▁He', 'll', 'o', '▁', '<unk>', 's', '<unk>', '▁h', 'i', '<unk>', 's', '<unk>', 't', 'he', 're', '▁The', '▁f', 'o', 'll', 'ow', 'ing', '▁st', 'r', 'ing', '▁sh', 'ould', '▁be', '▁p', 'ro', 'p', 'er', 'ly', '▁', 'en', 'c', 'od', 'ed', ':', '▁He', 'll', 'o', '.', '▁But', '▁', 'ir', 'd', '▁and', '▁', '<unk>', '▁', 'ir', 'd', '▁', '<unk>', '▁He', 'y', '▁h', 'ow', '▁are', '▁you', '▁do', 'ing']
    integration_expected_token_ids = [108, 265, 24, 111, 4, 3, 249, 33, 59, 17, 38, 263, 39, 258, 0, 277, 27, 221, 111, 22, 94, 266, 0, 278, 258, 0, 96, 264, 126, 32, 262, 96, 264, 126, 32, 262, 126, 32, 262, 258, 0, 266, 0, 31, 264, 0, 266, 0, 260, 5, 10, 140, 22, 262, 32, 77, 20, 74, 267, 20, 168, 106, 49, 40, 186, 279, 16, 48, 258, 25, 274, 227, 19, 315, 126, 32, 262, 278, 231, 258, 91, 268, 27, 258, 0, 258, 91, 268, 258, 0, 126, 272, 31, 77, 157, 41, 137, 20]
    integration_expected_decoded_text = 'This is a test <unk> I was born in <unk>, and this is fals<unk>. <unk> Hi Hello Hi Hello Hello <unk>s<unk> hi<unk>s<unk>there The following string should be properly encoded: Hello. But ird and <unk> ird <unk> Hey how are you doing'
    integration_expected_text_from_tokens = 'This is a test <unk> I was born in <unk>, and this is fals<unk>. <unk> Hi Hello Hi Hello Hello <unk>s<unk> hi<unk>s<unk>there The following string should be properly encoded: Hello. But ird and <unk> ird <unk> Hey how are you doing'
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "google/reformer-crime-and-punishment"

        tokenizer = ReformerTokenizer.from_pretrained(from_pretrained_id)
        tokenizer.save_pretrained(cls.tmpdirname)

        # Build backend for slow tokenizer from the fast tokenizer's SentencePiece model
        vocab_file = getattr(tokenizer, "vocab_file", None)

        extractor = SentencePieceExtractor(vocab_file)
        vocab_ids, vocab_scores, merges = extractor.extract()
        tokenizer_from_vocab = ReformerTokenizer(vocab=vocab_ids, merges=merges)

        cls.tokenizers = [tokenizer, tokenizer_from_vocab]
