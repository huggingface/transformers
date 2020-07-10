import json
import os
import unittest

from transformers.testing_utils import slow
from transformers.tokenization_blenderbot import VOCAB_FILES_NAMES, BlenderbotTokenizer, BlenderbotSmallTokenizer

from .test_tokenization_common import TokenizerTesterMixin

class BlenderbotSmallTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    
    tokenizer_class = BlenderbotSmallTokenizer

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = "/home/mariama/data/models/blender/blender_90M/vocab-90M.json"
        merges = "/home/mariama/data/models/blender/blender_90M/merges.txt"
        with open(vocab) as f:
            vocab_tokens = json.load(f)
        with open(merges) as f1:
            merges = f1.readlines()
        # merges = ["#version: 0.2", "a p", "ap t</w>", "r e", "a d", "ad apt</w>", ""]
        self.special_tokens_map = {"bos_token": "__start", "eos_token": "__end__", "pad_token": "__null__", "unk_token": "__unk__"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return BlenderbotSmallTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "adapt react readapt apt"
        output_text = "adapt react readapt apt"
        return input_text, output_text
    
    def test_full_blenderbot_small_tokenizer(self):
        tokenizer = BlenderbotSmallTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "adapt react readapt apt"
        bpe_tokens = ['adapt', 'react', 'read@@', 'ap@@', 't', 'ap@@', 't']
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = [tokenizer.bos_token] + tokens + [tokenizer.eos_token]
        print(input_tokens)

        # input_bpe_tokens = [0, 1, 2, 4, 5, 1, 0, 3, 6]
        # self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)
