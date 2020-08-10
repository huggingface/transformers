import json
import os
import unittest

from transformers.testing_utils import slow
from transformers.tokenization_blenderbot import VOCAB_FILES_NAMES, BlenderbotSmallTokenizer, BlenderbotTokenizer

from .test_tokenization_common import TokenizerTesterMixin


class BlenderbotSmallTokenizerTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = BlenderbotSmallTokenizer

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = ["__start__", "adapt", "act", "ap@@", "te", "__end__", "__unk__"]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
      
        merges = ["#version: 0.2", "a p", "t e</w>", "ap t</w>", "a d", "ad apt</w>", "a c", "ac t</w>", ""]
        self.special_tokens_map = {"unk_token": "__unk__", "bos_token": "__start__", "eos_token": "__end__"}

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
        input_text = "adapt act apte"
        output_text = "adapt act apte"
        return input_text, output_text

    def test_full_blenderbot_small_tokenizer(self):
        tokenizer = BlenderbotSmallTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "adapt act apte"
        bpe_tokens  = ["adapt", "act", "ap@@", "te"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = [tokenizer.bos_token] + tokens + [tokenizer.eos_token]

        input_bpe_tokens = [0, 1, 2, 3, 4, 5]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)
        
        
class BlenderbotTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    
    tokenizer_class = BlenderbotTokenizer

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return BlenderbotTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return BlenderbotTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = " lower newer"
        return input_text, output_text

    # def test_full_blenderbot_tokenizer(self):
    #     tokenizer = BlenderbotTokenizer(self.vocab_file, self.merges_file, **self.special_tokens_map)
    #     text = "hello world"
        
    #     bpe_tokens =  ["Ġhell", "o", "Ġworld"]
    #     tokens = tokenizer.tokenize(text)
    #     self.assertListEqual(tokens, bpe_tokens)


        # input_bpe_tokens = [1710,   86, 1085,    2]
        # self.assertListEqual(tokenizer.convert_tokens_to_ids(tokens), input_bpe_tokens)

