import unittest

import numpy as np

from transformers import BertTokenizer, BertTokenizerFast, DistilBertTokenizer, GPT2Tokenizer, \
    GPT2TokenizerFast, RobertaTokenizer, OpenAIGPTTokenizer, TransfoXLTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizerFast
from transformers.tokenization_openai import OpenAIGPTTokenizerFast
from transformers.tokenization_roberta import RobertaTokenizerFast
from transformers.tokenization_transfo_xl import TransfoXLTokenizerFast


class FastTokenizerMatchingTest(unittest.TestCase):

    def setUp(self) -> None:
        with open('fixtures/sample_text.txt') as f_data:
            self._data = f_data.read().replace("\n\n", "\n").strip()

    def assert_sequence_almost_equals(self, a, b, threshold):

        # Handle padding
        if len(a) != len(b):
            max_len = max(len(a), len(b))

            # Pad with a negative number as vocab doesnt allow idx < 0
            # if will be tracked as differences
            if len(a) < max_len:
                a += [-1] * (max_len - len(a))

            if len(b) < max_len:
                b += [-1] * (max_len - len(b))

        # Convert to numpy for convenience
        a_, b_ = np.array(a), np.array(b)

        # Compute elementwise difference
        inputs_diffs = a_ - b_
        inputs_diff = np.count_nonzero(inputs_diffs)
        self.assertLessEqual(inputs_diff / a_.shape[0], threshold)

    def assert_tokenization_python_rust_almost_equals(self, tokenizer_p, tokenizer_r, threshold: float):
        # Ensure basic input match
        input_p = tokenizer_p.encode_plus(self._data)
        input_r = tokenizer_r.encode_plus(self._data)

        for key in input_p.keys():
            self.assert_sequence_almost_equals(input_p[key], input_r[key], threshold)

        input_pairs_p = tokenizer_p.encode_plus(self._data, self._data)
        input_pairs_r = tokenizer_r.encode_plus(self._data, self._data)

        for key in input_p.keys():
            self.assert_sequence_almost_equals(input_pairs_p[key], input_pairs_r[key], threshold)

        # Ensure truncation match
        input_p = tokenizer_p.encode_plus(self._data, max_length=512, pad_to_max_length=True)
        input_r = tokenizer_r.encode_plus(self._data, max_length=512, pad_to_max_length=True)

        self.assertSequenceEqual(input_p['input_ids'], input_r['input_ids'])
        self.assertSequenceEqual(input_p['token_type_ids'], input_r['token_type_ids'])
        self.assertSequenceEqual(input_p['attention_mask'], input_r['attention_mask'])

        # Ensure truncation with stride match
        # input_p = tokenizer_p.encode_plus(self._data, max_length=512, stride=3, return_overflowing_tokens=True)
        # input_r = tokenizer_r.encode_plus(self._data, max_length=512, stride=3, return_overflowing_tokens=True)
        #
        # self.assertSequenceEqual(input_p['input_ids'], input_r['input_ids'])
        # self.assertSequenceEqual(input_p['token_type_ids'], input_r['token_type_ids'])
        # self.assertSequenceEqual(input_p['attention_mask'], input_r['attention_mask'])

    def test_bert(self):
        for tokenizer_name in BertTokenizer.pretrained_vocab_files_map['vocab_file'].keys():
            tokenizer_p = BertTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = BertTokenizerFast.from_pretrained(tokenizer_name)

            # Bert should match 100%
            self.assert_tokenization_python_rust_almost_equals(tokenizer_p, tokenizer_r, 0.0)

    def test_transfoxl(self):
        for tokenizer_name in TransfoXLTokenizer.pretrained_vocab_files_map['pretrained_vocab_file'].keys():
            tokenizer_p = TransfoXLTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = TransfoXLTokenizerFast.from_pretrained(tokenizer_name)

            self.assert_tokenization_python_rust_almost_equals(tokenizer_p, tokenizer_r, 0.0)

    def test_distilbert(self):
        for tokenizer_name in DistilBertTokenizer.pretrained_vocab_files_map['vocab_file'].keys():
            tokenizer_p = DistilBertTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = DistilBertTokenizerFast.from_pretrained(tokenizer_name)

            # DistilBert should match 100%
            self.assert_tokenization_python_rust_almost_equals(tokenizer_p, tokenizer_r, 0.0)

    def test_gpt2(self):
        for tokenizer_name in GPT2Tokenizer.pretrained_vocab_files_map['vocab_file'].keys():
            tokenizer_p = GPT2Tokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = GPT2TokenizerFast.from_pretrained(tokenizer_name)

            self.assert_tokenization_python_rust_almost_equals(tokenizer_p, tokenizer_r, 0.05)

    def test_roberta(self):
        for tokenizer_name in RobertaTokenizer.pretrained_vocab_files_map['vocab_file'].keys():
            tokenizer_p = RobertaTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = RobertaTokenizerFast.from_pretrained(tokenizer_name)

            self.assert_tokenization_python_rust_almost_equals(tokenizer_p, tokenizer_r, 0.05)

    def test_openai(self):
        for tokenizer_name in OpenAIGPTTokenizer.pretrained_vocab_files_map['vocab_file'].keys():
            tokenizer_p = OpenAIGPTTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = OpenAIGPTTokenizerFast.from_pretrained(tokenizer_name)

            self.assert_tokenization_python_rust_almost_equals(tokenizer_p, tokenizer_r, 0.05)


if __name__ == '__main__':
    unittest.main()
