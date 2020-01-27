import unittest

from transformers import BertTokenizer, BertTokenizerFast, CTRLTokenizer, DistilBertTokenizer, GPT2Tokenizer, \
    GPT2TokenizerFast, RobertaTokenizer, OpenAIGPTTokenizer
from transformers.tokenization_ctrl import CTRLTokenizerFast
from transformers.tokenization_distilbert import DistilBertTokenizerFast
from transformers.tokenization_openai import OpenAIGPTTokenizerFast
from transformers.tokenization_roberta import RobertaTokenizerFast


class FastTokenizerMatchingTest(unittest.TestCase):

    def setUp(self) -> None:
        with open('fixtures/sample_text.txt') as f_data:
            self._data = f_data.read()

    def _tokenize_inputs_and_check_matching(self, tokenizer_p, tokenizer_r):
        # Ensure basic input match
        input_p = tokenizer_p.encode_plus(self._data)
        input_r = tokenizer_r.encode_plus(self._data)

        self.assertSequenceEqual(input_p['input_ids'], input_r['input_ids'])
        self.assertSequenceEqual(input_p['token_type_ids'], input_r['token_type_ids'])
        self.assertSequenceEqual(input_p['attention_mask'], input_r['attention_mask'])

        input_pairs_p = tokenizer_p.encode_plus(self._data, self._data)
        input_pairs_r = tokenizer_r.encode_plus(self._data, self._data)

        self.assertSequenceEqual(input_pairs_p['input_ids'], input_pairs_r['input_ids'])
        self.assertSequenceEqual(input_pairs_p['token_type_ids'], input_pairs_r['token_type_ids'])
        self.assertSequenceEqual(input_pairs_p['attention_mask'], input_pairs_r['attention_mask'])

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

            self._tokenize_inputs_and_check_matching(tokenizer_p, tokenizer_r)

    def test_ctrl(self):
        for tokenizer_name in CTRLTokenizer.pretrained_vocab_files_map['vocab_file'].keys():
            tokenizer_p = CTRLTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = CTRLTokenizerFast.from_pretrained(tokenizer_name)

            self._tokenize_inputs_and_check_matching(tokenizer_p, tokenizer_r)

    def test_distilbert(self):
        for tokenizer_name in DistilBertTokenizer.pretrained_vocab_files_map['vocab_file'].keys():
            tokenizer_p = DistilBertTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = DistilBertTokenizerFast.from_pretrained(tokenizer_name)

            self._tokenize_inputs_and_check_matching(tokenizer_p, tokenizer_r)

    def test_gpt2(self):
        for tokenizer_name in GPT2Tokenizer.pretrained_vocab_files_map['vocab_file'].keys():
            tokenizer_p = GPT2Tokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = GPT2TokenizerFast.from_pretrained(tokenizer_name)

            self._tokenize_inputs_and_check_matching(tokenizer_p, tokenizer_r)

    def test_roberta(self):
        for tokenizer_name in RobertaTokenizer.pretrained_vocab_files_map['vocab_file'].keys():
            tokenizer_p = RobertaTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = RobertaTokenizerFast.from_pretrained(tokenizer_name)

            self._tokenize_inputs_and_check_matching(tokenizer_p, tokenizer_r)

    def test_openai(self):
        for tokenizer_name in OpenAIGPTTokenizer.pretrained_vocab_files_map['vocab_file'].keys():
            tokenizer_p = OpenAIGPTTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = OpenAIGPTTokenizerFast.from_pretrained(tokenizer_name)

            self._tokenize_inputs_and_check_matching(tokenizer_p, tokenizer_r)


if __name__ == '__main__':
    unittest.main()
