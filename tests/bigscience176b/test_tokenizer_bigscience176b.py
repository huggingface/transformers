import json
import os
import torch
import unittest

import numpy as np

from transformers import AutoTokenizer

from utils_test_tokenizer import MMapIndexedDataset

# TO PORT ON transformers.test_utils ?

class BigScienceTokenizationTest(unittest.TestCase):
    """
        The goal here is to compare some text that has been tokenized by a model trained 
        using Megatron-LM. For now:
            - Read the tokenized text (.bin file) + the raw text
            - Load a tokenizer from the hub (fast tokenizer or python tokenizer)
        
        Tokenizer used during training:
            - https://huggingface.co/bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles
        Tokenizer that has been pushed to the hub:
            - https://huggingface.co/bigscience/tokenizer/
        
        # TODO design a test that loads both tokenizers and gets a dataset from datasets to compare the tokenizations
        
        This code is not device agnostic --> figure out what to do?
    """

    def setUp(self):
        super().setUp()
        self.path_tokenizer = "bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles"
        self.path_bin_data = "/home/thomwolf/bigscience/megatron-debug/preprocessed_dataset_text_document"
        self.path_json_dataset = "/home/thomwolf/bigscience/megatron-debug/train_dataset.jsonl"
        self.NB_SENTENCES = 2
    
    def file_exists(self):
        return os.path.isfile(self.path_bin_data+'.idx') and os.path.isfile(self.path_bin_data+'.bin') and os.path.isfile(self.path_json_dataset)
    
    def test_load_tokenizer(self):
        """
            Assert that we can correctly load the tokenizer 
        """
        try:
            _ = AutoTokenizer.from_pretrained(self.path_tokenizer)
        except:
            self.fail("Failed loading tokenizer")

    @unittest.skipUnless(file_exists, "requires data stored in the local machine!")
    def test_load_custom_text_and_bin_data(self):
        """
            Assert that we can correctly load the custom data (json and bin)
        """
        with open(self.path_json_dataset) as f:
            json_data = [json.loads(line) for line in f]
            self.assertTrue(json_data)
        with open(self.path_bin_data+'.bin', mode='rb') as file: # b is important -> binary
            bin_data = file.read()
            self.assertTrue(bin_data)
        with open(self.path_bin_data+'.idx', mode='rb') as file: # b is important -> binary
            bin_data = file.read()
            self.assertTrue(bin_data)
    
    # @unittest.skip("demonstrating skipping")
    @unittest.skipUnless(file_exists, "requires data stored in the local machine!")
    def test_encodings_from_bin_data(self):
        """
            Assert that the created tokens are the same than the one created statically
        """
        tokenizer = AutoTokenizer.from_pretrained(self.path_tokenizer)

        with open(self.path_json_dataset) as f:
            input_text = [json.loads(line)['text']+tokenizer.eos_token for line in f][:self.NB_SENTENCES]
        
        mmapdataset = MMapIndexedDataset(self.path_bin_data)
        computed_tokens = list(map(tokenizer.encode, input_text))
        _ = list(map(np.testing.assert_equal, mmapdataset[:self.NB_SENTENCES], computed_tokens)) # if this passes then the tests pass
        # self.assertListEqual(computed_tokens, mmapdataset[:self.NB_SENTENCES])

        decoded_tokens = list(map(tokenizer.decode, mmapdataset[:self.NB_SENTENCES]))
        self.assertListEqual(decoded_tokens, input_text)

    def test_encodings_from_dataset(self):
        # TODO: get a multilingual dataset and test the tokenizers
        pass    


if __name__ == '__main__':
    # TO DO: Merge argparse with unittest
    # parser = argparse.ArgumentParser()
    # Required parameters
    # parser.add_argument(
    #     "--path_bin_file", default="/home/thomwolf/bigscience/megatron-debug/preprocessed_dataset_text_document.bin", type=str, required=True, help="Path to the created bin file."
    # )
    # parser.add_argument(
    #     "--path_json_dataset", default="/home/thomwolf/bigscience/megatron-debug/train_dataset.jsonl", type=str, required=True, help="Path to the json file dataset (json file that contains the dataset)."
    # )
    # parser.add_argument(
    #     "--path_tokenizer",
    #     default="_",
    #     type=str,
    #     required=True,
    #     help="Name of the tokenizer that has been used \n"
    #     "This specifies the model architecture.",
    # )
    # args = parser.parse_args()
    unittest.main()