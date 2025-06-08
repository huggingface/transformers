import unittest
import os
import shutil
import torch

from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES
# Import Bert2DTokenizer directly, and the module for helper functions
from transformers.models.bert2d import tokenization_bert2d as bert2d_tokenizer_module
from transformers.models.bert2d.tokenization_bert2d import Bert2DTokenizer

from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

class Bert2DSlowTokenizerFinalTest(unittest.TestCase):
    def setUp(self):
        self.tmpdirname = "./tmp_bert2d_final_test"
        os.makedirs(self.tmpdirname, exist_ok=True)

        self.vocab_tokens = [
            "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]",  # 0-4
            "hello", "world",  # 5-6
            "un", "##want", "##ed",  # 7-9
            "runn", "##ing",  # 10-11
            "example", "##with", "##many", "##sub", "##words",  # 12-16
            "a", "##b", "##c", "##d", "##e", "##f", "##g", # 17-23 for multi-subword tests
            "##leading" # 24
        ]
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("\n".join(self.vocab_tokens) + "\n")

        self.tokenizer_default = Bert2DTokenizer(vocab_file=self.vocab_file)
        # Add the requested print and assertion
        print(f"Type of self.tokenizer_default in setUp: {type(self.tokenizer_default)}")
        self.assertTrue(isinstance(self.tokenizer_default, Bert2DTokenizer),
                        f"self.tokenizer_default is not an instance of Bert2DTokenizer, but {type(self.tokenizer_default)}")

        self.tokenizer_custom = Bert2DTokenizer(
            vocab_file=self.vocab_file,
            max_intermediate_subword_positions_per_word=3,
            intermediate_subword_distribution_strategy="leftover_as_last",
            do_lower_case=True # Explicitly set for consistency, though it's default
        )
        # Vocab mapping for reference:
        # [UNK]=0, [CLS]=1, [SEP]=2, [PAD]=3, [MASK]=4
        # hello=5, world=6
        # un=7, ##want=8, ##ed=9
        # runn=10, ##ing=11
        # example=12, ##with=13, ##many=14, ##sub=15, ##words=16
        # a=17, ##b=18, ##c=19, ##d=20, ##e=21, ##f=22, ##g=23
        # ##leading=24

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_initialization_parameters(self):
        self.assertEqual(self.tokenizer_default.max_intermediate_subword_positions_per_word, 1)
        self.assertEqual(self.tokenizer_default.subword_embedding_order, "ending_first")
        self.assertEqual(self.tokenizer_default.intermediate_subword_distribution_strategy, "uniform")
        self.assertIn("word_ids", self.tokenizer_default.model_input_names)
        self.assertIn("subword_ids", self.tokenizer_default.model_input_names)

        self.assertEqual(self.tokenizer_custom.max_intermediate_subword_positions_per_word, 3)
        self.assertEqual(self.tokenizer_custom.intermediate_subword_distribution_strategy, "leftover_as_last")

    def test_single_sentence_defaults_list_output(self):
        text = "unwanted running"
        encoded = self.tokenizer_default(text, add_special_tokens=True)

        self.assertIsInstance(encoded["input_ids"], list)
        self.assertIsInstance(encoded["word_ids"], list)
        self.assertIsInstance(encoded["subword_ids"], list)

        expected_input_ids = [1, 7, 8, 9, 10, 11, 2]  # [CLS] un ##want ##ed runn ##ing [SEP]
        self.assertEqual(encoded["input_ids"], expected_input_ids)

        expected_word_ids = [0, 1, 1, 1, 2, 2, 3]    # CLS, un, un, un, runn, runn, SEP
        self.assertEqual(encoded["word_ids"], expected_word_ids)

        # un ##want ##ed -> 0, 2, 1 (root, 1 inter, last)
        # runn ##ing    -> 0, 1 (root, last)
        expected_subword_ids = [0, 0, 2, 1, 0, 1, 0] # CLS, un, ##want, ##ed, runn, ##ing, SEP
        self.assertEqual(encoded["subword_ids"], expected_subword_ids)

    def test_single_sentence_defaults_tensor_output(self):
        text = "unwanted running"
        encoded = self.tokenizer_default(text, add_special_tokens=True, return_tensors="pt")

        self.assertIsInstance(encoded["input_ids"], torch.Tensor)
        self.assertIsInstance(encoded["word_ids"], torch.Tensor)
        self.assertIsInstance(encoded["subword_ids"], torch.Tensor)

        expected_input_ids = torch.tensor([1, 7, 8, 9, 10, 11, 2])
        self.assertTrue(torch.equal(encoded["input_ids"], expected_input_ids))

        expected_word_ids = torch.tensor([0, 1, 1, 1, 2, 2, 3])
        self.assertTrue(torch.equal(encoded["word_ids"], expected_word_ids))

        expected_subword_ids = torch.tensor([0, 0, 2, 1, 0, 1, 0])
        self.assertTrue(torch.equal(encoded["subword_ids"], expected_subword_ids))

    def test_sentence_pair_defaults_list_output(self):
        text1 = "hello"
        text2 = "world"
        encoded = self.tokenizer_default(text1, text_pair=text2, add_special_tokens=True)
        # Expected: [CLS] hello [SEP] world [SEP]
        # IDs:      1     5     2     6     2
        expected_input_ids = [1, 5, 2, 6, 2]
        self.assertEqual(encoded["input_ids"], expected_input_ids)
        self.assertEqual(encoded["token_type_ids"], [0, 0, 0, 1, 1])

        # Word IDs: CLS(0) hello(1) SEP(2) world(0) SEP(1) (restart after first SEP due to text_pair)
        expected_word_ids = [0, 1, 2, 0, 1]
        self.assertEqual(encoded["word_ids"], expected_word_ids)
        
        # Subword IDs: All single tokens, so mostly 0
        expected_subword_ids = [0, 0, 0, 0, 0] # CLS, hello, SEP, world, SEP
        self.assertEqual(encoded["subword_ids"], expected_subword_ids)

    def test_batch_right_padding_tensors(self):
        texts = ["hello", "unwanted running"] # "hello" is shorter
        encoded = self.tokenizer_default(texts, padding=True, return_tensors="pt", add_special_tokens=True)
        # Max length from "unwanted running": [CLS] un ##want ##ed runn ##ing [SEP] -> 7 tokens
        # "hello": [CLS] hello [SEP] -> 3 tokens. Padded to 7.
        # PAD_ID is 3

        self.assertEqual(encoded["input_ids"].shape, torch.Size([2, 7]))
        self.assertEqual(encoded["word_ids"].shape, torch.Size([2, 7]))
        self.assertEqual(encoded["subword_ids"].shape, torch.Size([2, 7]))
        self.assertEqual(encoded["attention_mask"].shape, torch.Size([2, 7]))

        expected_input_ids = torch.tensor([
            [1, 5, 2, 3, 3, 3, 3],  # [CLS] hello [SEP] [PAD] [PAD] [PAD] [PAD]
            [1, 7, 8, 9, 10, 11, 2] # [CLS] un ##want ##ed runn ##ing [SEP]
        ])
        self.assertTrue(torch.equal(encoded["input_ids"], expected_input_ids))

        expected_word_ids = torch.tensor([
            [0, 1, 2, 0, 0, 0, 0],  # Word IDs for "hello", padded
            [0, 1, 1, 1, 2, 2, 3]   # Word IDs for "unwanted running"
        ])
        self.assertTrue(torch.equal(encoded["word_ids"], expected_word_ids))

        expected_subword_ids = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0],  # Subword IDs for "hello", padded
            [0, 0, 2, 1, 0, 1, 0]   # Subword IDs for "unwanted running"
        ])
        self.assertTrue(torch.equal(encoded["subword_ids"], expected_subword_ids))
        
        expected_attention_mask = torch.tensor([
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1]
        ])
        self.assertTrue(torch.equal(encoded["attention_mask"], expected_attention_mask))

    def test_batch_left_padding_tensors(self):
        texts = ["hello", "unwanted running"]
        self.tokenizer_default.padding_side = "left"
        encoded = self.tokenizer_default(texts, padding=PaddingStrategy.LONGEST, return_tensors="pt", add_special_tokens=True)
        self.tokenizer_default.padding_side = "right" # Reset for other tests

        self.assertEqual(encoded["input_ids"].shape, torch.Size([2, 7]))

        expected_input_ids = torch.tensor([
            [3, 3, 3, 3, 1, 5, 2],  # [PAD] [PAD] [PAD] [PAD] [CLS] hello [SEP]
            [1, 7, 8, 9, 10, 11, 2] # [CLS] un ##want ##ed runn ##ing [SEP]
        ])
        self.assertTrue(torch.equal(encoded["input_ids"], expected_input_ids))

        expected_word_ids = torch.tensor([
            [0, 0, 0, 0, 0, 1, 2],  # Word IDs for "hello", left-padded
            [0, 1, 1, 1, 2, 2, 3]
        ])
        self.assertTrue(torch.equal(encoded["word_ids"], expected_word_ids))

        expected_subword_ids = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0],  # Subword IDs for "hello", left-padded
            [0, 0, 2, 1, 0, 1, 0]
        ])
        self.assertTrue(torch.equal(encoded["subword_ids"], expected_subword_ids))
        
        expected_attention_mask = torch.tensor([
            [0, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ])
        self.assertTrue(torch.equal(encoded["attention_mask"], expected_attention_mask))

    def test_subword_ids_custom_max_intermediate(self):
        # Vocab: example=12, ##with=13, ##many=14, ##sub=15, ##words=16
        text = "examplewithmanysubwords" # tokenized: example ##with ##many ##sub ##words
        # This word has 5 tokens: 1 root, 1 last, 3 intermediate (##with, ##many, ##sub)
        # tokenizer_custom has max_intermediate_subword_positions_per_word=3
        # Since num_intermediate (3) <= max_intermediate (3), they should get distinct IDs: 2, 3, 4
        encoded = self.tokenizer_custom(text)
        
        expected_input_ids = [1, 12, 13, 14, 15, 16, 2] # [CLS] example ##with ##many ##sub ##words [SEP]
        self.assertEqual(encoded["input_ids"], expected_input_ids)
        
        expected_word_ids = [0, 1, 1, 1, 1, 1, 2]
        self.assertEqual(encoded["word_ids"], expected_word_ids)

        # Subword IDs: example(0) ##with(2) ##many(3) ##sub(4) ##words(1)
        expected_subword_ids = [0, 0, 2, 3, 4, 1, 0] # CLS, example, ##with, ##many, ##sub, ##words, SEP
        self.assertEqual(encoded["subword_ids"], expected_subword_ids)

    def test_subword_ids_custom_distribution_leftover(self):
        # Vocab: a=17, ##b=18, ##c=19, ##d=20, ##e=21, ##f=22, ##g=23
        text = "abcdefg" # tokenized: a ##b ##c ##d ##e ##f ##g (7 tokens)
        # Word has 7 tokens: 1 root (a), 1 last (##g), 5 intermediate (##b to ##f)
        # tokenizer_custom has max_intermediate_subword_positions_per_word=3, strategy="leftover_as_last"
        # 5 intermediate tokens, 3 positions.
        # ##b -> 2+0 = 2
        # ##c -> 2+1 = 3
        # ##d -> 2+2 = 4
        # ##e -> leftover, gets ID 1 (last type)
        # ##f -> leftover, gets ID 1 (last type)
        encoded = self.tokenizer_custom(text)

        expected_input_ids = [1, 17, 18, 19, 20, 21, 22, 23, 2] # [CLS] a ##b ##c ##d ##e ##f ##g [SEP]
        self.assertEqual(encoded["input_ids"], expected_input_ids)

        expected_word_ids = [0, 1, 1, 1, 1, 1, 1, 1, 2] # CLS, a, a, a, a, a, a, a, SEP
        self.assertEqual(encoded["word_ids"], expected_word_ids)
        
        # Subword IDs: a(0) ##b(2) ##c(3) ##d(4) ##e(1) ##f(1) ##g(1)
        expected_subword_ids = [0, 0, 2, 3, 4, 1, 1, 1, 0] # CLS, a,##b,##c,##d,##e,##f,##g, SEP
        self.assertEqual(encoded["subword_ids"], expected_subword_ids)

    def test_subword_ids_starts_with_subword(self):
        # Vocab has "##leading"=24
        text = "##leading" # This is a single token word that starts with a subword prefix.
        encoded = self.tokenizer_default(text)
        # Expected tokens: [CLS] ##leading [SEP]
        # Input IDs:       1     24        2
        self.assertEqual(encoded["input_ids"], [1, 24, 2])

        # Word IDs: CLS(0) ##leading(0) SEP(1)
        self.assertEqual(encoded["word_ids"], [0, 0, 1])

        # Subword IDs:
        # Word segment is ["##leading"]. `first_content_token_is_subword` is True.
        # `get_ids_from_subwords(num=1, ..., current_word_starts_with_subword=True)` should return `[1]`.
        # So, [CLS](0) ##leading(1) [SEP](0)
        self.assertEqual(encoded["subword_ids"], [0, 1, 0])

    def test_empty_input(self):
        encoded = self.tokenizer_default("") # No special tokens added by default
        self.assertEqual(encoded["input_ids"], [])
        self.assertEqual(encoded["word_ids"], [])
        self.assertEqual(encoded["subword_ids"], [])

        encoded_special = self.tokenizer_default("", add_special_tokens=True)
        self.assertEqual(encoded_special["input_ids"], [1, 2]) # [CLS], [SEP]
        self.assertEqual(encoded_special["word_ids"], [0, 1])
        self.assertEqual(encoded_special["subword_ids"], [0, 0])

    def test_truncation_behavior(self):
        text = "hello world unwanted running example" # longer sequence
        max_len = 5
        encoded = self.tokenizer_default(text, max_length=max_len, truncation=True, add_special_tokens=True)
        # [CLS] hello world un [SEP] -> if add_special_tokens=True, CLS and SEP count towards max_length
        # So, 3 content tokens: hello world un
        # Expected: [CLS] hello world un [SEP] -> [1, 5, 6, 7, 2] (length 5)
        self.assertEqual(len(encoded["input_ids"]), max_len)
        self.assertEqual(encoded["input_ids"], [1, 5, 6, 7, 2]) # hello=5, world=6, un=7
        self.assertEqual(len(encoded["word_ids"]), max_len)
        self.assertEqual(encoded["word_ids"], [0,1,2,3,4]) # CLS, hello, world, un, SEP
        self.assertEqual(len(encoded["subword_ids"]), max_len)
        self.assertEqual(encoded["subword_ids"], [0,0,0,0,0]) # CLS, hello, world, un, SEP

    def test_model_input_names_check(self):
        self.assertListEqual(
            self.tokenizer_default.model_input_names,
            ["input_ids", "token_type_ids", "attention_mask", "word_ids", "subword_ids"]
        )

if __name__ == "__main__":
    unittest.main()
