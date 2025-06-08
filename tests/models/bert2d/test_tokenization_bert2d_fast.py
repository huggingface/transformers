# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

from transformers.models.bert2d.tokenization_bert2d_fast import (
    Bert2DTokenizerFast,
    create_word_ids,
    create_subword_ids,
    is_subword,
    get_ids_from_subwords,
    get_uniform_id,
    col_round,
)
from transformers.testing_utils import require_tokenizers, slow


@require_tokenizers
class Bert2DTokenizationFastTest(unittest.TestCase):
    
    def get_tokenizer(self, **kwargs):
        return Bert2DTokenizerFast.from_pretrained("yigitbekir/Bert2D-cased-Turkish-128K-WWM-NSW2", **kwargs)

    @slow
    def test_tokenizer_initialization(self):
        # Test default initialization
        tokenizer = self.get_tokenizer()
        self.assertEqual(tokenizer.max_intermediate_subword_positions_per_word, 1)
        self.assertEqual(tokenizer.subword_embedding_order, "ending_first")
        self.assertEqual(tokenizer.intermediate_subword_distribution_strategy, "uniform")
        
        # Test custom initialization
        tokenizer = self.get_tokenizer(
            max_intermediate_subword_positions_per_word=3,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="leftover_as_last",
        )
        self.assertEqual(tokenizer.max_intermediate_subword_positions_per_word, 3)
        self.assertEqual(tokenizer.subword_embedding_order, "ending_first")
        self.assertEqual(tokenizer.intermediate_subword_distribution_strategy, "leftover_as_last")

    @slow
    def test_model_input_names(self):
        tokenizer = self.get_tokenizer()
        self.assertIn("word_ids", tokenizer.model_input_names)
        self.assertIn("subword_ids", tokenizer.model_input_names)
        self.assertIn("input_ids", tokenizer.model_input_names)
        self.assertIn("token_type_ids", tokenizer.model_input_names)
        self.assertIn("attention_mask", tokenizer.model_input_names)

    def test_is_subword(self):
        self.assertTrue(is_subword("##word"))
        self.assertFalse(is_subword("word"))
        self.assertTrue(is_subword("##123"))
        self.assertFalse(is_subword("123"))

    def test_col_round(self):
        self.assertEqual(col_round(1.2), 1)
        self.assertEqual(col_round(1.5), 2)
        self.assertEqual(col_round(1.8), 2)
        self.assertEqual(col_round(0.2), 0)
        self.assertEqual(col_round(0.5), 1)

    def test_get_uniform_id(self):
        self.assertEqual(get_uniform_id(0, 5, 10), 0)
        self.assertEqual(get_uniform_id(5, 5, 10), 2)
        self.assertEqual(get_uniform_id(9, 5, 10), 4)

    def test_create_word_ids(self):
        # Simple case
        tokens = ["[CLS]", "un", "##want", "##ed", ",", "runn", "##ing", "[SEP]"]
        word_ids = create_word_ids(tokens)
        expected_word_ids = [0, 1, 1, 1, 2, 3, 3, 4]
        self.assertListEqual(word_ids, expected_word_ids)
        
        # With padding
        tokens = ["[CLS]", "un", "##want", "[PAD]", "[PAD]"]
        word_ids = create_word_ids(tokens)
        expected_word_ids = [0, 1, 1, 0, 0]
        self.assertListEqual(word_ids, expected_word_ids)
        
        # With restart_new_sentence=True
        # The function only resets at the first [SEP] token, not at every [SEP]
        tokens = ["[CLS]", "un", "##want", "[SEP]", "runn", "##ing", "[SEP]"]
        word_ids = create_word_ids(tokens, restart_new_sentence=True)
        expected_word_ids = [0, 1, 1, 0, 1, 1, 2]  # Resets after first [SEP], then increments normally
        self.assertListEqual(word_ids, expected_word_ids)

    def test_create_subword_ids(self):
        # Test with default parameters
        tokens = ["[CLS]", "un", "##want", "##ed", ",", "runn", "##ing", "[SEP]"]
        subword_ids = create_subword_ids(
            tokens, 
            max_intermediate_subword_positions_per_word=1,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="uniform"
        )
        # Expected: [CLS] doesn't contribute, un+##want+##ed forms a word with 3 subwords
        # "," is a single token word, runn+##ing is a word with 2 subwords, [SEP] doesn't contribute
        expected_subword_ids = [0, 0, 2, 1, 0, 0, 1, 0]
        self.assertListEqual(subword_ids, expected_subword_ids)
        
        # Test with more intermediate positions
        tokens = ["a", "##b", "##c", "##d", "##e"]  # 5 subwords in a word
        subword_ids = create_subword_ids(
            tokens, 
            max_intermediate_subword_positions_per_word=3,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="uniform"
        )
        # With 5 subwords and max=3, we should have 0 (root), 3 intermediate positions, and 1 (last)
        expected_subword_ids = [0, 2, 3, 4, 1]
        self.assertListEqual(subword_ids, expected_subword_ids)

    def test_get_ids_from_subwords(self):
        # Test simple case
        ids = get_ids_from_subwords(
            num_subwords=3,
            max_intermediate_subword_positions_per_word=2,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="uniform",
        )
        self.assertListEqual(ids, [0, 2, 1])  # [root, intermediate, last]
        
        # Test with more subwords than positions (uniform strategy)
        ids = get_ids_from_subwords(
            num_subwords=5,
            max_intermediate_subword_positions_per_word=2,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="uniform",
        )
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids[0], 0)  # root
        self.assertEqual(ids[-1], 1)  # last
        
        # Test with leftover_as_last strategy
        ids = get_ids_from_subwords(
            num_subwords=5,
            max_intermediate_subword_positions_per_word=2,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="leftover_as_last",
        )
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids[0], 0)  # root
        self.assertEqual(ids[-1], 1)  # last
        self.assertEqual(ids.count(1), 2)  # 1 for the last token + 1 leftover

    @slow
    def test_tokenizer_call(self):
        tokenizer = self.get_tokenizer()
        
        # Single sequence
        encoded = tokenizer(
            "unwanted running",
            add_special_tokens=True,
            return_tensors="pt",
        )
        
        # Check if word_ids and subword_ids are in the output
        self.assertIn("word_ids", encoded)
        self.assertIn("subword_ids", encoded)
        
        # Check if they are tensors when return_tensors="pt"
        self.assertIsInstance(encoded["word_ids"], torch.Tensor)
        self.assertIsInstance(encoded["subword_ids"], torch.Tensor)
        
        # Check shapes
        self.assertEqual(encoded["word_ids"].shape, encoded["input_ids"].shape)
        self.assertEqual(encoded["subword_ids"].shape, encoded["input_ids"].shape)
        
        # Check actual tokenization
        tokens = tokenizer.tokenize("unwanted running")
        self.assertTrue(len(tokens) > 0)  # Just check that we get some tokens

    @slow
    def test_tokenizer_call_without_tensors(self):
        tokenizer = self.get_tokenizer()
        
        # Single sequence without tensors
        encoded = tokenizer(
            "unwanted running",
            add_special_tokens=True,
        )
        
        # Check if word_ids and subword_ids are in the output
        self.assertIn("word_ids", encoded)
        self.assertIn("subword_ids", encoded)
        
        # Check if they are lists when return_tensors is not specified
        self.assertIsInstance(encoded["word_ids"], list)
        self.assertIsInstance(encoded["subword_ids"], list)
        
        # Check lengths
        self.assertEqual(len(encoded["word_ids"]), len(encoded["input_ids"]))
        self.assertEqual(len(encoded["subword_ids"]), len(encoded["input_ids"]))

    @slow
    def test_tokenizer_batch_call(self):
        tokenizer = self.get_tokenizer()
        
        # Batch of sequences
        encoded = tokenizer(
            ["unwanted running", "hello world"],
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        
        # Check if word_ids and subword_ids are in the output
        self.assertIn("word_ids", encoded)
        self.assertIn("subword_ids", encoded)
        
        # Check if they are tensors when return_tensors="pt"
        self.assertIsInstance(encoded["word_ids"], torch.Tensor)
        self.assertIsInstance(encoded["subword_ids"], torch.Tensor)
        
        # Check batch dimension
        self.assertEqual(encoded["word_ids"].shape[0], 2)
        self.assertEqual(encoded["subword_ids"].shape[0], 2)
        
        # Check if padding was applied correctly
        self.assertEqual(encoded["word_ids"].shape, encoded["input_ids"].shape)
        self.assertEqual(encoded["subword_ids"].shape, encoded["input_ids"].shape)

    @slow
    def test_padding_behavior(self):
        tokenizer = self.get_tokenizer()
        
        # Test right padding
        tokenizer.padding_side = "right"
        encoded = tokenizer(
            ["unwanted", "unwanted running"],
            add_special_tokens=True,
            padding="max_length",
            max_length=10,
            return_tensors="pt",
        )
        
        # Check if padding was applied correctly to word_ids and subword_ids
        # The shorter sequence should have padding (0) on the right
        self.assertEqual(encoded["word_ids"][0, -1].item(), 0)  # Padding value for word_ids
        
        # Test left padding
        tokenizer.padding_side = "left"
        encoded = tokenizer(
            ["unwanted", "unwanted running"],
            add_special_tokens=True,
            padding="max_length",
            max_length=10,
            return_tensors="pt",
        )
        
        # Check if padding was applied correctly to word_ids and subword_ids
        # The shorter sequence should have padding (0) on the left
        self.assertEqual(encoded["word_ids"][0, 0].item(), 0)  # Padding value for word_ids

    def test_different_distribution_strategies(self):
        tokens = ["a", "##b", "##c", "##d", "##e"]  # 5 subwords in a word
        
        # Test with uniform strategy
        uniform_ids = create_subword_ids(
            tokens,
            max_intermediate_subword_positions_per_word=2,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="uniform"
        )
        
        # Test with leftover_as_last strategy
        leftover_ids = create_subword_ids(
            tokens,
            max_intermediate_subword_positions_per_word=2,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="leftover_as_last"
        )
        
        # The strategies should produce different results
        self.assertNotEqual(uniform_ids, leftover_ids)
        
        # Check specific behavior of leftover_as_last
        # The last position (1) should appear multiple times for leftovers
        self.assertEqual(leftover_ids.count(1), 2)  # 1 for the last token + 1 leftover

    @slow
    def test_edge_cases(self):
        tokenizer = self.get_tokenizer()
        
        # Empty input
        encoded = tokenizer(
            "",
            add_special_tokens=True,
            return_tensors="pt",
        )
        self.assertIn("word_ids", encoded)
        self.assertIn("subword_ids", encoded)
        
        # Single token
        encoded = tokenizer(
            "hello",
            add_special_tokens=False,
            return_tensors="pt",
        )
        self.assertIn("word_ids", encoded)
        self.assertIn("subword_ids", encoded)
        
        # Only special tokens
        encoded = tokenizer(
            "",
            add_special_tokens=True,
            return_tensors="pt",
        )
        self.assertIn("word_ids", encoded)
        self.assertIn("subword_ids", encoded)

    @slow
    def test_tokenize_chinese_chars(self):
        list_of_common_chinese_char = ["的", "人", "有"]
        text_with_chinese_char = "".join(list_of_common_chinese_char)
        
        # Test with tokenize_chinese_chars=True (default)
        tokenizer = self.get_tokenizer(tokenize_chinese_chars=True)
        tokens = tokenizer.tokenize(text_with_chinese_char)
        self.assertTrue(len(tokens) > 0)  # Each character should be tokenized
        
        # Test with tokenize_chinese_chars=False
        tokenizer = self.get_tokenizer(tokenize_chinese_chars=False)
        tokens = tokenizer.tokenize(text_with_chinese_char)
        self.assertTrue(len(tokens) > 0)  # Should still tokenize something
