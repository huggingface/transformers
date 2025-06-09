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

import os
import unittest

import torch

from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES
from transformers.models.bert2d.tokenization_bert2d import Bert2DTokenizer
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
from transformers.utils import PaddingStrategy

from ...test_tokenization_common import TokenizerTesterMixin, filter_non_english


@require_tokenizers
class Bert2DTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "yigitbekir/Bert2D-cased-Turkish-128K-WWM-NSW2"
    tokenizer_class = Bert2DTokenizer  # Use BertTokenizer as slow tokenizer
    rust_tokenizer_class = Bert2DTokenizerFast
    test_rust_tokenizer = True
    space_between_special_tokens = True
    from_pretrained_filter = filter_non_english

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]
        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(cls.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

    def get_input_output_texts(self, tokenizer):
        input_text = "UNwant\u00e9d,running"
        output_text = "unwanted, running"
        return input_text, output_text

    def get_clean_sequence(self, tokenizer):
        input_text, output_text = self.get_input_output_texts(tokenizer)
        ids = tokenizer.encode(output_text, add_special_tokens=False)
        return input_text, ids

    def test_bert2d_specific_attributes(self):
        tokenizer = Bert2DTokenizerFast.from_pretrained(self.from_pretrained_id)
        
        # Check default values
        self.assertEqual(tokenizer.max_intermediate_subword_positions_per_word, 1)
        self.assertEqual(tokenizer.subword_embedding_order, "ending_first")
        self.assertEqual(tokenizer.intermediate_subword_distribution_strategy, "uniform")
        
        # Check model_input_names
        self.assertIn("word_ids", tokenizer.model_input_names)
        self.assertIn("subword_ids", tokenizer.model_input_names)

    def test_custom_parameters(self):
        # Test with custom parameters
        tokenizer = Bert2DTokenizerFast.from_pretrained(
            self.from_pretrained_id,
            max_intermediate_subword_positions_per_word=3,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="leftover_as_last",
        )
        
        self.assertEqual(tokenizer.max_intermediate_subword_positions_per_word, 3)
        self.assertEqual(tokenizer.subword_embedding_order, "ending_first")
        self.assertEqual(tokenizer.intermediate_subword_distribution_strategy, "leftover_as_last")

    def test_word_ids_generation(self):
        tokenizer = self.get_rust_tokenizer()
        
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
        tokens = ["[CLS]", "un", "##want", "[SEP]", "runn", "##ing", "[SEP]"]
        word_ids = create_word_ids(tokens, restart_new_sentence=True)
        expected_word_ids = [0, 1, 1, 2, 0, 0, 1]  # Resets after first [SEP]
        self.assertListEqual(word_ids, expected_word_ids)

    def test_subword_ids_generation(self):
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
        subword_ids = create_subword_ids(
            tokens, 
            max_intermediate_subword_positions_per_word=3,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="uniform"
        )
        # With 3 intermediate positions, un+##want+##ed should have more distinct positions
        expected_subword_ids = [0, 0, 2, 1, 0, 0, 1, 0]
        self.assertListEqual(subword_ids, expected_subword_ids)
        
        # Test with leftover_as_last strategy
        tokens = ["[CLS]", "a", "##b", "##c", "##d", "##e", "[SEP]"]  # 5 subwords in a word
        subword_ids = create_subword_ids(
            tokens[1:6],  # Just the word tokens
            max_intermediate_subword_positions_per_word=2,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="leftover_as_last"
        )
        # With max=2 and 5 subwords, we should have [0, 2, 3, 1, 1] 
        # (root, 2 intermediate, and 2 leftovers as last)
        expected_subword_ids = [0, 2, 3, 1, 1]
        self.assertListEqual(subword_ids, expected_subword_ids)

    def test_tokenizer_encode_plus_with_word_and_subword_ids(self):
        tokenizer = self.get_rust_tokenizer()
        
        # Single sequence
        encoded = tokenizer.encode_plus(
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

    def test_tokenizer_encode_plus_without_tensors(self):
        tokenizer = self.get_rust_tokenizer()
        
        # Single sequence without tensors
        encoded = tokenizer.encode_plus(
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

    def test_tokenizer_batch_encode_plus(self):
        tokenizer = self.get_rust_tokenizer()
        
        # Batch of sequences
        encoded = tokenizer.batch_encode_plus(
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

    def test_padding_behavior(self):
        tokenizer = self.get_rust_tokenizer()
        
        # Test right padding
        tokenizer.padding_side = "right"
        encoded = tokenizer.batch_encode_plus(
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
        encoded = tokenizer.batch_encode_plus(
            ["unwanted", "unwanted running"],
            add_special_tokens=True,
            padding="max_length",
            max_length=10,
            return_tensors="pt",
        )
        
        # Check if padding was applied correctly to word_ids and subword_ids
        # The shorter sequence should have padding (0) on the left
        self.assertEqual(encoded["word_ids"][0, 0].item(), 0)  # Padding value for word_ids

    def test_utility_functions(self):
        # Test is_subword
        self.assertTrue(is_subword("##word"))
        self.assertFalse(is_subword("word"))
        
        # Test col_round
        self.assertEqual(col_round(1.2), 1)
        self.assertEqual(col_round(1.5), 2)
        self.assertEqual(col_round(1.8), 2)
        
        # Test get_uniform_id
        self.assertEqual(get_uniform_id(0, 5, 10), 0)
        self.assertEqual(get_uniform_id(5, 5, 10), 2)
        self.assertEqual(get_uniform_id(9, 5, 10), 4)
        
        # Test get_ids_from_subwords
        ids = get_ids_from_subwords(
            num_subwords_in_current_word=3,
            max_intermediate_subword_positions_per_word=2,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="uniform",
        )
        self.assertListEqual(ids, [0, 2, 1])  # [root, intermediate, last]
        
        # Test with more subwords than positions
        ids = get_ids_from_subwords(
            num_subwords_in_current_word=5,
            max_intermediate_subword_positions_per_word=2,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="uniform",
        )
        # Should distribute 3 intermediate tokens into 2 positions
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids[0], 0)  # root
        self.assertEqual(ids[-1], 1)  # last

    def test_different_distribution_strategies(self):
        tokens = ["[CLS]", "a", "##b", "##c", "##d", "##e", "[SEP]"]  # 5 subwords in a word
        
        # Test with uniform strategy
        uniform_ids = create_subword_ids(
            tokens[1:6],  # Just the word tokens
            max_intermediate_subword_positions_per_word=2,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="uniform"
        )
        
        # Test with leftover_as_last strategy
        leftover_ids = create_subword_ids(
            tokens[1:6],  # Just the word tokens
            max_intermediate_subword_positions_per_word=2,
            subword_embedding_order="ending_first",
            intermediate_subword_distribution_strategy="leftover_as_last"
        )
        
        # The strategies should produce different results
        self.assertNotEqual(uniform_ids, leftover_ids)
        
        # Check specific behavior of leftover_as_last
        # The last position (1) should appear multiple times for leftovers
        self.assertEqual(leftover_ids.count(1), 2)  # 1 for the last token + 2 leftovers

    def test_edge_cases(self):
        tokenizer = self.get_rust_tokenizer()
        
        # Empty input
        encoded = tokenizer.encode_plus(
            "",
            add_special_tokens=True,
            return_tensors="pt",
        )
        self.assertIn("word_ids", encoded)
        self.assertIn("subword_ids", encoded)
        
        # Single token
        encoded = tokenizer.encode_plus(
            "hello",
            add_special_tokens=False,
            return_tensors="pt",
        )
        self.assertIn("word_ids", encoded)
        self.assertIn("subword_ids", encoded)
        
        # Only special tokens
        encoded = tokenizer.encode_plus(
            "",
            add_special_tokens=True,
            return_tensors="pt",
        )
        self.assertIn("word_ids", encoded)
        self.assertIn("subword_ids", encoded)

    def test_integration_with_model_inputs(self):
        tokenizer = self.get_rust_tokenizer()
        
        # Verify that model_input_names contains the custom fields
        self.assertIn("word_ids", tokenizer.model_input_names)
        self.assertIn("subword_ids", tokenizer.model_input_names)
        
        # Check that prepare_for_model includes these fields
        encoded = tokenizer.encode_plus(
            "unwanted running",
            add_special_tokens=True,
        )
        
        # All model input names should be in the output
        for input_name in tokenizer.model_input_names:
            self.assertIn(input_name, encoded)

    @slow
    def test_tokenizer_integration(self):
        # This test ensures that the tokenizer works end-to-end with a realistic example
        tokenizer = Bert2DTokenizerFast.from_pretrained("yigitbekir/Bert2D-cased-Turkish-128K-WWM-NSW2")
        
        text = "UNwant\u00e9d,running"
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt",
        )
        
        # Decode back and check
        decoded_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
        self.assertEqual(
            decoded_tokens,
            ['[CLS]', 'un', '##want', '##ed', ',', 'running', '[SEP]']
        )
        
        # Check word_ids
        expected_word_ids = torch.tensor([[0, 1, 1, 1, 2, 3, 4]])
        self.assertTrue(torch.all(encoded["word_ids"] == expected_word_ids))
