import unittest

import numpy as np

from tests.utils import require_torch
from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    DistilBertTokenizer,
    GPT2Tokenizer,
    GPT2TokenizerFast,
    OpenAIGPTTokenizer,
    PreTrainedTokenizer,
    RobertaTokenizer,
    TransfoXLTokenizer,
    is_torch_available,
)
from transformers.tokenization_distilbert import DistilBertTokenizerFast
from transformers.tokenization_openai import OpenAIGPTTokenizerFast
from transformers.tokenization_roberta import RobertaTokenizerFast
from transformers.tokenization_transfo_xl import TransfoXLTokenizerFast


class FastTokenizerMatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        with open("tests/fixtures/sample_text.txt") as f_data:
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

        for key in filter(lambda x: x in ["input_ids", "token_type_ids", "attention_mask"], input_p.keys()):
            self.assert_sequence_almost_equals(input_p[key], input_r[key], threshold)

        input_pairs_p = tokenizer_p.encode_plus(self._data, self._data)
        input_pairs_r = tokenizer_r.encode_plus(self._data, self._data)

        for key in filter(lambda x: x in ["input_ids", "token_type_ids", "attention_mask"], input_p.keys()):
            self.assert_sequence_almost_equals(input_pairs_p[key], input_pairs_r[key], threshold)

        # Ensure truncation match
        input_p = tokenizer_p.encode_plus(self._data, max_length=512)
        input_r = tokenizer_r.encode_plus(self._data, max_length=512)

        for key in filter(lambda x: x in ["input_ids", "token_type_ids", "attention_mask"], input_p.keys()):
            self.assert_sequence_almost_equals(input_p[key], input_r[key], threshold)

        # Ensure truncation with stride match
        input_p = tokenizer_p.encode_plus(self._data, max_length=512, stride=3, return_overflowing_tokens=True)
        input_r = tokenizer_r.encode_plus(self._data, max_length=512, stride=3, return_overflowing_tokens=True)

        for key in filter(lambda x: x in ["input_ids", "token_type_ids", "attention_mask"], input_p.keys()):
            self.assert_sequence_almost_equals(input_p[key], input_r[key], threshold)

    def assert_padding(self, tokenizer_r, tokenizer_p):
        # Simple input
        input_r = tokenizer_r.encode("This is a simple input", max_length=15, pad_to_max_length=True)
        input_p = tokenizer_p.encode("This is a simple input", max_length=15, pad_to_max_length=True)

        self.assertSequenceEqual(input_r, input_p)

        # Simple input
        input_r = tokenizer_r.encode_plus("This is a simple input", max_length=15, pad_to_max_length=True)
        input_p = tokenizer_p.encode_plus("This is a simple input", max_length=15, pad_to_max_length=True)

        self.assertSequenceEqual(input_r, input_p)

        # Simple input
        # TODO: Re-enable this test when batch_encode_plus with padding correctly handles padding
        # input_r = tokenizer_r.batch_encode_plus(
        #     ["This is a simple input 1", "This is a simple input 2"], max_length=15, pad_to_max_length=True
        # )
        # input_p = tokenizer_p.batch_encode_plus(
        #     ["This is a simple input 1", "This is a simple input 2"], max_length=15, pad_to_max_length=True
        # )

        # self.assertSequenceEqual(input_r, input_p)

        # Pair input
        input_r = tokenizer_r.encode("This is a simple input", "This is a pair", max_length=15, pad_to_max_length=True)
        input_p = tokenizer_p.encode("This is a simple input", "This is a pair", max_length=15, pad_to_max_length=True)

        self.assertSequenceEqual(input_r, input_p)

        # Pair input
        input_r = tokenizer_r.encode_plus(
            "This is a simple input", "This is a pair", max_length=15, pad_to_max_length=True
        )
        input_p = tokenizer_p.encode_plus(
            "This is a simple input", "This is a pair", max_length=15, pad_to_max_length=True
        )

        self.assertSequenceEqual(input_r, input_p)

        # Pair input
        # TODO: Re-enable this test when batch_encode_plus with padding correctly handles padding
        # input_r = tokenizer_r.batch_encode_plus(
        #     ["This is a simple input 1", "This is a simple input 2"],
        #     ["This is a simple pair 1", "This is a simple pair 2"],
        #     max_length=15,
        #     pad_to_max_length=True,
        # )
        # input_p = tokenizer_p.batch_encode_plus(
        #     ["This is a simple input 1", "This is a simple input 2"],
        #     ["This is a simple pair 1", "This is a simple pair 2"],
        #     max_length=15,
        #     pad_to_max_length=True,
        # )

        # self.assertSequenceEqual(input_r, input_p)

    def assert_add_tokens(self, tokenizer_r):
        vocab_size = tokenizer_r.vocab_size
        self.assertEqual(tokenizer_r.add_tokens(""), 0)
        self.assertEqual(tokenizer_r.add_tokens("testoken"), 1)
        self.assertEqual(tokenizer_r.add_tokens(["testoken1", "testtoken2"]), 2)
        self.assertEqual(len(tokenizer_r), vocab_size + 3)

        self.assertEqual(tokenizer_r.add_special_tokens({}), 0)
        self.assertRaises(
            AssertionError, tokenizer_r.add_special_tokens, {"additional_special_tokens": "<testtoken1>"}
        )
        self.assertEqual(tokenizer_r.add_special_tokens({"additional_special_tokens": ["<testtoken2>"]}), 1)
        self.assertEqual(
            tokenizer_r.add_special_tokens({"additional_special_tokens": ["<testtoken3>", "<testtoken4>"]}), 2
        )
        self.assertEqual(len(tokenizer_r), vocab_size + 6)

    def assert_offsets_mapping(self, tokenizer):
        text = "Wonderful no inspiration example with subtoken"
        pair = "Along with an awesome pair"

        # No pair
        tokens_with_offsets = tokenizer.encode_plus(text, return_special_tokens_mask=True, return_offsets_mapping=True)
        added_tokens = tokenizer.num_added_tokens(False)
        offsets = tokens_with_offsets["offset_mapping"]

        # Assert there is the same number of tokens and offsets
        self.assertEqual(len(offsets), len(tokens_with_offsets["input_ids"]))

        # Assert there is online added_tokens special_tokens
        self.assertEqual(sum([0 if x else 1 for x in offsets]), added_tokens)
        self.assertEqual(sum(tokens_with_offsets["special_tokens_mask"]), added_tokens)

        # Pairs
        tokens_with_offsets = tokenizer.encode_plus(
            text, pair, return_special_tokens_mask=True, return_offsets_mapping=True
        )
        added_tokens = tokenizer.num_added_tokens(True)
        offsets = tokens_with_offsets["offset_mapping"]

        # Assert there is the same number of tokens and offsets
        self.assertEqual(len(offsets), len(tokens_with_offsets["input_ids"]))

        # Assert there is online added_tokens special_tokens
        self.assertEqual(sum([0 if x else 1 for x in offsets]), added_tokens)
        self.assertEqual(sum(tokens_with_offsets["special_tokens_mask"]), added_tokens)

    def assert_batch_encode_dynamic_overflowing(self, tokenizer: PreTrainedTokenizer):
        """
        When calling batch_encode with multiple sequence it can returns different number of
        overflowing encoding for each sequence:
        [
          Sequence 1: [Encoding 1, Encoding 2],
          Sequence 2: [Encoding 1],
          Sequence 3: [Encoding 1, Encoding 2, ... Encoding N]
        ]
        This needs to be padded so that it can represented as a tensor
        """
        returned_tensor = "pt" if is_torch_available() else "tf"

        tokens = tokenizer.encode_plus(
            "HuggingFace is solving NLP one commit at a time",
            max_length=6,
            return_tensors=returned_tensor,
            return_overflowing_tokens=True,
        )

        for key in filter(lambda x: "overflow_to_sample_mapping" not in x, tokens.keys()):
            self.assertEqual(len(tokens[key].shape), 2)

        # Mono sample
        tokens = tokenizer.batch_encode_plus(
            ["HuggingFace is solving NLP one commit at a time"],
            max_length=6,
            pad_to_max_len=True,
            return_tensors=returned_tensor,
            return_overflowing_tokens=True,
        )

        for key in filter(lambda x: "overflow_to_sample_mapping" not in x, tokens.keys()):
            self.assertEqual(len(tokens[key].shape), 2)
            self.assertEqual(tokens[key].shape[-1], 6)

        # Multi sample
        tokens = tokenizer.batch_encode_plus(
            ["HuggingFace is solving NLP one commit at a time", "Very tiny input"],
            max_length=6,
            pad_to_max_len=True,
            return_tensors=returned_tensor,
            return_overflowing_tokens=True,
        )

        for key in filter(lambda x: "overflow_to_sample_mapping" not in x, tokens.keys()):
            self.assertEqual(len(tokens[key].shape), 2)
            self.assertEqual(tokens[key].shape[-1], 6)

    def assert_build_inputs_with_special_tokens(self, tokenizer_r, tokenizer_p):
        # Input string
        input_simple = tokenizer_p.tokenize("This is a sample input")
        input_pair = tokenizer_p.tokenize("This is a sample pair")

        # Generate output
        output_r = tokenizer_r.build_inputs_with_special_tokens(input_simple)
        output_p = tokenizer_p.build_inputs_with_special_tokens(input_simple)
        self.assertEqual(output_p, output_r)

        # Generate pair output
        output_r = tokenizer_r.build_inputs_with_special_tokens(input_simple, input_pair)
        output_p = tokenizer_p.build_inputs_with_special_tokens(input_simple, input_pair)
        self.assertEqual(output_p, output_r)

        # Input tokens id
        input_simple = tokenizer_p.encode("This is a sample input")
        input_pair = tokenizer_p.encode("This is a sample pair")

        # Generate output
        output_r = tokenizer_r.build_inputs_with_special_tokens(input_simple)
        output_p = tokenizer_p.build_inputs_with_special_tokens(input_simple)
        self.assertEqual(output_p, output_r)

        # Generate pair output
        output_r = tokenizer_r.build_inputs_with_special_tokens(input_simple, input_pair)
        output_p = tokenizer_p.build_inputs_with_special_tokens(input_simple, input_pair)
        self.assertEqual(output_p, output_r)

    def assert_save_pretrained(self, tokenizer_r, tokenizer_p):

        # Checks it save with the same files
        self.assertSequenceEqual(tokenizer_r.save_vocabulary("."), tokenizer_p.save_vocabulary("."))

        # Checks everything loads correctly in the same way
        tokenizer_rp, tokenizer_pp = tokenizer_r.from_pretrained("."), tokenizer_p.from_pretrained(".")

        # Check special tokens are set accordingly on Rust and Python
        for key in tokenizer_pp.special_tokens_map:
            self.assertTrue(hasattr(tokenizer_rp, key))
            # self.assertEqual(getattr(tokenizer_rp, key), getattr(tokenizer_pp, key))
            # self.assertEqual(getattr(tokenizer_rp, key + "_id"), getattr(tokenizer_pp, key + "_id"))

    def test_bert(self):
        for tokenizer_name in BertTokenizer.pretrained_vocab_files_map["vocab_file"].keys():
            tokenizer_p = BertTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = BertTokenizerFast.from_pretrained(tokenizer_name)

            # Check we have the same number of added_tokens for both pair and non-pair inputs.
            self.assertEqual(tokenizer_r.num_added_tokens(False), tokenizer_p.num_added_tokens(False))
            self.assertEqual(tokenizer_r.num_added_tokens(True), tokenizer_p.num_added_tokens(True))

            # Check we have the correct max_length for both pair and non-pair inputs.
            self.assertEqual(tokenizer_r.max_len_single_sentence, tokenizer_p.max_len_single_sentence)
            self.assertEqual(tokenizer_r.max_len_sentences_pair, tokenizer_p.max_len_sentences_pair)

            # Assert the set of special tokens match.
            self.assertSequenceEqual(
                tokenizer_p.special_tokens_map.items(),
                tokenizer_r.special_tokens_map.items(),
                "Bert tokenizers doesn't have the same set of special_tokens",
            )

            # Assure tokenization overlap between python and rust impl.
            self.assert_tokenization_python_rust_almost_equals(tokenizer_p, tokenizer_r, 0.0)

            # Ensure add_tokens and add_special_tokens return the correct vocab size
            self.assert_add_tokens(tokenizer_r)

            # Check for offsets mapping
            self.assert_offsets_mapping(tokenizer_r)

            # Check for dynamic encoding sequence handling in batch_encode_plus
            self.assert_batch_encode_dynamic_overflowing(tokenizer_r)

            # Check alignment for build_inputs_with_special_tokens
            self.assert_build_inputs_with_special_tokens(tokenizer_r, tokenizer_p)

            # Check the number of returned files for save_vocabulary
            self.assert_save_pretrained(tokenizer_r, tokenizer_p)

            # Check for padding
            self.assert_padding(tokenizer_r, tokenizer_p)

    @require_torch
    def test_transfoxl(self):
        for tokenizer_name in TransfoXLTokenizer.pretrained_vocab_files_map["pretrained_vocab_file"].keys():
            tokenizer_p = TransfoXLTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = TransfoXLTokenizerFast.from_pretrained(tokenizer_name)

            # Check we have the same number of added_tokens for both pair and non-pair inputs.
            self.assertEqual(tokenizer_r.num_added_tokens(False), tokenizer_p.num_added_tokens(False))
            self.assertEqual(tokenizer_r.num_added_tokens(True), tokenizer_p.num_added_tokens(True))

            # Check we have the correct max_length for both pair and non-pair inputs.
            self.assertEqual(tokenizer_r.max_len_single_sentence, tokenizer_p.max_len_single_sentence)
            self.assertEqual(tokenizer_r.max_len_sentences_pair, tokenizer_p.max_len_sentences_pair)

            # Assert the set of special tokens match.
            self.assertSequenceEqual(
                tokenizer_p.special_tokens_map.items(),
                tokenizer_r.special_tokens_map.items(),
                "TransfoXL tokenizers doesn't have the same set of special_tokens",
            )

            # Assure tokenization overlap between python and rust impl.
            self.assert_tokenization_python_rust_almost_equals(tokenizer_p, tokenizer_r, 0.0)

            # Ensure add_tokens and add_special_tokens return the correct vocab size
            self.assert_add_tokens(tokenizer_r)

            # Check for offsets mapping
            self.assert_offsets_mapping(tokenizer_r)

            # Check for dynamic encoding sequence handling in batch_encode_plus
            self.assertRaises(ValueError, self.assert_batch_encode_dynamic_overflowing, tokenizer_r)

            # Check alignment for build_inputs_with_special_tokens
            self.assert_build_inputs_with_special_tokens(tokenizer_r, tokenizer_p)

            # Check for padding
            self.assertRaises(ValueError, self.assert_padding, tokenizer_r, tokenizer_p)

            # Check the number of returned files for save_vocabulary
            # TransfoXL tokenizers comes in a special format which is not compatible at all
            # with rust tokenizers. We ensure the errors detection at correctly raised
            tokenizer_r_files = tokenizer_r.save_pretrained(".")
            self.assertSequenceEqual(
                tokenizer_r_files, ["./vocab.json", "./special_tokens_map.json", "./added_tokens.json"]
            )

            # Check loading Python-tokenizer save through Rust doesnt work (and the opposite)
            self.assertRaises(ValueError, tokenizer_p.from_pretrained, *tokenizer_r_files)
            self.assertRaises(ValueError, tokenizer_r.from_pretrained, *tokenizer_p.save_pretrained("."))

            # Check loading works for Python to Python and Rust to Rust
            # Issue: https://github.com/huggingface/transformers/issues/3000
            # self.assertIsNotNone(tokenizer_p.__class__.from_pretrained('./'))
            self.assertIsNotNone(tokenizer_r.__class__.from_pretrained("./"))

    def test_distilbert(self):
        for tokenizer_name in DistilBertTokenizer.pretrained_vocab_files_map["vocab_file"].keys():
            tokenizer_p = DistilBertTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = DistilBertTokenizerFast.from_pretrained(tokenizer_name)

            # Check we have the same number of added_tokens for both pair and non-pair inputs.
            self.assertEqual(tokenizer_r.num_added_tokens(False), tokenizer_p.num_added_tokens(False))
            self.assertEqual(tokenizer_r.num_added_tokens(True), tokenizer_p.num_added_tokens(True))

            # Check we have the correct max_length for both pair and non-pair inputs.
            self.assertEqual(tokenizer_r.max_len_single_sentence, tokenizer_p.max_len_single_sentence)
            self.assertEqual(tokenizer_r.max_len_sentences_pair, tokenizer_p.max_len_sentences_pair)

            # DistilBert should match 100%
            # Assert the set of special tokens match.
            self.assertSequenceEqual(
                tokenizer_p.special_tokens_map.items(),
                tokenizer_r.special_tokens_map.items(),
                "DistilBert tokenizers doesn't have the same set of special_tokens",
            )

            # Assure tokenization overlap between python and rust impl.
            self.assert_tokenization_python_rust_almost_equals(tokenizer_p, tokenizer_r, 0.0)

            # Ensure add_tokens and add_special_tokens return the correct vocab size
            self.assert_add_tokens(tokenizer_r)

            # Check for offsets mapping
            self.assert_offsets_mapping(tokenizer_r)

            # Check for dynamic encoding sequence handling in batch_encode_plus
            self.assert_batch_encode_dynamic_overflowing(tokenizer_r)

            # Check alignment for build_inputs_with_special_tokens
            self.assert_build_inputs_with_special_tokens(tokenizer_r, tokenizer_p)

            # Check the number of returned files for save_vocabulary
            self.assert_save_pretrained(tokenizer_r, tokenizer_p)

            # Check for padding
            self.assert_padding(tokenizer_r, tokenizer_p)

    def test_gpt2(self):
        for tokenizer_name in GPT2Tokenizer.pretrained_vocab_files_map["vocab_file"].keys():
            tokenizer_p = GPT2Tokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = GPT2TokenizerFast.from_pretrained(tokenizer_name)

            # Check we have the same number of added_tokens for both pair and non-pair inputs.
            self.assertEqual(tokenizer_r.num_added_tokens(False), tokenizer_p.num_added_tokens(False))
            self.assertEqual(tokenizer_r.num_added_tokens(True), tokenizer_p.num_added_tokens(True))

            # Check we have the correct max_length for both pair and non-pair inputs.
            self.assertEqual(tokenizer_r.max_len_single_sentence, tokenizer_p.max_len_single_sentence)
            self.assertEqual(tokenizer_r.max_len_sentences_pair, tokenizer_p.max_len_sentences_pair)

            # Assert the set of special tokens match.
            self.assertSequenceEqual(
                tokenizer_p.special_tokens_map.items(),
                tokenizer_r.special_tokens_map.items(),
                "GPT2 tokenizers doesn't have the same set of special_tokens",
            )

            # Assure tokenization overlap between python and rust impl.
            self.assert_tokenization_python_rust_almost_equals(tokenizer_p, tokenizer_r, 0.0)

            # Ensure add_tokens and add_special_tokens return the correct vocab size
            self.assert_add_tokens(tokenizer_r)

            # Check for offsets mapping
            self.assert_offsets_mapping(tokenizer_r)

            # Check for dynamic encoding sequence handling in batch_encode_plus
            self.assertRaises(ValueError, self.assert_batch_encode_dynamic_overflowing, tokenizer_r)

            # Check alignment for build_inputs_with_special_tokens
            self.assert_build_inputs_with_special_tokens(tokenizer_r, tokenizer_p)

            # Check the number of returned files for save_vocabulary
            self.assert_save_pretrained(tokenizer_r, tokenizer_p)

            # Check for padding
            self.assertRaises(ValueError, self.assert_padding, tokenizer_r, tokenizer_p)

    def test_roberta(self):
        for tokenizer_name in RobertaTokenizer.pretrained_vocab_files_map["vocab_file"].keys():
            tokenizer_p = RobertaTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = RobertaTokenizerFast.from_pretrained(tokenizer_name)

            # Check we have the same number of added_tokens for both pair and non-pair inputs.
            self.assertEqual(tokenizer_r.num_added_tokens(False), tokenizer_p.num_added_tokens(False))
            self.assertEqual(tokenizer_r.num_added_tokens(True), tokenizer_p.num_added_tokens(True))

            # Check we have the correct max_length for both pair and non-pair inputs.
            self.assertEqual(tokenizer_r.max_len_single_sentence, tokenizer_p.max_len_single_sentence)
            self.assertEqual(tokenizer_r.max_len_sentences_pair, tokenizer_p.max_len_sentences_pair)

            # Assert the set of special tokens match.
            self.assertSequenceEqual(
                tokenizer_p.special_tokens_map.items(),
                tokenizer_r.special_tokens_map.items(),
                "Roberta tokenizers doesn't have the same set of special_tokens",
            )

            # Assure tokenization overlap between python and rust impl.
            self.assert_tokenization_python_rust_almost_equals(tokenizer_p, tokenizer_r, 0.01)

            # Ensure add_tokens and add_special_tokens return the correct vocab size
            self.assert_add_tokens(tokenizer_r)

            # Check for offsets mapping
            self.assert_offsets_mapping(tokenizer_r)

            # Check for dynamic encoding sequence handling in batch_encode_plus
            self.assert_batch_encode_dynamic_overflowing(tokenizer_r)

            # Check alignment for build_inputs_with_special_tokens
            self.assert_build_inputs_with_special_tokens(tokenizer_r, tokenizer_p)

            # Check the number of returned files for save_vocabulary
            self.assert_save_pretrained(tokenizer_r, tokenizer_p)

            # Check for padding
            # TODO: Re-enable this test as soon as Roberta align with the python tokenizer.
            # self.assert_padding(tokenizer_r, tokenizer_p)

    def test_openai(self):
        for tokenizer_name in OpenAIGPTTokenizer.pretrained_vocab_files_map["vocab_file"].keys():
            tokenizer_p = OpenAIGPTTokenizer.from_pretrained(tokenizer_name)
            tokenizer_r = OpenAIGPTTokenizerFast.from_pretrained(tokenizer_name)

            # Check we have the same number of added_tokens for both pair and non-pair inputs.
            self.assertEqual(tokenizer_r.num_added_tokens(False), tokenizer_p.num_added_tokens(False))
            self.assertEqual(tokenizer_r.num_added_tokens(True), tokenizer_p.num_added_tokens(True))

            # Check we have the correct max_length for both pair and non-pair inputs.
            self.assertEqual(tokenizer_r.max_len_single_sentence, tokenizer_p.max_len_single_sentence)
            self.assertEqual(tokenizer_r.max_len_sentences_pair, tokenizer_p.max_len_sentences_pair)

            # Assert the set of special tokens match.
            self.assertSequenceEqual(
                tokenizer_p.special_tokens_map.items(),
                tokenizer_r.special_tokens_map.items(),
                "GPT tokenizers doesn't have the same set of special_tokens",
            )

            # Assure tokenization overlap between python and rust impl.
            self.assert_tokenization_python_rust_almost_equals(tokenizer_p, tokenizer_r, 0.0)

            # Ensure add_tokens and add_special_tokens return the correct vocab size
            self.assert_add_tokens(tokenizer_r)

            # Check for offsets mapping
            self.assert_offsets_mapping(tokenizer_r)

            # Check for dynamic encoding sequence handling in batch_encode_plus
            self.assertRaises(ValueError, self.assert_batch_encode_dynamic_overflowing, tokenizer_r)

            # Check alignment for build_inputs_with_special_tokens
            self.assert_build_inputs_with_special_tokens(tokenizer_r, tokenizer_p)

            self.assertEqual(len(tokenizer_r.save_vocabulary(".")), len(tokenizer_p.save_vocabulary(".")))

            # Check for padding
            self.assertRaises(ValueError, self.assert_padding, tokenizer_r, tokenizer_p)

            # Check the number of returned files for save_vocabulary
            self.assert_save_pretrained(tokenizer_r, tokenizer_p)

    def test_embedded_special_tokens(self):
        sentence = "A, <mask> AllenNLP sentence."
        from transformers import AutoTokenizer

        tokenizer_r = AutoTokenizer.from_pretrained("roberta-base", add_special_tokens=True, use_fast=True)
        tokenizer_p = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)

        tokens_r = tokenizer_r.encode_plus(sentence, return_attention_mask=False, return_token_type_ids=True)
        tokens_p = tokenizer_p.encode_plus(
            sentence, add_special_tokens=True, return_attention_mask=False, return_token_type_ids=True
        )
        type_ids_r = tokens_r["token_type_ids"]
        type_ids_p = tokens_p["token_type_ids"]

        expected_tokens = [
            "<s>",
            "A",
            ",",
            "Ġ",
            "<mask>",
            "ĠAllen",
            "N",
            "LP",
            "Ġsentence",
            ".",
            "</s>",
        ]

        tokens_r = tokenizer_r.convert_ids_to_tokens(tokens_r["input_ids"])
        tokens_p = tokenizer_p.convert_ids_to_tokens(tokens_p["input_ids"])
        self.assertEqual(expected_tokens, tokens_r)
        self.assertEqual(expected_tokens, tokens_p)

        self.assertEqual([0] * len(expected_tokens), type_ids_r)
        self.assertEqual([0] * len(expected_tokens), type_ids_p)

    def test_embedded_special_tokens_pair(self):
        sentence_1 = "A, [MASK] AllenNLP sentence."
        sentence_2 = "A sentence."
        from transformers import AutoTokenizer

        tokenizer_r = AutoTokenizer.from_pretrained("bert-base-cased", add_special_tokens=True, use_fast=True)
        tokenizer_p = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)

        tokens_r = tokenizer_r.encode_plus(
            sentence_1, sentence_2, return_attention_mask=False, return_token_type_ids=True
        )
        tokens_p = tokenizer_p.encode_plus(
            sentence_1, sentence_2, add_special_tokens=True, return_attention_mask=False, return_token_type_ids=True
        )
        type_ids_r = tokens_r["token_type_ids"]
        type_ids_p = tokens_p["token_type_ids"]

        expected_result = [
            (0, "[CLS]"),
            (0, "A"),
            (0, ","),
            (0, "[MASK]"),
            (0, "Allen"),
            (0, "##NL"),
            (0, "##P"),
            (0, "sentence"),
            (0, "."),
            (0, "[SEP]"),
            (1, "A"),
            (1, "sentence"),
            (1, "."),
            (1, "[SEP]"),
        ]

        tokens_r = tokenizer_r.convert_ids_to_tokens(tokens_r["input_ids"])
        tokens_p = tokenizer_p.convert_ids_to_tokens(tokens_p["input_ids"])
        self.assertEqual([e[1] for e in expected_result], tokens_r)
        self.assertEqual([e[1] for e in expected_result], tokens_p)

        self.assertEqual([e[0] for e in expected_result], type_ids_r)
        self.assertEqual([e[0] for e in expected_result], type_ids_p)

    def test_embedded_special_tokens_pair_roberta(self):
        sentence_1 = "A, <mask> AllenNLP sentence."
        sentence_2 = "A sentence."
        from transformers import AutoTokenizer

        tokenizer_r = AutoTokenizer.from_pretrained("roberta-base", add_special_tokens=True, use_fast=True)
        tokenizer_p = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)

        tokens_r = tokenizer_r.encode_plus(
            sentence_1, sentence_2, return_attention_mask=False, return_token_type_ids=True
        )
        tokens_p = tokenizer_p.encode_plus(
            sentence_1, sentence_2, add_special_tokens=True, return_attention_mask=False, return_token_type_ids=True
        )
        type_ids_r = tokens_r["token_type_ids"]
        type_ids_p = tokens_p["token_type_ids"]

        expected_tokens = [
            "<s>",
            "A",
            ",",
            "Ġ",
            "<mask>",
            "ĠAllen",
            "N",
            "LP",
            "Ġsentence",
            ".",
            "</s>",
            "</s>",
            "A",
            "Ġsentence",
            ".",
            "</s>",
        ]

        tokens_r = tokenizer_r.convert_ids_to_tokens(tokens_r["input_ids"])
        tokens_p = tokenizer_p.convert_ids_to_tokens(tokens_p["input_ids"])
        self.assertEqual(expected_tokens, tokens_r)
        self.assertEqual(expected_tokens, tokens_p)

        self.assertEqual([0] * len(expected_tokens), type_ids_r)
        self.assertEqual([0] * len(expected_tokens), type_ids_p)

    def test_offsets_with_special_characters(self):
        from transformers import AutoTokenizer

        sentence = "A, naïve [MASK] AllenNLP sentence."
        for model in ["bert-base-cased", "bert-base-uncased"]:
            with self.subTest(model=model):
                tokenizer = AutoTokenizer.from_pretrained(model, add_special_tokens=True, use_fast=True)

                tokens = tokenizer.encode_plus(
                    sentence, return_attention_mask=False, return_token_type_ids=False, return_offsets_mapping=True
                )

                expected_results = [
                    (None, "[CLS]"),
                    ((0, 1), "A"),
                    ((1, 2), ","),
                    ((3, 8), "naive"),  # BERT normalizes this away
                    ((9, 15), "[MASK]"),
                    ((16, 21), "Allen"),
                    ((22, 24), "##NL"),
                    ((24, 25), "##P"),
                    ((26, 34), "sentence"),
                    ((35, 36), "."),
                    (None, "[SEP]"),
                ]

                self.assertEqual(
                    [e[1] for e in expected_results], tokenizer.convert_ids_to_tokens(tokens["input_ids"])
                )
                self.assertEqual([e[0] for e in expected_results], tokens["offset_mapping"])

    def test_offsets_with_special_characters_roberta(self):
        from transformers import AutoTokenizer

        sentence = "A, naïve <mask> AllenNLP sentence."
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_special_tokens=True, use_fast=True)

        tokens = tokenizer.encode_plus(
            sentence, return_attention_mask=False, return_token_type_ids=False, return_offsets_mapping=True
        )

        expected_results = [
            (None, "<s>"),
            ((0, 1), "A"),
            ((1, 2), ","),
            ((3, 8), "ĠnaÃ¯ve"),  # RoBERTa mangles this
            ((8, 9), "Ġ"),
            ((9, 15), "<mask>"),
            ((16, 21), "ĠAllen"),
            ((22, 23), "N"),
            ((23, 25), "LP"),
            ((26, 34), "Ġsentence"),
            ((35, 36), "."),
            (None, "</s>"),
        ]

        self.assertEqual([e[1] for e in expected_results], tokenizer.convert_ids_to_tokens(tokens["input_ids"]))
        self.assertEqual([e[0] for e in expected_results], tokens["offset_mapping"])

    def test_encode_plus_with_ids(self):
        from transformers import AutoTokenizer
        for fast in [False, True]:
            with self.subTest(fast=fast):
                tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=fast)
                results = tokenizer.encode_plus([1000])
                self.assertEqual([101, 1000, 102], results['input_ids'])