import logging
import unittest
from collections import namedtuple
from itertools import takewhile

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
from transformers.testing_utils import require_torch
from transformers.tokenization_distilbert import DistilBertTokenizerFast
from transformers.tokenization_openai import OpenAIGPTTokenizerFast
from transformers.tokenization_roberta import RobertaTokenizerFast
from transformers.tokenization_transfo_xl import TransfoXLTokenizerFast


logger = logging.getLogger(__name__)

NON_ENGLISH_TAGS = ["chinese", "dutch", "french", "finnish", "german", "multilingual"]
Tokenizer = namedtuple("Tokenizer", ["name", "rust_cls", "python_cls", "vocab_key", "filter", "kwargs"])


def filter_non_english(_: Tokenizer, pretrained_name: str):
    """ Filter all the model for non-english language """
    return not any([lang in pretrained_name for lang in NON_ENGLISH_TAGS])


def filter_roberta_detectors(_: Tokenizer, pretrained_name: str):
    return "detector" not in pretrained_name


class CommonFastTokenizerTest(unittest.TestCase):

    TOKENIZERS_CLASSES = frozenset([])

    def setUp(self) -> None:
        with open("tests/fixtures/sample_text.txt", encoding="utf-8") as f_data:
            self._data = f_data.read().replace("\n\n", "\n").strip()

    def test_all_tokenizers(self):
        for tok_case in self.TOKENIZERS_CLASSES:
            for pretrained_name in tok_case.python_cls.pretrained_vocab_files_map[tok_case.vocab_key].keys():

                # Tokenizer.filter makes it possible to filter which Tokenizer to case based on all the
                # information available in Tokenizer (name, rust class, python class, vocab key name)
                if tok_case.filter is None or (
                    tok_case.filter is not None and tok_case.filter(tok_case, pretrained_name)
                ):
                    kwargs = dict(t for t in tok_case.kwargs) if tok_case.kwargs else {}
                    with self.subTest("{} ({})".format(tok_case.name, pretrained_name)):
                        tokenizer_r = tok_case.rust_cls.from_pretrained(pretrained_name, **kwargs)
                        tokenizer_p = tok_case.python_cls.from_pretrained(pretrained_name, **kwargs)

                        self.fast_align_python(tokenizer_r, tokenizer_p, tok_case, pretrained_name)
                        self.fast_only(tokenizer_r)

    def test_pretokenized_tokenizers(self):
        for tok_case in self.TOKENIZERS_CLASSES:
            for pretrained_name in tok_case.python_cls.pretrained_vocab_files_map[tok_case.vocab_key].keys():

                # Tokenizer.filter makes it possible to filter which Tokenizer to case based on all the
                # information available in Tokenizer (name, rust class, python class, vocab key name)
                if tok_case.filter is None or (
                    tok_case.filter is not None and tok_case.filter(tok_case, pretrained_name)
                ):
                    with self.subTest("{} ({})".format(tok_case.name, pretrained_name)):
                        tokenizer_r = tok_case.rust_cls.from_pretrained(pretrained_name, add_prefix_space=True)
                        tokenizer_p = tok_case.python_cls.from_pretrained(pretrained_name, add_prefix_space=True)

                        self.assert_pretokenized_inputs(tokenizer_r, tokenizer_p)

    def fast_align_python(self, tokenizer_r, tokenizer_p, tok_case, pretrained_name):
        # Check is_fast is set correctly
        self.assertFalse(tokenizer_p.is_fast)
        self.assertTrue(tokenizer_r.is_fast)

        # Check that Rust and Python align
        self.assert_tokenization_python_rust_equals(tokenizer_r, tokenizer_p)
        self.assert_num_special_tokens_to_add_equal(tokenizer_r, tokenizer_p)
        self.assert_max_length_equal(tokenizer_r, tokenizer_p)
        self.assert_special_tokens_map_equal(tokenizer_r, tokenizer_p)
        self.assert_embeded_special_tokens(tokenizer_r, tokenizer_p)
        self.assert_padding(tokenizer_r, tokenizer_p)
        self.assert_create_token_type_ids(tokenizer_r, tokenizer_p)
        self.assert_prepare_for_model(tokenizer_r, tokenizer_p)

    def fast_only(self, tokenizer_r):
        # Ensure None raise an error
        self.assertRaises(ValueError, tokenizer_r.tokenize, None)
        self.assertRaises(ValueError, tokenizer_r.encode, None)
        self.assertRaises(ValueError, tokenizer_r.encode_plus, None)
        self.assertRaises(ValueError, tokenizer_r.batch_encode_plus, None)

        self.assert_add_tokens(tokenizer_r)
        self.assert_offsets_mapping(tokenizer_r)
        self.assert_add_special_tokens(tokenizer_r)
        self.assert_alignement_methods(tokenizer_r)
        self.assert_batch_encode_dynamic_overflowing(tokenizer_r)

    def assert_alignement_methods(self, tokenizer_r):
        words = ["Wonderful", "no", "inspiration", "example", "with", "subtoken"]
        text = " ".join(words)
        batch_size = 3

        encoding = tokenizer_r.encode_plus(text, add_special_tokens=False)

        batch_encoding = tokenizer_r.batch_encode_plus([text] * batch_size, add_special_tokens=False)
        num_tokens = len(encoding["input_ids"])

        last_word_index = len(words) - 1
        last_token_index = num_tokens - 1
        last_batch_index = batch_size - 1
        last_char_index = len(text) - 1

        # words, tokens
        self.assertEqual(len(encoding.words(0)), num_tokens)
        self.assertEqual(max(encoding.words(0)), last_word_index)
        self.assertEqual(min(encoding.words(0)), 0)
        self.assertEqual(len(batch_encoding.words(last_batch_index)), num_tokens)
        self.assertEqual(max(batch_encoding.words(last_batch_index)), last_word_index)
        self.assertEqual(min(batch_encoding.words(last_batch_index)), 0)
        self.assertEqual(len(encoding.tokens(0)), num_tokens)

        # Assert token_to_word
        self.assertEqual(encoding.token_to_word(0), 0)
        self.assertEqual(encoding.token_to_word(0, 0), 0)
        self.assertEqual(encoding.token_to_word(last_token_index), last_word_index)
        self.assertEqual(encoding.token_to_word(0, last_token_index), last_word_index)
        self.assertEqual(batch_encoding.token_to_word(1, 0), 0)
        self.assertEqual(batch_encoding.token_to_word(0, last_token_index), last_word_index)
        self.assertEqual(batch_encoding.token_to_word(last_batch_index, last_token_index), last_word_index)

        # Assert word_to_tokens
        self.assertEqual(encoding.word_to_tokens(0).start, 0)
        self.assertEqual(encoding.word_to_tokens(0, 0).start, 0)
        self.assertEqual(encoding.word_to_tokens(last_word_index).end, last_token_index + 1)
        self.assertEqual(encoding.word_to_tokens(0, last_word_index).end, last_token_index + 1)
        self.assertEqual(batch_encoding.word_to_tokens(1, 0).start, 0)
        self.assertEqual(batch_encoding.word_to_tokens(0, last_word_index).end, last_token_index + 1)
        self.assertEqual(batch_encoding.word_to_tokens(last_batch_index, last_word_index).end, last_token_index + 1)

        # Assert token_to_chars
        self.assertEqual(encoding.token_to_chars(0).start, 0)
        self.assertEqual(encoding.token_to_chars(0, 0).start, 0)
        self.assertEqual(encoding.token_to_chars(last_token_index).end, last_char_index + 1)
        self.assertEqual(encoding.token_to_chars(0, last_token_index).end, last_char_index + 1)
        self.assertEqual(batch_encoding.token_to_chars(1, 0).start, 0)
        self.assertEqual(batch_encoding.token_to_chars(0, last_token_index).end, last_char_index + 1)
        self.assertEqual(batch_encoding.token_to_chars(last_batch_index, last_token_index).end, last_char_index + 1)

        # Assert char_to_token
        self.assertEqual(encoding.char_to_token(0), 0)
        self.assertEqual(encoding.char_to_token(0, 0), 0)
        self.assertEqual(encoding.char_to_token(last_char_index), last_token_index)
        self.assertEqual(encoding.char_to_token(0, last_char_index), last_token_index)
        self.assertEqual(batch_encoding.char_to_token(1, 0), 0)
        self.assertEqual(batch_encoding.char_to_token(0, last_char_index), last_token_index)
        self.assertEqual(batch_encoding.char_to_token(last_batch_index, last_char_index), last_token_index)

        # Assert char_to_word
        self.assertEqual(encoding.char_to_word(0), 0)
        self.assertEqual(encoding.char_to_word(0, 0), 0)
        self.assertEqual(encoding.char_to_word(last_char_index), last_word_index)
        self.assertEqual(encoding.char_to_word(0, last_char_index), last_word_index)
        self.assertEqual(batch_encoding.char_to_word(1, 0), 0)
        self.assertEqual(batch_encoding.char_to_word(0, last_char_index), last_word_index)
        self.assertEqual(batch_encoding.char_to_word(last_batch_index, last_char_index), last_word_index)

        # Assert word_to_chars
        self.assertEqual(encoding.word_to_chars(0).start, 0)
        self.assertEqual(encoding.word_to_chars(0, 0).start, 0)
        self.assertEqual(encoding.word_to_chars(last_word_index).end, last_char_index + 1)
        self.assertEqual(encoding.word_to_chars(0, last_word_index).end, last_char_index + 1)
        self.assertEqual(batch_encoding.word_to_chars(1, 0).start, 0)
        self.assertEqual(batch_encoding.word_to_chars(0, last_word_index).end, last_char_index + 1)
        self.assertEqual(batch_encoding.word_to_chars(last_batch_index, last_word_index).end, last_char_index + 1)

    def assert_tokenization_python_rust_equals(self, tokenizer_r, tokenizer_p):
        # Ensure basic input match
        input_p = tokenizer_p.encode_plus(self._data)
        input_r = tokenizer_r.encode_plus(self._data)

        for key in filter(lambda x: x in ["input_ids", "token_type_ids", "attention_mask"], input_p.keys()):
            self.assertSequenceEqual(input_p[key], input_r[key])

        input_pairs_p = tokenizer_p.encode_plus(self._data, self._data)
        input_pairs_r = tokenizer_r.encode_plus(self._data, self._data)

        for key in filter(lambda x: x in ["input_ids", "token_type_ids", "attention_mask"], input_p.keys()):
            self.assertSequenceEqual(input_pairs_p[key], input_pairs_r[key])

        # Ensure truncation match
        input_p = tokenizer_p.encode_plus(self._data, max_length=512, truncation=True)
        input_r = tokenizer_r.encode_plus(self._data, max_length=512, truncation=True)

        for key in filter(lambda x: x in ["input_ids", "token_type_ids", "attention_mask"], input_p.keys()):
            self.assertSequenceEqual(input_p[key], input_r[key])

        # Ensure truncation with stride match
        input_p = tokenizer_p.encode_plus(
            self._data, max_length=512, truncation=True, stride=3, return_overflowing_tokens=True
        )
        input_r = tokenizer_r.encode_plus(
            self._data, max_length=512, truncation=True, stride=3, return_overflowing_tokens=True
        )

        for key in filter(lambda x: x in ["input_ids", "token_type_ids", "attention_mask"], input_p.keys()):
            self.assertSequenceEqual(input_p[key], input_r[key][0])

    def assert_num_special_tokens_to_add_equal(self, tokenizer_r, tokenizer_p):
        # Check we have the same number of added_tokens for both pair and non-pair inputs.
        self.assertEqual(tokenizer_r.num_special_tokens_to_add(False), tokenizer_p.num_special_tokens_to_add(False))
        self.assertEqual(tokenizer_r.num_special_tokens_to_add(True), tokenizer_p.num_special_tokens_to_add(True))

    def assert_max_length_equal(self, tokenizer_r, tokenizer_p):
        # Check we have the correct max_length for both pair and non-pair inputs.
        self.assertEqual(tokenizer_r.max_len_single_sentence, tokenizer_p.max_len_single_sentence)
        self.assertEqual(tokenizer_r.max_len_sentences_pair, tokenizer_p.max_len_sentences_pair)

    def assert_special_tokens_map_equal(self, tokenizer_r, tokenizer_p):
        # Assert the set of special tokens match.
        self.assertSequenceEqual(
            tokenizer_p.special_tokens_map.items(), tokenizer_r.special_tokens_map.items(),
        )

    def assert_add_tokens(self, tokenizer_r):
        vocab_size = tokenizer_r.vocab_size
        self.assertEqual(tokenizer_r.add_tokens(""), 0)
        self.assertEqual(tokenizer_r.add_tokens("testoken"), 1)
        self.assertEqual(tokenizer_r.add_tokens(["testoken1", "testtoken2"]), 2)
        self.assertEqual(len(tokenizer_r), vocab_size + 3)

        self.assertEqual(tokenizer_r.add_special_tokens({}), 0)
        self.assertEqual(tokenizer_r.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"}), 2)
        self.assertRaises(
            AssertionError, tokenizer_r.add_special_tokens, {"additional_special_tokens": "<testtoken1>"}
        )
        self.assertEqual(tokenizer_r.add_special_tokens({"additional_special_tokens": ["<testtoken2>"]}), 1)
        self.assertEqual(
            tokenizer_r.add_special_tokens({"additional_special_tokens": ["<testtoken3>", "<testtoken4>"]}), 2
        )
        self.assertEqual(len(tokenizer_r), vocab_size + 8)

    def assert_offsets_mapping(self, tokenizer_r):
        text = "Wonderful no inspiration example with subtoken"
        pair = "Along with an awesome pair"

        # No pair
        tokens_with_offsets = tokenizer_r.encode_plus(
            text, return_special_tokens_mask=True, return_offsets_mapping=True, add_special_tokens=True
        )
        added_tokens = tokenizer_r.num_special_tokens_to_add(False)
        offsets = tokens_with_offsets["offset_mapping"]

        # Assert there is the same number of tokens and offsets
        self.assertEqual(len(offsets), len(tokens_with_offsets["input_ids"]))

        # Assert there is online added_tokens special_tokens
        self.assertEqual(sum(tokens_with_offsets["special_tokens_mask"]), added_tokens)

        # Pairs
        tokens_with_offsets = tokenizer_r.encode_plus(
            text, pair, return_special_tokens_mask=True, return_offsets_mapping=True, add_special_tokens=True
        )
        added_tokens = tokenizer_r.num_special_tokens_to_add(True)
        offsets = tokens_with_offsets["offset_mapping"]

        # Assert there is the same number of tokens and offsets
        self.assertEqual(len(offsets), len(tokens_with_offsets["input_ids"]))

        # Assert there is online added_tokens special_tokens
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

        if not tokenizer.pad_token or tokenizer.pad_token_id < 0:
            return

        tokens = tokenizer.encode_plus(
            "HuggingFace is solving NLP one commit at a time",
            max_length=6,
            padding=True,
            truncation=True,
            return_tensors=returned_tensor,
            return_overflowing_tokens=True,
        )

        for key in filter(lambda x: "overflow_to_sample_mapping" not in x, tokens.keys()):
            self.assertEqual(len(tokens[key].shape), 2)

        # Mono sample
        tokens = tokenizer.batch_encode_plus(
            ["HuggingFace is solving NLP one commit at a time"],
            max_length=6,
            padding=True,
            truncation="only_first",
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
            padding=True,
            truncation="only_first",
            return_tensors=returned_tensor,
            return_overflowing_tokens=True,
        )

        for key in filter(lambda x: "overflow_to_sample_mapping" not in x, tokens.keys()):
            self.assertEqual(len(tokens[key].shape), 2)
            self.assertEqual(tokens[key].shape[-1], 6)

    def assert_pretokenized_inputs(self, tokenizer_r, tokenizer_p):
        # Input string
        pretokenized_input_simple = "This is a sample input".split()
        pretokenized_input_pair = "This is a sample pair".split()

        # Test encode for pretokenized inputs
        output_r = tokenizer_r.encode(pretokenized_input_simple, is_pretokenized=True)
        output_p = tokenizer_p.encode(pretokenized_input_simple, is_pretokenized=True)
        self.assertEqual(output_p, output_r)

        kwargs = {
            "is_pretokenized": True,
            "return_token_type_ids": True,
            "return_attention_mask": True,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": True,
            "return_offsets_mapping": False,  # Not implemented in python tokenizers
        }
        batch_kwargs = {
            "is_pretokenized": True,
            "return_token_type_ids": True,
            "return_attention_mask": True,  # we have an 's' here
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": True,  # we have an 's' here
            "return_offsets_mapping": False,  # Not implemented in python tokenizers
        }
        # Test encode_plus for pretokenized inputs
        output_r = tokenizer_r.encode_plus(pretokenized_input_simple, **kwargs)
        output_p = tokenizer_p.encode_plus(pretokenized_input_simple, **kwargs)
        for key in output_p.keys():
            self.assertEqual(output_p[key], output_r[key])

        # Test batch_encode_plus for pretokenized inputs
        input_batch = ([pretokenized_input_simple] * 2) + [pretokenized_input_simple + pretokenized_input_pair]
        output_r = tokenizer_r.batch_encode_plus(input_batch, **batch_kwargs)
        output_p = tokenizer_p.batch_encode_plus(input_batch, **batch_kwargs)
        for key in output_p.keys():
            self.assertEqual(output_p[key], output_r[key])

        # Test encode for pretokenized inputs pairs
        output_r = tokenizer_r.encode(pretokenized_input_simple, pretokenized_input_pair, is_pretokenized=True)
        output_p = tokenizer_p.encode(pretokenized_input_simple, pretokenized_input_pair, is_pretokenized=True)
        self.assertEqual(output_p, output_r)

        # Test encode_plus for pretokenized inputs
        output_r = tokenizer_r.encode_plus(pretokenized_input_simple, pretokenized_input_pair, **kwargs)
        output_p = tokenizer_p.encode_plus(pretokenized_input_simple, pretokenized_input_pair, **kwargs)
        for key in output_p.keys():
            self.assertEqual(output_p[key], output_r[key])

        # Test batch_encode_plus for pretokenized inputs
        input_batch_pair = ([pretokenized_input_simple, pretokenized_input_pair] * 2) + [
            pretokenized_input_simple + pretokenized_input_pair,
            pretokenized_input_pair,
        ]
        output_r = tokenizer_r.batch_encode_plus(input_batch_pair, **batch_kwargs)
        output_p = tokenizer_p.batch_encode_plus(input_batch_pair, **batch_kwargs)
        for key in output_p.keys():
            self.assertEqual(output_p[key], output_r[key])

    def assert_create_token_type_ids(self, tokenizer_r, tokenizer_p):
        input_simple = [1, 2, 3]
        input_pair = [1, 2, 3]

        # Generate output
        output_r = tokenizer_r.create_token_type_ids_from_sequences(input_simple)
        output_p = tokenizer_p.create_token_type_ids_from_sequences(input_simple)
        self.assertEqual(output_p, output_r)

        # Generate pair output
        output_r = tokenizer_r.create_token_type_ids_from_sequences(input_simple, input_pair)
        output_p = tokenizer_p.create_token_type_ids_from_sequences(input_simple, input_pair)
        self.assertEqual(output_p, output_r)

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

    def assert_padding(self, tokenizer_r, tokenizer_p, max_length=15):
        def assert_padded_input_match(input_r: list, input_p: list, max_length: int):

            # Ensure we match max_length
            self.assertEqual(len(input_r), max_length)
            self.assertEqual(len(input_p), max_length)

            # Ensure the number of padded tokens is the same
            padded_tokens_r = list(takewhile(lambda i: i == tokenizer_r.pad_token_id, reversed(input_r)))
            padded_tokens_p = list(takewhile(lambda i: i == tokenizer_p.pad_token_id, reversed(input_p)))
            self.assertSequenceEqual(padded_tokens_r, padded_tokens_p)

        def assert_batch_padded_input_match(input_r: dict, input_p: dict, max_length: int):
            for i_r in input_r.values():
                self.assertEqual(len(i_r), 2), self.assertEqual(len(i_r[0]), max_length), self.assertEqual(
                    len(i_r[1]), max_length
                )
                self.assertEqual(len(i_r), 2), self.assertEqual(len(i_r[0]), max_length), self.assertEqual(
                    len(i_r[1]), max_length
                )

            for i_r, i_p in zip(input_r["input_ids"], input_p["input_ids"]):
                assert_padded_input_match(i_r, i_p, max_length)

            for i_r, i_p in zip(input_r["attention_mask"], input_p["attention_mask"]):
                self.assertSequenceEqual(i_r, i_p)

        # Encode - Simple input
        input_r = tokenizer_r.encode("This is a simple input", max_length=max_length, pad_to_max_length=True)
        input_p = tokenizer_p.encode("This is a simple input", max_length=max_length, pad_to_max_length=True)
        assert_padded_input_match(input_r, input_p, max_length)
        input_r = tokenizer_r.encode("This is a simple input", max_length=max_length, padding="max_length")
        input_p = tokenizer_p.encode("This is a simple input", max_length=max_length, padding="max_length")
        assert_padded_input_match(input_r, input_p, max_length)

        input_r = tokenizer_r.encode("This is a simple input", padding="longest")
        input_p = tokenizer_p.encode("This is a simple input", padding=True)
        assert_padded_input_match(input_r, input_p, len(input_r))

        # Encode - Pair input
        input_r = tokenizer_r.encode(
            "This is a simple input", "This is a pair", max_length=max_length, pad_to_max_length=True
        )
        input_p = tokenizer_p.encode(
            "This is a simple input", "This is a pair", max_length=max_length, pad_to_max_length=True
        )
        assert_padded_input_match(input_r, input_p, max_length)
        input_r = tokenizer_r.encode(
            "This is a simple input", "This is a pair", max_length=max_length, padding="max_length"
        )
        input_p = tokenizer_p.encode(
            "This is a simple input", "This is a pair", max_length=max_length, padding="max_length"
        )
        assert_padded_input_match(input_r, input_p, max_length)
        input_r = tokenizer_r.encode("This is a simple input", "This is a pair", padding=True)
        input_p = tokenizer_p.encode("This is a simple input", "This is a pair", padding="longest")
        assert_padded_input_match(input_r, input_p, len(input_r))

        # Encode_plus - Simple input
        input_r = tokenizer_r.encode_plus("This is a simple input", max_length=max_length, pad_to_max_length=True)
        input_p = tokenizer_p.encode_plus("This is a simple input", max_length=max_length, pad_to_max_length=True)
        assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length)
        self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])
        input_r = tokenizer_r.encode_plus("This is a simple input", max_length=max_length, padding="max_length")
        input_p = tokenizer_p.encode_plus("This is a simple input", max_length=max_length, padding="max_length")
        assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length)
        self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])

        input_r = tokenizer_r.encode_plus("This is a simple input", padding="longest")
        input_p = tokenizer_p.encode_plus("This is a simple input", padding=True)
        assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]))

        self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])

        # Encode_plus - Pair input
        input_r = tokenizer_r.encode_plus(
            "This is a simple input", "This is a pair", max_length=max_length, pad_to_max_length=True
        )
        input_p = tokenizer_p.encode_plus(
            "This is a simple input", "This is a pair", max_length=max_length, pad_to_max_length=True
        )
        assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length)
        self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])
        input_r = tokenizer_r.encode_plus(
            "This is a simple input", "This is a pair", max_length=max_length, padding="max_length"
        )
        input_p = tokenizer_p.encode_plus(
            "This is a simple input", "This is a pair", max_length=max_length, padding="max_length"
        )
        assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length)
        self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])
        input_r = tokenizer_r.encode_plus("This is a simple input", "This is a pair", padding="longest")
        input_p = tokenizer_p.encode_plus("This is a simple input", "This is a pair", padding=True)
        assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]))
        self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])

        # Batch_encode_plus - Simple input
        input_r = tokenizer_r.batch_encode_plus(
            ["This is a simple input 1", "This is a simple input 2"], max_length=max_length, pad_to_max_length=True
        )
        input_p = tokenizer_p.batch_encode_plus(
            ["This is a simple input 1", "This is a simple input 2"], max_length=max_length, pad_to_max_length=True
        )
        assert_batch_padded_input_match(input_r, input_p, max_length)

        input_r = tokenizer_r.batch_encode_plus(
            ["This is a simple input 1", "This is a simple input 2"], max_length=max_length, padding="max_length",
        )
        input_p = tokenizer_p.batch_encode_plus(
            ["This is a simple input 1", "This is a simple input 2"], max_length=max_length, padding="max_length",
        )
        assert_batch_padded_input_match(input_r, input_p, max_length)

        input_r = tokenizer_r.batch_encode_plus(
            ["This is a simple input 1", "This is a simple input 2"], max_length=max_length, padding="longest",
        )
        input_p = tokenizer_p.batch_encode_plus(
            ["This is a simple input 1", "This is a simple input 2"], max_length=max_length, padding=True,
        )
        assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]))

        input_r = tokenizer_r.batch_encode_plus(
            ["This is a simple input 1", "This is a simple input 2"], padding="longest"
        )
        input_p = tokenizer_p.batch_encode_plus(["This is a simple input 1", "This is a simple input 2"], padding=True)
        assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]))

        # Batch_encode_plus - Pair input
        input_r = tokenizer_r.batch_encode_plus(
            [
                ("This is a simple input 1", "This is a simple input 2"),
                ("This is a simple pair 1", "This is a simple pair 2"),
            ],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_p = tokenizer_p.batch_encode_plus(
            [
                ("This is a simple input 1", "This is a simple input 2"),
                ("This is a simple pair 1", "This is a simple pair 2"),
            ],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        assert_batch_padded_input_match(input_r, input_p, max_length)

        input_r = tokenizer_r.batch_encode_plus(
            [
                ("This is a simple input 1", "This is a simple input 2"),
                ("This is a simple pair 1", "This is a simple pair 2"),
            ],
            padding=True,
        )
        input_p = tokenizer_p.batch_encode_plus(
            [
                ("This is a simple input 1", "This is a simple input 2"),
                ("This is a simple pair 1", "This is a simple pair 2"),
            ],
            padding="longest",
        )
        assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]))

        # Using pad on single examples after tokenization
        input_r = tokenizer_r.encode_plus("This is a input 1")
        input_r = tokenizer_r.pad(input_r)

        input_p = tokenizer_r.encode_plus("This is a input 1")
        input_p = tokenizer_r.pad(input_p)

        assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]))

        # Using pad on single examples after tokenization
        input_r = tokenizer_r.encode_plus("This is a input 1")
        input_r = tokenizer_r.pad(input_r, max_length=max_length, padding="max_length")

        input_p = tokenizer_r.encode_plus("This is a input 1")
        input_p = tokenizer_r.pad(input_p, max_length=max_length, padding="max_length")

        assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length)

        # Using pad after tokenization
        input_r = tokenizer_r.batch_encode_plus(
            ["This is a input 1", "This is a much longer input whilch should be padded"]
        )
        input_r = tokenizer_r.pad(input_r)

        input_p = tokenizer_r.batch_encode_plus(
            ["This is a input 1", "This is a much longer input whilch should be padded"]
        )
        input_p = tokenizer_r.pad(input_p)

        assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]))

        # Using pad after tokenization
        input_r = tokenizer_r.batch_encode_plus(
            ["This is a input 1", "This is a much longer input whilch should be padded"]
        )
        input_r = tokenizer_r.pad(input_r, max_length=max_length, padding="max_length")

        input_p = tokenizer_r.batch_encode_plus(
            ["This is a input 1", "This is a much longer input whilch should be padded"]
        )
        input_p = tokenizer_r.pad(input_p, max_length=max_length, padding="max_length")

        assert_batch_padded_input_match(input_r, input_p, max_length)

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

    def assert_embeded_special_tokens(self, tokenizer_r, tokenizer_p):
        sentence = "A, <mask> AllenNLP sentence."
        tokens_r = tokenizer_r.encode_plus(
            sentence, add_special_tokens=True, return_attention_mask=False, return_token_type_ids=True
        )
        tokens_p = tokenizer_p.encode_plus(
            sentence, add_special_tokens=True, return_attention_mask=False, return_token_type_ids=True
        )

        for key in tokens_p.keys():
            self.assertEqual(tokens_r[key], tokens_p[key])

        self.assertEqual(sum(tokens_r["token_type_ids"]), 0)
        self.assertEqual(sum(tokens_p["token_type_ids"]), 0)

        tokens_r = tokenizer_r.convert_ids_to_tokens(tokens_r["input_ids"])
        tokens_p = tokenizer_p.convert_ids_to_tokens(tokens_p["input_ids"])
        self.assertSequenceEqual(tokens_r, tokens_p)

    def assert_add_special_tokens(self, tokenizer_r):
        simple_num_special_tokens_to_add = tokenizer_r.num_special_tokens_to_add(pair=False)
        # pair_num_special_tokens_to_add = tokenizer_r.num_special_tokens_to_add(pair=True)

        for text in ["", " "]:
            # tokenize()
            no_special_tokens = tokenizer_r.tokenize(text, add_special_tokens=False)
            with_special_tokens = tokenizer_r.tokenize(text, add_special_tokens=True)
            self.assertEqual(len(no_special_tokens), len(with_special_tokens) - simple_num_special_tokens_to_add)

            # encode()
            no_special_tokens = tokenizer_r.encode(text, add_special_tokens=False)
            with_special_tokens = tokenizer_r.encode(text, add_special_tokens=True)
            self.assertEqual(len(no_special_tokens), len(with_special_tokens) - simple_num_special_tokens_to_add)

            # encode_plus()
            no_special_tokens = tokenizer_r.encode_plus(text, add_special_tokens=False)
            with_special_tokens = tokenizer_r.encode_plus(text, add_special_tokens=True)
            for key in no_special_tokens.keys():
                self.assertEqual(
                    len(no_special_tokens[key]), len(with_special_tokens[key]) - simple_num_special_tokens_to_add
                )

            # # batch_encode_plus
            no_special_tokens = tokenizer_r.batch_encode_plus([text, text], add_special_tokens=False)
            with_special_tokens = tokenizer_r.batch_encode_plus([text, text], add_special_tokens=True)
            for key in no_special_tokens.keys():
                for i_no, i_with in zip(no_special_tokens[key], with_special_tokens[key]):
                    self.assertEqual(len(i_no), len(i_with) - simple_num_special_tokens_to_add)

    def assert_prepare_for_model(self, tokenizer_r, tokenizer_p):
        string_sequence = "Asserting that both tokenizers are equal"
        python_output = tokenizer_p.prepare_for_model(tokenizer_p.encode(string_sequence))
        rust_output = tokenizer_r.prepare_for_model(tokenizer_r.encode(string_sequence))
        self.assertEqual(python_output, rust_output)


class WordPieceFastTokenizerTest(CommonFastTokenizerTest):
    """
    Override all the specific methods to test WordPiece behavior
    """

    TOKENIZERS_CLASSES = frozenset(
        [
            Tokenizer("Bert", BertTokenizerFast, BertTokenizer, "vocab_file", filter_non_english, None),
            Tokenizer(
                "DistilBert", DistilBertTokenizerFast, DistilBertTokenizer, "vocab_file", filter_non_english, None
            ),
        ]
    )

    def fast_only(self, tokenizer_r):
        super().fast_only(tokenizer_r)
        self.assert_offsets_with_special_characters(tokenizer_r)

    def assert_add_special_tokens(self, tokenizer_r):
        super().assert_add_special_tokens(tokenizer_r)

    def assert_offsets_with_special_characters(self, tokenizer_r):
        sentence = "A, naïve [MASK] AllenNLP sentence."
        tokens = tokenizer_r.encode_plus(
            sentence,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        do_lower_case = tokenizer_r.init_kwargs.get("do_lower_case")
        expected_results = (
            [
                ((0, 0), "[CLS]"),
                ((0, 1), "A"),
                ((1, 2), ","),
                ((3, 5), "na"),
                ((5, 6), "##ï"),
                ((6, 8), "##ve"),
                ((9, 15), "[MASK]"),
                ((16, 21), "Allen"),
                ((21, 23), "##NL"),
                ((23, 24), "##P"),
                ((25, 33), "sentence"),
                ((33, 34), "."),
                ((0, 0), "[SEP]"),
            ]
            if not do_lower_case
            else [
                ((0, 0), "[CLS]"),
                ((0, 1), "a"),
                ((1, 2), ","),
                ((3, 8), "naive"),
                ((9, 15), "[MASK]"),
                ((16, 21), "allen"),
                ((21, 23), "##nl"),
                ((23, 24), "##p"),
                ((25, 33), "sentence"),
                ((33, 34), "."),
                ((0, 0), "[SEP]"),
            ]
        )

        self.assertEqual([e[1] for e in expected_results], tokenizer_r.convert_ids_to_tokens(tokens["input_ids"]))
        self.assertEqual([e[0] for e in expected_results], tokens["offset_mapping"])


class RobertaFastTokenizerTest(CommonFastTokenizerTest):
    TOKENIZERS_CLASSES = frozenset(
        [
            Tokenizer(
                "Roberta",
                RobertaTokenizerFast,
                RobertaTokenizer,
                "vocab_file",
                filter_roberta_detectors,
                (("cls_token", "<s>"),),
            )
        ]
    )

    def assert_embeded_special_tokens(self, tokenizer_r, tokenizer_p):
        sentence = "A, <mask> AllenNLP sentence."
        tokens_r = tokenizer_r.encode_plus(sentence, add_special_tokens=True, return_token_type_ids=True)
        tokens_p = tokenizer_p.encode_plus(sentence, add_special_tokens=True, return_token_type_ids=True)

        # Rust correctly handles the space before the mask while python doesnt
        self.assertSequenceEqual(tokens_r["input_ids"], [0, 250, 6, 50264, 3823, 487, 21992, 3645, 4, 2])
        self.assertSequenceEqual(tokens_p["input_ids"], [0, 250, 6, 50264, 3823, 487, 21992, 3645, 4, 2])

        # token_type_ids should put 0 everywhere
        self.assertEquals(sum(tokens_r["token_type_ids"]), sum(tokens_p["token_type_ids"]))

        # attention_mask should put 1 everywhere, so sum over length should be 1
        self.assertEquals(
            sum(tokens_r["attention_mask"]) / len(tokens_r["attention_mask"]),
            sum(tokens_p["attention_mask"]) / len(tokens_p["attention_mask"]),
        )

        tokens_r = tokenizer_r.convert_ids_to_tokens(tokens_r["input_ids"])
        tokens_p = tokenizer_p.convert_ids_to_tokens(tokens_p["input_ids"])
        self.assertSequenceEqual(tokens_r, ["<s>", "A", ",", "<mask>", "ĠAllen", "N", "LP", "Ġsentence", ".", "</s>"])
        self.assertSequenceEqual(tokens_p, ["<s>", "A", ",", "<mask>", "ĠAllen", "N", "LP", "Ġsentence", ".", "</s>"])


class NoPaddingTokenFastTokenizerMatchingTest(CommonFastTokenizerTest):
    TOKENIZERS_CLASSES = [
        Tokenizer("OpenAI GPT", OpenAIGPTTokenizerFast, OpenAIGPTTokenizer, "vocab_file", None, None),
        Tokenizer("GPT2", GPT2TokenizerFast, GPT2Tokenizer, "vocab_file", None, [("add_prefix_space", True)]),
    ]

    def fast_align_python(self, tokenizer_r, tokenizer_p, tok_case, pretrained_name):
        # Check is_fast is set correctly
        self.assertFalse(tokenizer_p.is_fast)
        self.assertTrue(tokenizer_r.is_fast)

        # Check that Rust and Python align
        self.assert_tokenization_python_rust_equals(tokenizer_r, tokenizer_p)
        self.assert_num_special_tokens_to_add_equal(tokenizer_r, tokenizer_p)
        self.assert_max_length_equal(tokenizer_r, tokenizer_p)
        self.assert_special_tokens_map_equal(tokenizer_r, tokenizer_p)
        self.assert_embeded_special_tokens(tokenizer_r, tokenizer_p)
        self.assert_padding(tokenizer_r, tokenizer_p)

        # Specific for
        kwargs = {}
        if tok_case.kwargs is not None:
            kwargs = dict(tok_case.kwargs)
        tokenizer_r = tok_case.rust_cls.from_pretrained(pretrained_name, **kwargs)
        self.assert_pretokenized_inputs(tokenizer_r, tokenizer_p)

    def assert_padding(self, tokenizer_r, tokenizer_p, max_length=15):
        # Simple input
        s = "This is a simple input"
        s2 = ["This is a simple input 1", "This is a simple input 2"]
        p = ("This is a simple input", "This is a pair")
        p2 = [
            ("This is a simple input 1", "This is a simple input 2"),
            ("This is a simple pair 1", "This is a simple pair 2"),
        ]

        # Simple input tests
        self.assertRaises(ValueError, tokenizer_r.encode, s, max_length=max_length, padding="max_length")

        # Simple input
        self.assertRaises(ValueError, tokenizer_r.encode_plus, s, max_length=max_length, padding="max_length")

        # Simple input
        self.assertRaises(
            ValueError, tokenizer_r.batch_encode_plus, s2, max_length=max_length, padding="max_length",
        )

        # Pair input
        self.assertRaises(ValueError, tokenizer_r.encode, p, max_length=max_length, padding="max_length")

        # Pair input
        self.assertRaises(ValueError, tokenizer_r.encode_plus, p, max_length=max_length, padding="max_length")

        # Pair input
        self.assertRaises(
            ValueError, tokenizer_r.batch_encode_plus, p2, max_length=max_length, padding="max_length",
        )


class TransfoXLFastTokenizerTest(NoPaddingTokenFastTokenizerMatchingTest):
    TOKENIZERS_CLASSES = frozenset(
        [Tokenizer("TransfoXL", TransfoXLTokenizerFast, TransfoXLTokenizer, "pretrained_vocab_file", None, None)]
    )

    @require_torch
    def test_all_tokenizers(self):
        super().test_all_tokenizers()

    @require_torch
    def test_pretokenized_tokenizers(self):
        super().test_pretokenized_tokenizers()
