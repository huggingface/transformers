# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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


import inspect
import itertools
import json
import os
import pickle
import re
import shutil
import tempfile
import unittest
from collections import OrderedDict
from itertools import takewhile
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

from huggingface_hub import HfApi
from requests.exceptions import HTTPError
from transformers import (
    AlbertTokenizer,
    AlbertTokenizerFast,
    BertTokenizer,
    BertTokenizerFast,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    SpecialTokensMixin,
    Trainer,
    TrainingArguments,
    is_tf_available,
    is_torch_available,
)
from transformers.testing_utils import (
    ENDPOINT_STAGING,
    PASS,
    USER,
    get_tests_dir,
    is_pt_tf_cross_test,
    is_staging_test,
    require_tf,
    require_tokenizers,
    require_torch,
    slow,
)
from transformers.tokenization_utils import AddedToken, Trie


if is_torch_available():
    import torch.nn as nn


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, TFPreTrainedModel


NON_ENGLISH_TAGS = ["chinese", "dutch", "french", "finnish", "german", "multilingual"]

SMALL_TRAINING_CORPUS = [
    ["This is the first sentence.", "This is the second one."],
    ["This sentence (contains #) over symbols and numbers 12 3.", "But not this one."],
]


def filter_non_english(_, pretrained_name: str):
    """Filter all the model for non-english language"""
    return not any([lang in pretrained_name for lang in NON_ENGLISH_TAGS])


def filter_roberta_detectors(_, pretrained_name: str):
    return "detector" not in pretrained_name


def merge_model_tokenizer_mappings(
    model_mapping: Dict["PretrainedConfig", Union["PreTrainedModel", "TFPreTrainedModel"]],
    tokenizer_mapping: Dict["PretrainedConfig", Tuple["PreTrainedTokenizer", "PreTrainedTokenizerFast"]],
) -> Dict[
    Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
    Tuple["PretrainedConfig", Union["PreTrainedModel", "TFPreTrainedModel"]],
]:
    configurations = list(model_mapping.keys())
    model_tokenizer_mapping = OrderedDict([])

    for configuration in configurations:
        if configuration in model_mapping and configuration in tokenizer_mapping:
            model = model_mapping[configuration]
            tokenizer = tokenizer_mapping[configuration][0]
            tokenizer_fast = tokenizer_mapping[configuration][1]

            model_tokenizer_mapping.update({tokenizer: (configuration, model)})
            if tokenizer_fast is not None:
                model_tokenizer_mapping.update({tokenizer_fast: (configuration, model)})

    return model_tokenizer_mapping


class TokenizerTesterMixin:

    tokenizer_class = None
    rust_tokenizer_class = None
    test_slow_tokenizer = True
    test_rust_tokenizer = True
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    from_pretrained_filter = None
    from_pretrained_vocab_key = "vocab_file"
    test_seq2seq = True

    # set to True to test a sentencepiece tokenizer
    test_sentencepiece = False

    # set to True to ignore casing when testing a sentencepiece tokenizer
    # test_sentencepiece must also be set to True
    test_sentencepiece_ignore_case = False

    def setUp(self) -> None:
        # Tokenizer.filter makes it possible to filter which Tokenizer to case based on all the
        # information available in Tokenizer (name, rust class, python class, vocab key name)
        if self.test_rust_tokenizer:
            tokenizers_list = [
                (
                    self.rust_tokenizer_class,
                    pretrained_name,
                    self.from_pretrained_kwargs if self.from_pretrained_kwargs is not None else {},
                )
                for pretrained_name in self.rust_tokenizer_class.pretrained_vocab_files_map[
                    self.from_pretrained_vocab_key
                ].keys()
                if self.from_pretrained_filter is None
                or (self.from_pretrained_filter is not None and self.from_pretrained_filter(pretrained_name))
            ]
            self.tokenizers_list = tokenizers_list[:1]  # Let's just test the first pretrained vocab for speed
        else:
            self.tokenizers_list = []
        with open(f"{get_tests_dir()}/fixtures/sample_text.txt", encoding="utf-8") as f_data:
            self._data = f_data.read().replace("\n\n", "\n").strip()

        self.tmpdirname = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def get_input_output_texts(self, tokenizer):
        input_txt = self.get_clean_sequence(tokenizer)[0]
        return input_txt, input_txt

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5) -> Tuple[str, list]:
        toks = [(i, tokenizer.decode([i], clean_up_tokenization_spaces=False)) for i in range(len(tokenizer))]
        toks = list(filter(lambda t: re.match(r"^[ a-zA-Z]+$", t[1]), toks))
        toks = list(filter(lambda t: [t[0]] == tokenizer.encode(t[1], add_special_tokens=False), toks))
        if max_length is not None and len(toks) > max_length:
            toks = toks[:max_length]
        if min_length is not None and len(toks) < min_length and len(toks) > 0:
            while len(toks) < min_length:
                toks = toks + toks
        # toks_str = [t[1] for t in toks]
        toks_ids = [t[0] for t in toks]

        # Ensure consistency
        output_txt = tokenizer.decode(toks_ids, clean_up_tokenization_spaces=False)
        if " " not in output_txt and len(toks_ids) > 1:
            output_txt = (
                tokenizer.decode([toks_ids[0]], clean_up_tokenization_spaces=False)
                + " "
                + tokenizer.decode(toks_ids[1:], clean_up_tokenization_spaces=False)
            )
        if with_prefix_space:
            output_txt = " " + output_txt
        output_ids = tokenizer.encode(output_txt, add_special_tokens=False)
        return output_txt, output_ids

    def get_tokenizers(self, fast=True, **kwargs) -> List[PreTrainedTokenizerBase]:
        if fast and self.test_rust_tokenizer and self.test_slow_tokenizer:
            return [self.get_tokenizer(**kwargs), self.get_rust_tokenizer(**kwargs)]
        elif fast and self.test_rust_tokenizer:
            return [self.get_rust_tokenizer(**kwargs)]
        elif self.test_slow_tokenizer:
            return [self.get_tokenizer(**kwargs)]
        else:
            raise ValueError("This tokenizer class has no tokenizer to be tested.")

    def get_tokenizer(self, **kwargs) -> PreTrainedTokenizer:
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs) -> PreTrainedTokenizerFast:
        return self.rust_tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def tokenizer_integration_test_util(
        self,
        expected_encoding: Dict,
        model_name: str,
        revision: str = None,
        sequences: List[str] = None,
        decode_kwargs: Dict[str, Any] = None,
        padding: bool = True,
    ):
        """
        Util for integration test.

        Text is tokenized and then reverted back to text. Both results are then checked.

        Args:
            expected_encoding:
                The expected result of the tokenizer output.
            model_name:
                The model name of the tokenizer to load and use.
            revision:
                The full git revision number of the model. This is to pin the
                tokenizer config and to avoid that tests start to fail if the
                config gets changed upstream.
            sequences:
                Can overwrite the texts that are used to check the tokenizer.
                This is useful if the tokenizer supports non english languages
                like france.
            decode_kwargs:
                Additional args for the ``decode`` function which reverts the
                tokenized text back to a string.
            padding:
                Activates and controls padding of the tokenizer.
        """
        decode_kwargs = {} if decode_kwargs is None else decode_kwargs

        if sequences is None:
            sequences = [
                "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides "
                "general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet...) for Natural "
                "Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained "
                "models in 100+ languages and deep interoperability between Jax, PyTorch and TensorFlow.",
                "BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly "
                "conditioning on both left and right context in all layers.",
                "The quick brown fox jumps over the lazy dog.",
            ]

        if self.test_sentencepiece_ignore_case:
            sequences = [sequence.lower() for sequence in sequences]

        tokenizer_classes = [self.tokenizer_class]
        if self.test_rust_tokenizer:
            tokenizer_classes.append(self.rust_tokenizer_class)

        for tokenizer_class in tokenizer_classes:
            tokenizer = tokenizer_class.from_pretrained(
                model_name,
                revision=revision,  # to pin the tokenizer version
            )

            encoding = tokenizer(sequences, padding=padding)
            decoded_sequences = [
                tokenizer.decode(seq, skip_special_tokens=True, **decode_kwargs) for seq in encoding["input_ids"]
            ]

            encoding_data = encoding.data
            self.assertDictEqual(encoding_data, expected_encoding)

            for expected, decoded in zip(sequences, decoded_sequences):
                if self.test_sentencepiece_ignore_case:
                    expected = expected.lower()
                self.assertEqual(expected, decoded)

    def assert_padded_input_match(self, input_r: list, input_p: list, max_length: int, pad_token_id: int):
        # Ensure we match max_length
        self.assertEqual(len(input_r), max_length)
        self.assertEqual(len(input_p), max_length)

        # Ensure the number of padded tokens is the same
        padded_tokens_r = list(takewhile(lambda i: i == pad_token_id, reversed(input_r)))
        padded_tokens_p = list(takewhile(lambda i: i == pad_token_id, reversed(input_p)))
        self.assertSequenceEqual(padded_tokens_r, padded_tokens_p)

    def assert_batch_padded_input_match(
        self,
        input_r: dict,
        input_p: dict,
        max_length: int,
        pad_token_id: int,
        model_main_input_name: str = "input_ids",
    ):
        for i_r in input_r.values():
            self.assertEqual(len(i_r), 2), self.assertEqual(len(i_r[0]), max_length), self.assertEqual(
                len(i_r[1]), max_length
            )
            self.assertEqual(len(i_r), 2), self.assertEqual(len(i_r[0]), max_length), self.assertEqual(
                len(i_r[1]), max_length
            )

        for i_r, i_p in zip(input_r[model_main_input_name], input_p[model_main_input_name]):
            self.assert_padded_input_match(i_r, i_p, max_length, pad_token_id)

        for i_r, i_p in zip(input_r["attention_mask"], input_p["attention_mask"]):
            self.assertSequenceEqual(i_r, i_p)

    @staticmethod
    def convert_batch_encode_plus_format_to_encode_plus(batch_encode_plus_sequences):
        # Switch from batch_encode_plus format:   {'input_ids': [[...], [...]], ...}
        # to the list of examples/ encode_plus format: [{'input_ids': [...], ...}, {'input_ids': [...], ...}]
        return [
            {value: batch_encode_plus_sequences[value][i] for value in batch_encode_plus_sequences.keys()}
            for i in range(len(batch_encode_plus_sequences["input_ids"]))
        ]

    # TODO: this test can be combined with `test_sentencepiece_tokenize_and_convert_tokens_to_string` after the latter is extended to all tokenizers.
    def test_tokenize_special_tokens(self):
        """Test `tokenize` with special tokens."""
        tokenizers = self.get_tokenizers(fast=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                SPECIAL_TOKEN_1 = "[SPECIAL_TOKEN_1]"
                SPECIAL_TOKEN_2 = "[SPECIAL_TOKEN_2]"

                # TODO:
                # Can we combine `unique_no_split_tokens` and `all_special_tokens`(and properties related to it)
                # with one variable(property) for a better maintainability?

                # `add_tokens` method stores special tokens only in `tokenizer.unique_no_split_tokens`. (in tokenization_utils.py)
                tokenizer.add_tokens([SPECIAL_TOKEN_1], special_tokens=True)
                # `add_special_tokens` method stores special tokens in `tokenizer.additional_special_tokens`,
                # which also occur in `tokenizer.all_special_tokens`. (in tokenization_utils_base.py)
                tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKEN_2]})

                token_1 = tokenizer.tokenize(SPECIAL_TOKEN_1)
                token_2 = tokenizer.tokenize(SPECIAL_TOKEN_2)

                self.assertEqual(len(token_1), 1)
                self.assertEqual(len(token_2), 1)
                self.assertEqual(token_1[0], SPECIAL_TOKEN_1)
                self.assertEqual(token_2[0], SPECIAL_TOKEN_2)

    # TODO: this test could be extended to all tokenizers - not just the sentencepiece
    def test_sentencepiece_tokenize_and_convert_tokens_to_string(self):
        """Test ``_tokenize`` and ``convert_tokens_to_string``."""
        if not self.test_sentencepiece:
            return

        tokenizer = self.get_tokenizer()
        text = "This is text to test the tokenizer."

        if self.test_sentencepiece_ignore_case:
            text = text.lower()

        tokens = tokenizer.tokenize(text)

        self.assertTrue(len(tokens) > 0)

        # check if converting back to original text works
        reverse_text = tokenizer.convert_tokens_to_string(tokens)

        if self.test_sentencepiece_ignore_case:
            reverse_text = reverse_text.lower()

        self.assertEqual(reverse_text, text)

    def test_subword_regularization_tokenizer(self) -> None:
        if not self.test_sentencepiece:
            return

        # Subword regularization is only available for the slow tokenizer.
        sp_model_kwargs = {"enable_sampling": True, "alpha": 0.1, "nbest_size": -1}
        tokenizer = self.get_tokenizer(sp_model_kwargs=sp_model_kwargs)

        self.assertTrue(hasattr(tokenizer, "sp_model_kwargs"))
        self.assertIsNotNone(tokenizer.sp_model_kwargs)
        self.assertTrue(isinstance(tokenizer.sp_model_kwargs, dict))
        self.assertEqual(tokenizer.sp_model_kwargs, sp_model_kwargs)
        self.check_subword_sampling(tokenizer)

    def test_pickle_subword_regularization_tokenizer(self) -> None:
        if not self.test_sentencepiece:
            return

        """Google pickle __getstate__ __setstate__ if you are struggling with this."""
        # Subword regularization is only available for the slow tokenizer.
        sp_model_kwargs = {"enable_sampling": True, "alpha": 0.1, "nbest_size": -1}
        tokenizer = self.get_tokenizer(sp_model_kwargs=sp_model_kwargs)
        tokenizer_bin = pickle.dumps(tokenizer)
        del tokenizer
        tokenizer_new = pickle.loads(tokenizer_bin)

        self.assertTrue(hasattr(tokenizer_new, "sp_model_kwargs"))
        self.assertIsNotNone(tokenizer_new.sp_model_kwargs)
        self.assertTrue(isinstance(tokenizer_new.sp_model_kwargs, dict))
        self.assertEqual(tokenizer_new.sp_model_kwargs, sp_model_kwargs)
        self.check_subword_sampling(tokenizer_new)

    def test_model_input_names_signature(self):
        accepted_model_main_input_names = [
            "input_ids",  # nlp models
            "input_values",  # speech models
        ]

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            # first name of model_input_names has to correspond to main model input name
            # to make sure `tokenizer.pad(...)` works correctly
            self.assertTrue(tokenizer.model_input_names[0] in accepted_model_main_input_names)

    def test_rust_tokenizer_signature(self):
        if not self.test_rust_tokenizer:
            return

        signature = inspect.signature(self.rust_tokenizer_class.__init__)

        self.assertIn("tokenizer_file", signature.parameters)
        self.assertIsNone(signature.parameters["tokenizer_file"].default)

    def test_tokenizer_slow_store_full_signature(self):
        if not self.test_slow_tokenizer:
            return

        signature = inspect.signature(self.tokenizer_class.__init__)
        tokenizer = self.get_tokenizer()

        for parameter_name, parameter in signature.parameters.items():
            if parameter.default != inspect.Parameter.empty:
                self.assertIn(parameter_name, tokenizer.init_kwargs)

    def test_tokenizer_fast_store_full_signature(self):
        if not self.test_rust_tokenizer:
            return

        signature = inspect.signature(self.rust_tokenizer_class.__init__)
        tokenizer = self.get_rust_tokenizer()

        for parameter_name, parameter in signature.parameters.items():
            if parameter.default != inspect.Parameter.empty and parameter_name not in [
                "vocab_file",
                "merges_file",
                "tokenizer_file",
            ]:
                self.assertIn(parameter_name, tokenizer.init_kwargs)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence, _ = self.get_input_output_texts(tokenizer)

        # We don't have an exact equivalence on `tokenize()` between Rust and Slow
        # Slow tokenizer only split tokens, Rust tokenizers will replace with <unk>
        # tokens = tokenizer.tokenize(sequence)
        # rust_tokens = rust_tokenizer.tokenize(sequence)
        # self.assertListEqual(tokens, rust_tokens)

        ids = tokenizer.encode(sequence, add_special_tokens=False)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=False)
        self.assertListEqual(ids, rust_ids)

        ids = tokenizer.encode(sequence, add_special_tokens=True)
        rust_ids = rust_tokenizer.encode(sequence, add_special_tokens=True)
        self.assertListEqual(ids, rust_ids)

    def test_tokenizers_common_properties(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                attributes_list = [
                    "bos_token",
                    "eos_token",
                    "unk_token",
                    "sep_token",
                    "pad_token",
                    "cls_token",
                    "mask_token",
                ]
                for attr in attributes_list:
                    self.assertTrue(hasattr(tokenizer, attr))
                    self.assertTrue(hasattr(tokenizer, attr + "_id"))

                self.assertTrue(hasattr(tokenizer, "additional_special_tokens"))
                self.assertTrue(hasattr(tokenizer, "additional_special_tokens_ids"))

                attributes_list = [
                    "model_max_length",
                    "init_inputs",
                    "init_kwargs",
                ]
                if not isinstance(tokenizer, PreTrainedTokenizerFast):
                    attributes_list += [
                        "added_tokens_encoder",
                        "added_tokens_decoder",
                    ]
                for attr in attributes_list:
                    self.assertTrue(hasattr(tokenizer, attr))

    def test_save_and_load_tokenizer(self):
        # safety check on max_len default value so we are sure the test works
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                self.assertNotEqual(tokenizer.model_max_length, 42)

        # Now let's start the test
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertDictEqual(before_vocab, after_vocab)

                shutil.rmtree(tmpdirname)

        tokenizers = self.get_tokenizers(model_max_length=42)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                tokenizer.add_tokens(["bim", "bambam"])
                additional_special_tokens = tokenizer.additional_special_tokens
                additional_special_tokens.append("new_additional_special_token")
                tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertDictEqual(before_vocab, after_vocab)
                self.assertIn("bim", after_vocab)
                self.assertIn("bambam", after_vocab)
                self.assertIn("new_additional_special_token", after_tokenizer.additional_special_tokens)
                self.assertEqual(after_tokenizer.model_max_length, 42)

                tokenizer = tokenizer.__class__.from_pretrained(tmpdirname, model_max_length=43)
                self.assertEqual(tokenizer.model_max_length, 43)

                shutil.rmtree(tmpdirname)

        # Test that we can also use the non-legacy saving format for fast tokenizers
        tokenizers = self.get_tokenizers(model_max_length=42)
        for tokenizer in tokenizers:
            if not tokenizer.is_fast:
                continue
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Isolate this from the other tests because we save additional tokens/etc
                tmpdirname = tempfile.mkdtemp()

                sample_text = " He is very happy, UNwant\u00E9d,running"
                tokenizer.add_tokens(["bim", "bambam"])
                additional_special_tokens = tokenizer.additional_special_tokens
                additional_special_tokens.append("new_additional_special_token")
                tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
                before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
                before_vocab = tokenizer.get_vocab()
                tokenizer.save_pretrained(tmpdirname)

                after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
                after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
                after_vocab = after_tokenizer.get_vocab()
                self.assertListEqual(before_tokens, after_tokens)
                self.assertDictEqual(before_vocab, after_vocab)
                self.assertIn("bim", after_vocab)
                self.assertIn("bambam", after_vocab)
                self.assertIn("new_additional_special_token", after_tokenizer.additional_special_tokens)
                self.assertEqual(after_tokenizer.model_max_length, 42)

                tokenizer = tokenizer.__class__.from_pretrained(tmpdirname, model_max_length=43)
                self.assertEqual(tokenizer.model_max_length, 43)

                shutil.rmtree(tmpdirname)

    def test_pickle_tokenizer(self):
        """Google pickle __getstate__ __setstate__ if you are struggling with this."""
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                self.assertIsNotNone(tokenizer)

                text = "Munich and Berlin are nice cities"
                subwords = tokenizer.tokenize(text)

                filename = os.path.join(self.tmpdirname, "tokenizer.bin")
                with open(filename, "wb") as handle:
                    pickle.dump(tokenizer, handle)

                with open(filename, "rb") as handle:
                    tokenizer_new = pickle.load(handle)

                subwords_loaded = tokenizer_new.tokenize(text)

                self.assertListEqual(subwords, subwords_loaded)

    @require_tokenizers
    def test_pickle_added_tokens(self):
        tok1 = AddedToken("<s>", rstrip=True, lstrip=True, normalized=False, single_word=True)
        tok2 = pickle.loads(pickle.dumps(tok1))

        self.assertEqual(tok1.__getstate__(), tok2.__getstate__())

    def test_added_tokens_do_lower_case(self):
        # TODO(thom) activate fast tokenizer tests once Rust tokenizers accepts white spaces in added tokens.
        tokenizers = [self.get_tokenizer(do_lower_case=True)] if self.test_slow_tokenizer else []
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if not hasattr(tokenizer, "do_lower_case") or not tokenizer.do_lower_case:
                    continue

                special_token = tokenizer.all_special_tokens[0]

                text = special_token + " aaaaa bbbbbb low cccccccccdddddddd l " + special_token
                text2 = special_token + " AAAAA BBBBBB low CCCCCCCCCDDDDDDDD l " + special_token

                toks0 = tokenizer.tokenize(text)  # toks before adding new_toks

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd", "AAAAA BBBBBB", "CCCCCCCCCDDDDDDDD"]
                added = tokenizer.add_tokens(new_toks)
                self.assertEqual(added, 2)

                toks = tokenizer.tokenize(text)
                toks2 = tokenizer.tokenize(text2)

                self.assertEqual(len(toks), len(toks2))
                self.assertListEqual(toks, toks2)
                if not isinstance(tokenizer, PreTrainedTokenizerFast):
                    # Python tokenizers can have added tokens with spaces inside them
                    # cf https://github.com/huggingface/tokenizers/issues/302
                    self.assertNotEqual(len(toks), len(toks0))  # toks0 should be longer

                # Check that none of the special tokens are lowercased
                sequence_with_special_tokens = "A " + " yEs ".join(tokenizer.all_special_tokens) + " B"
                tokenized_sequence = tokenizer.tokenize(sequence_with_special_tokens)

                for special_token in tokenizer.all_special_tokens:
                    self.assertTrue(special_token in tokenized_sequence)

        tokenizers = [self.get_tokenizer(do_lower_case=True)] if self.test_slow_tokenizer else []
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if hasattr(tokenizer, "do_lower_case") and tokenizer.do_lower_case:
                    continue

                special_token = tokenizer.all_special_tokens[0]

                text = special_token + " aaaaa bbbbbb low cccccccccdddddddd l " + special_token
                text2 = special_token + " AAAAA BBBBBB low CCCCCCCCCDDDDDDDD l " + special_token

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd", "AAAAA BBBBBB", "CCCCCCCCCDDDDDDDD"]

                toks0 = tokenizer.tokenize(text)  # toks before adding new_toks

                added = tokenizer.add_tokens(new_toks)
                self.assertIn(added, [2, 4])

                toks = tokenizer.tokenize(text)
                toks2 = tokenizer.tokenize(text2)

                self.assertEqual(len(toks), len(toks2))  # Length should still be the same
                self.assertNotEqual(toks[1], toks2[1])  # But at least the first non-special tokens should differ
                if not isinstance(tokenizer, PreTrainedTokenizerFast):
                    # Python tokenizers can have added tokens with spaces inside them
                    # cf https://github.com/huggingface/tokenizers/issues/302
                    self.assertNotEqual(len(toks), len(toks0))  # toks0 should be longer

    def test_add_tokens_tokenizer(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode("aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)

                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l", add_special_tokens=False
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    def test_add_special_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, ids = self.get_clean_sequence(tokenizer)

                special_token = "[SPECIAL_TOKEN]"

                tokenizer.add_special_tokens({"cls_token": special_token})
                encoded_special_token = tokenizer.encode(special_token, add_special_tokens=False)
                self.assertEqual(len(encoded_special_token), 1)

                text = tokenizer.decode(ids + encoded_special_token, clean_up_tokenization_spaces=False)
                encoded = tokenizer.encode(text, add_special_tokens=False)

                input_encoded = tokenizer.encode(input_text, add_special_tokens=False)
                special_token_id = tokenizer.encode(special_token, add_special_tokens=False)
                self.assertEqual(encoded, input_encoded + special_token_id)

                decoded = tokenizer.decode(encoded, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

    def test_internal_consistency(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_text, output_text = self.get_input_output_texts(tokenizer)

                tokens = tokenizer.tokenize(input_text)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_2 = tokenizer.encode(input_text, add_special_tokens=False)
                self.assertListEqual(ids, ids_2)

                tokens_2 = tokenizer.convert_ids_to_tokens(ids)
                self.assertNotEqual(len(tokens_2), 0)
                text_2 = tokenizer.decode(ids)
                self.assertIsInstance(text_2, str)

                self.assertEqual(text_2, output_text)

    @require_tokenizers
    def test_encode_decode_with_spaces(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                # new_toks = ["[ABC]", "[DEF]"]  # TODO(thom) add this one back when Rust toks are ready: , "GHI IHG"]
                new_toks = [AddedToken("[ABC]", normalized=False), AddedToken("[DEF]", normalized=False)]
                tokenizer.add_tokens(new_toks)
                input = "[ABC][DEF][ABC][DEF]"  # TODO(thom) add back cf above: "[ABC] [DEF] [ABC] GHI IHG [DEF]"
                if self.space_between_special_tokens:
                    output = "[ABC] [DEF] [ABC] [DEF]"
                else:
                    output = input
                encoded = tokenizer.encode(input, add_special_tokens=False)
                decoded = tokenizer.decode(encoded, spaces_between_special_tokens=self.space_between_special_tokens)
                self.assertIn(decoded, [output, output.lower()])

    def test_pretrained_model_lists(self):
        # We should have at least one default checkpoint for each tokenizer
        # We should specify the max input length as well (used in some part to list the pretrained checkpoints)
        self.assertGreaterEqual(len(self.tokenizer_class.pretrained_vocab_files_map), 1)
        self.assertGreaterEqual(len(list(self.tokenizer_class.pretrained_vocab_files_map.values())[0]), 1)
        self.assertEqual(
            len(list(self.tokenizer_class.pretrained_vocab_files_map.values())[0]),
            len(self.tokenizer_class.max_model_input_sizes),
        )

        weights_list = list(self.tokenizer_class.max_model_input_sizes.keys())
        weights_lists_2 = []
        for file_id, map_list in self.tokenizer_class.pretrained_vocab_files_map.items():
            weights_lists_2.append(list(map_list.keys()))

        for weights_list_2 in weights_lists_2:
            self.assertListEqual(weights_list, weights_list_2)

    def test_mask_output(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                if (
                    tokenizer.build_inputs_with_special_tokens.__qualname__.split(".")[0] != "PreTrainedTokenizer"
                    and "token_type_ids" in tokenizer.model_input_names
                ):
                    seq_0 = "Test this method."
                    seq_1 = "With these inputs."
                    information = tokenizer.encode_plus(seq_0, seq_1, add_special_tokens=True)
                    sequences, mask = information["input_ids"], information["token_type_ids"]
                    self.assertEqual(len(sequences), len(mask))

    def test_token_type_ids(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                seq_0 = "Test this method."

                # We want to have sequence 0 and sequence 1 are tagged
                # respectively with 0 and 1 token_ids
                # (regardless of whether the model use token type ids)
                # We use this assumption in the QA pipeline among other place
                output = tokenizer(seq_0, return_token_type_ids=True)
                self.assertIn(0, output["token_type_ids"])

    def test_sequence_ids(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            if not tokenizer.is_fast:
                continue
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                seq_0 = "Test this method."
                seq_1 = "With these inputs."

                # We want to have sequence 0 and sequence 1 are tagged
                # respectively with 0 and 1 token_ids
                # (regardless of whether the model use token type ids)
                # We use this assumption in the QA pipeline among other place
                output = tokenizer(seq_0)
                self.assertIn(0, output.sequence_ids())

                output = tokenizer(seq_0, seq_1)
                self.assertIn(0, output.sequence_ids())
                self.assertIn(1, output.sequence_ids())

                if tokenizer.num_special_tokens_to_add(pair=True):
                    self.assertIn(None, output.sequence_ids())

    def test_number_of_added_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                seq_0 = "Test this method."
                seq_1 = "With these inputs."

                sequences = tokenizer.encode(seq_0, seq_1, add_special_tokens=False)
                attached_sequences = tokenizer.encode(seq_0, seq_1, add_special_tokens=True)

                # Method is implemented (e.g. not GPT-2)
                if len(attached_sequences) != 2:
                    self.assertEqual(
                        tokenizer.num_special_tokens_to_add(pair=True), len(attached_sequences) - len(sequences)
                    )

    def test_maximum_encoding_length_single_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False, model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                seq_0, ids = self.get_clean_sequence(tokenizer, max_length=20)

                sequence = tokenizer.encode(seq_0, add_special_tokens=False)
                total_length = len(sequence)

                assert total_length > 4, "Issue with the testing sequence, please update it it's too short"

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_1 = seq_0 * model_max_length

                sequence1 = tokenizer(seq_1, add_special_tokens=False)
                total_length1 = len(sequence1["input_ids"])
                assert (
                    total_length1 > model_max_length
                ), "Issue with the testing sequence, please update it it's too short"

                # Simple
                padding_strategies = (
                    [False, True, "longest"] if tokenizer.pad_token and tokenizer.pad_token_id >= 0 else [False]
                )
                for padding_state in padding_strategies:
                    with self.subTest(f"Padding: {padding_state}"):
                        for truncation_state in [True, "longest_first", "only_first"]:
                            with self.subTest(f"Truncation: {truncation_state}"):
                                output = tokenizer(seq_1, padding=padding_state, truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"]), model_max_length)

                                output = tokenizer([seq_1], padding=padding_state, truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"][0]), model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer(seq_1, padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer([seq_1], padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                # Overflowing tokens
                stride = 2
                information = tokenizer(
                    seq_0,
                    max_length=total_length - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="longest_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )

                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information["input_ids"][0]
                    overflowing_tokens = information["input_ids"][1]
                    self.assertEqual(len(information["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), total_length - 2)
                    self.assertEqual(truncated_sequence, sequence[:-2])

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, sequence[-(2 + stride) :])
                else:
                    truncated_sequence = information["input_ids"]
                    overflowing_tokens = information["overflowing_tokens"]

                    self.assertEqual(len(truncated_sequence), total_length - 2)
                    self.assertEqual(truncated_sequence, sequence[:-2])

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, sequence[-(2 + stride) :])

    def test_maximum_encoding_length_pair_input(self):
        tokenizers = self.get_tokenizers(do_lower_case=False, model_max_length=100)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Build a sequence from our model's vocabulary
                stride = 2
                seq_0, ids = self.get_clean_sequence(tokenizer, max_length=20)
                if len(ids) <= 2 + stride:
                    seq_0 = (seq_0 + " ") * (2 + stride)
                    ids = None

                seq0_tokens = tokenizer.encode(seq_0, add_special_tokens=False)
                assert len(seq0_tokens) > 2 + stride

                seq_1 = "This is another sentence to be encoded."
                seq1_tokens = tokenizer.encode(seq_1, add_special_tokens=False)
                if abs(len(seq0_tokens) - len(seq1_tokens)) <= 2:
                    seq1_tokens = seq1_tokens + seq1_tokens
                    seq_1 = tokenizer.decode(seq1_tokens, clean_up_tokenization_spaces=False)
                seq1_tokens = tokenizer.encode(seq_1, add_special_tokens=False)

                assert len(seq1_tokens) > 2 + stride

                smallest = seq1_tokens if len(seq0_tokens) > len(seq1_tokens) else seq0_tokens

                # We are not using the special tokens - a bit too hard to test all the tokenizers with this
                # TODO try this again later
                sequence = tokenizer.encode(seq_0, seq_1, add_special_tokens=False)  # , add_prefix_space=False)

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_2 = seq_0 * model_max_length
                assert len(seq_2) > model_max_length

                sequence1 = tokenizer(seq_1, add_special_tokens=False)
                total_length1 = len(sequence1["input_ids"])
                sequence2 = tokenizer(seq_2, seq_1, add_special_tokens=False)
                total_length2 = len(sequence2["input_ids"])
                assert total_length1 < model_max_length - 10, "Issue with the testing sequence, please update it."
                assert total_length2 > model_max_length, "Issue with the testing sequence, please update it."

                # Simple
                padding_strategies = (
                    [False, True, "longest"] if tokenizer.pad_token and tokenizer.pad_token_id >= 0 else [False]
                )
                for padding_state in padding_strategies:
                    with self.subTest(f"{tokenizer.__class__.__name__} Padding: {padding_state}"):
                        for truncation_state in [True, "longest_first", "only_first"]:
                            with self.subTest(f"{tokenizer.__class__.__name__} Truncation: {truncation_state}"):
                                output = tokenizer(seq_2, seq_1, padding=padding_state, truncation=truncation_state)
                                self.assertEqual(len(output["input_ids"]), model_max_length)

                                output = tokenizer(
                                    [seq_2], [seq_1], padding=padding_state, truncation=truncation_state
                                )
                                self.assertEqual(len(output["input_ids"][0]), model_max_length)

                        # Simple
                        output = tokenizer(seq_1, seq_2, padding=padding_state, truncation="only_second")
                        self.assertEqual(len(output["input_ids"]), model_max_length)

                        output = tokenizer([seq_1], [seq_2], padding=padding_state, truncation="only_second")
                        self.assertEqual(len(output["input_ids"][0]), model_max_length)

                        # Simple with no truncation
                        # Reset warnings
                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer(seq_1, seq_2, padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer([seq_1], [seq_2], padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length for this model"
                            )
                        )

                truncated_first_sequence = tokenizer.encode(seq_0, add_special_tokens=False)[:-2] + tokenizer.encode(
                    seq_1, add_special_tokens=False
                )
                truncated_second_sequence = (
                    tokenizer.encode(seq_0, add_special_tokens=False)
                    + tokenizer.encode(seq_1, add_special_tokens=False)[:-2]
                )
                truncated_longest_sequence = (
                    truncated_first_sequence if len(seq0_tokens) > len(seq1_tokens) else truncated_second_sequence
                )

                overflow_first_sequence = tokenizer.encode(seq_0, add_special_tokens=False)[
                    -(2 + stride) :
                ] + tokenizer.encode(seq_1, add_special_tokens=False)
                overflow_second_sequence = (
                    tokenizer.encode(seq_0, add_special_tokens=False)
                    + tokenizer.encode(seq_1, add_special_tokens=False)[-(2 + stride) :]
                )
                overflow_longest_sequence = (
                    overflow_first_sequence if len(seq0_tokens) > len(seq1_tokens) else overflow_second_sequence
                )

                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    information = tokenizer(
                        seq_0,
                        seq_1,
                        max_length=len(sequence) - 2,
                        add_special_tokens=False,
                        stride=stride,
                        truncation="longest_first",
                        return_overflowing_tokens=True,
                        # add_prefix_space=False,
                    )
                    truncated_sequence = information["input_ids"][0]
                    overflowing_tokens = information["input_ids"][1]
                    self.assertEqual(len(information["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_longest_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(smallest))
                    self.assertEqual(overflowing_tokens, overflow_longest_sequence)
                else:
                    # No overflowing tokens when using 'longest' in python tokenizers
                    with self.assertRaises(ValueError) as context:
                        information = tokenizer(
                            seq_0,
                            seq_1,
                            max_length=len(sequence) - 2,
                            add_special_tokens=False,
                            stride=stride,
                            truncation="longest_first",
                            return_overflowing_tokens=True,
                            # add_prefix_space=False,
                        )

                    self.assertTrue(
                        context.exception.args[0].startswith(
                            "Not possible to return overflowing tokens for pair of sequences with the "
                            "`longest_first`. Please select another truncation strategy than `longest_first`, "
                            "for instance `only_second` or `only_first`."
                        )
                    )

                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    information = tokenizer(
                        seq_0,
                        seq_1,
                        max_length=len(sequence) - 2,
                        add_special_tokens=False,
                        stride=stride,
                        truncation=True,
                        return_overflowing_tokens=True,
                        # add_prefix_space=False,
                    )
                    truncated_sequence = information["input_ids"][0]
                    overflowing_tokens = information["input_ids"][1]
                    self.assertEqual(len(information["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_longest_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(smallest))
                    self.assertEqual(overflowing_tokens, overflow_longest_sequence)
                else:
                    # No overflowing tokens when using 'longest' in python tokenizers
                    with self.assertRaises(ValueError) as context:
                        information = tokenizer(
                            seq_0,
                            seq_1,
                            max_length=len(sequence) - 2,
                            add_special_tokens=False,
                            stride=stride,
                            truncation=True,
                            return_overflowing_tokens=True,
                            # add_prefix_space=False,
                        )

                    self.assertTrue(
                        context.exception.args[0].startswith(
                            "Not possible to return overflowing tokens for pair of sequences with the "
                            "`longest_first`. Please select another truncation strategy than `longest_first`, "
                            "for instance `only_second` or `only_first`."
                        )
                    )

                information_first_truncated = tokenizer(
                    seq_0,
                    seq_1,
                    max_length=len(sequence) - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="only_first",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information_first_truncated["input_ids"][0]
                    overflowing_tokens = information_first_truncated["input_ids"][1]
                    self.assertEqual(len(information_first_truncated["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_first_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(seq1_tokens))
                    self.assertEqual(overflowing_tokens, overflow_first_sequence)
                else:
                    truncated_sequence = information_first_truncated["input_ids"]
                    overflowing_tokens = information_first_truncated["overflowing_tokens"]

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_first_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, seq0_tokens[-(2 + stride) :])

                information_second_truncated = tokenizer(
                    seq_0,
                    seq_1,
                    max_length=len(sequence) - 2,
                    add_special_tokens=False,
                    stride=stride,
                    truncation="only_second",
                    return_overflowing_tokens=True,
                    # add_prefix_space=False,
                )
                # Overflowing tokens are handled quite differently in slow and fast tokenizers
                if isinstance(tokenizer, PreTrainedTokenizerFast):
                    truncated_sequence = information_second_truncated["input_ids"][0]
                    overflowing_tokens = information_second_truncated["input_ids"][1]
                    self.assertEqual(len(information_second_truncated["input_ids"]), 2)

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_second_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride + len(seq0_tokens))
                    self.assertEqual(overflowing_tokens, overflow_second_sequence)
                else:
                    truncated_sequence = information_second_truncated["input_ids"]
                    overflowing_tokens = information_second_truncated["overflowing_tokens"]

                    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
                    self.assertEqual(truncated_sequence, truncated_second_sequence)

                    self.assertEqual(len(overflowing_tokens), 2 + stride)
                    self.assertEqual(overflowing_tokens, seq1_tokens[-(2 + stride) :])

    # def test_encode_input_type(self):
    #     tokenizers = self.get_tokenizers(do_lower_case=False)
    #     for tokenizer in tokenizers:
    #         with self.subTest(f"{tokenizer.__class__.__name__}"):
    #             sequence = "Let's encode this sequence"

    #             tokens = sequence.split()  # tokenizer.tokenize(sequence)
    #             # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    #             formatted_input = tokenizer.encode(sequence, add_special_tokens=True, add_prefix_space=False)

    #             self.assertEqual(
    #                 tokenizer.encode(tokens, is_split_into_words=True, add_special_tokens=True), formatted_input
    #             )
    #             # This is not supported with the Rust tokenizers
    #             # self.assertEqual(tokenizer.encode(input_ids, add_special_tokens=True), formatted_input)

    # def test_swap_special_token(self):
    #     tokenizers = self.get_tokenizers(do_lower_case=False)
    #     for tokenizer in tokenizers:
    #         with self.subTest(f"{tokenizer.__class__.__name__}"):
    #             # Our mask token
    #             mask = "<mask>"
    #             # We take a single word in the middle of the vocabulary
    #             all_tokens = sorted(tokenizer.get_vocab().keys())
    #             word = tokenizer.decode(tokenizer.encode(all_tokens[len(all_tokens)//2], add_special_tokens=False)[:1])

    #             sequence_0 = "Encode " + word + " sequence"
    #             sequence_masked_0 = "Encode " + mask + " sequence"

    #             sequence_1 = word + " this sequence"
    #             sequence_masked_1 = mask + " this sequence"

    #             # Add tokens so that masked token isn't split
    #             # tokens = [AddedToken(t, lstrip=True, normalized=False) for t in sequence.split()]
    #             # tokenizer.add_tokens(tokens)
    #             tokenizer.add_special_tokens(
    #                 {"mask_token": AddedToken(mask, normalized=False)}
    #             )  # Eat left space on Byte-level BPE tokenizers
    #             mask_ind = tokenizer.convert_tokens_to_ids(mask)

    #             # Test first masked sequence
    #             encoded_0 = tokenizer.encode(sequence_0, add_special_tokens=False)
    #             encoded_masked = tokenizer.encode(sequence_masked_0, add_special_tokens=False)
    #             assert len(encoded_masked) == len(encoded_0)
    #             mask_loc = encoded_masked.index(mask_ind)
    #             encoded_masked[mask_loc] = encoded_0[mask_loc]

    #             self.assertEqual(encoded_masked, encoded_0)

    #             # Test second masked sequence
    #             encoded_1 = tokenizer.encode(sequence_1, add_special_tokens=False)
    #             encoded_masked = tokenizer.encode(sequence_masked_1, add_special_tokens=False)
    #             assert len(encoded_masked) == len(encoded_1)
    #             mask_loc = encoded_masked.index(mask_ind)
    #             encoded_masked[mask_loc] = encoded_1[mask_loc]

    #             self.assertEqual(encoded_masked, encoded_1)

    def test_special_tokens_mask(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = "Encode this."
                # Testing single inputs
                encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False)
                encoded_sequence_dict = tokenizer.encode_plus(
                    sequence_0, add_special_tokens=True, return_special_tokens_mask=True  # , add_prefix_space=False
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

                filtered_sequence = [x for i, x in enumerate(encoded_sequence_w_special) if not special_tokens_mask[i]]
                self.assertEqual(encoded_sequence, filtered_sequence)

    def test_special_tokens_mask_input_pairs(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = "Encode this."
                sequence_1 = "This one too please."
                encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False)
                encoded_sequence += tokenizer.encode(sequence_1, add_special_tokens=False)
                encoded_sequence_dict = tokenizer.encode_plus(
                    sequence_0,
                    sequence_1,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                    # add_prefix_space=False,
                )
                encoded_sequence_w_special = encoded_sequence_dict["input_ids"]
                special_tokens_mask = encoded_sequence_dict["special_tokens_mask"]
                self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))

                filtered_sequence = [
                    (x if not special_tokens_mask[i] else None) for i, x in enumerate(encoded_sequence_w_special)
                ]
                filtered_sequence = [x for x in filtered_sequence if x is not None]
                self.assertEqual(encoded_sequence, filtered_sequence)

    def test_right_and_left_padding(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id

                # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence = tokenizer.encode(sequence)
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    sequence, max_length=sequence_length + padding_size, padding="max_length"
                )
                padded_sequence_length = len(padded_sequence)
                assert sequence_length + padding_size == padded_sequence_length
                assert encoded_sequence + [padding_idx] * padding_size == padded_sequence

                # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "left"
                encoded_sequence = tokenizer.encode(sequence)
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    sequence, max_length=sequence_length + padding_size, padding="max_length"
                )
                padded_sequence_length = len(padded_sequence)
                assert sequence_length + padding_size == padded_sequence_length
                assert [padding_idx] * padding_size + encoded_sequence == padded_sequence

                # RIGHT & LEFT PADDING - Check that nothing is done for 'longest' and 'no_padding'
                encoded_sequence = tokenizer.encode(sequence)
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(sequence, padding=True)
                padded_sequence_right_length = len(padded_sequence_right)
                assert sequence_length == padded_sequence_right_length
                assert encoded_sequence == padded_sequence_right

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(sequence, padding="longest")
                padded_sequence_left_length = len(padded_sequence_left)
                assert sequence_length == padded_sequence_left_length
                assert encoded_sequence == padded_sequence_left

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(sequence)
                padded_sequence_right_length = len(padded_sequence_right)
                assert sequence_length == padded_sequence_right_length
                assert encoded_sequence == padded_sequence_right

                tokenizer.padding_side = "left"
                padded_sequence_left = tokenizer.encode(sequence, padding=False)
                padded_sequence_left_length = len(padded_sequence_left)
                assert sequence_length == padded_sequence_left_length
                assert encoded_sequence == padded_sequence_left

    def test_padding_to_max_length(self):
        """We keep this test for backward compatibility but it should be remove when `pad_to_max_length` will e deprecated"""
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"
                padding_size = 10

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_idx = tokenizer.pad_token_id

                # Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "right"
                encoded_sequence = tokenizer.encode(sequence)
                sequence_length = len(encoded_sequence)
                # FIXME: the next line should be padding(max_length) to avoid warning
                padded_sequence = tokenizer.encode(
                    sequence, max_length=sequence_length + padding_size, pad_to_max_length=True
                )
                padded_sequence_length = len(padded_sequence)
                assert sequence_length + padding_size == padded_sequence_length
                assert encoded_sequence + [padding_idx] * padding_size == padded_sequence

                # Check that nothing is done when a maximum length is not specified
                encoded_sequence = tokenizer.encode(sequence)
                sequence_length = len(encoded_sequence)

                tokenizer.padding_side = "right"
                padded_sequence_right = tokenizer.encode(sequence, pad_to_max_length=True)
                padded_sequence_right_length = len(padded_sequence_right)
                assert sequence_length == padded_sequence_right_length
                assert encoded_sequence == padded_sequence_right

    def test_padding_to_multiple_of(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.pad_token is None:
                    self.skipTest("No padding token.")
                else:
                    empty_tokens = tokenizer("", padding=True, pad_to_multiple_of=8)
                    normal_tokens = tokenizer("This is a sample input", padding=True, pad_to_multiple_of=8)
                    for key, value in empty_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    normal_tokens = tokenizer("This", pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertNotEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    # Should also work with truncation
                    normal_tokens = tokenizer("This", padding=True, truncation=True, pad_to_multiple_of=8)
                    for key, value in normal_tokens.items():
                        self.assertEqual(len(value) % 8, 0, f"BatchEncoding.{key} is not multiple of 8")

                    # truncation to something which is not a multiple of pad_to_multiple_of raises an error
                    self.assertRaises(
                        ValueError,
                        tokenizer.__call__,
                        "This",
                        padding=True,
                        truncation=True,
                        max_length=12,
                        pad_to_multiple_of=8,
                    )

    def test_encode_plus_with_padding(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "Sequence"

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequence)

                padding_size = 10
                padding_idx = tokenizer.pad_token_id
                token_type_padding_idx = tokenizer.pad_token_type_id

                encoded_sequence = tokenizer.encode_plus(sequence, return_special_tokens_mask=True)
                input_ids = encoded_sequence["input_ids"]
                special_tokens_mask = encoded_sequence["special_tokens_mask"]
                sequence_length = len(input_ids)

                # Test 'longest' and 'no_padding' don't do anything
                tokenizer.padding_side = "right"

                not_padded_sequence = tokenizer.encode_plus(
                    sequence,
                    padding=True,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                assert sequence_length == not_padded_sequence_length
                assert input_ids == not_padded_input_ids
                assert special_tokens_mask == not_padded_special_tokens_mask

                not_padded_sequence = tokenizer.encode_plus(
                    sequence,
                    padding=False,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                assert sequence_length == not_padded_sequence_length
                assert input_ids == not_padded_input_ids
                assert special_tokens_mask == not_padded_special_tokens_mask

                # Test right padding
                tokenizer.padding_side = "right"

                right_padded_sequence = tokenizer.encode_plus(
                    sequence,
                    max_length=sequence_length + padding_size,
                    padding="max_length",
                    return_special_tokens_mask=True,
                )
                right_padded_input_ids = right_padded_sequence["input_ids"]

                right_padded_special_tokens_mask = right_padded_sequence["special_tokens_mask"]
                right_padded_sequence_length = len(right_padded_input_ids)

                assert sequence_length + padding_size == right_padded_sequence_length
                assert input_ids + [padding_idx] * padding_size == right_padded_input_ids
                assert special_tokens_mask + [1] * padding_size == right_padded_special_tokens_mask

                # Test left padding
                tokenizer.padding_side = "left"
                left_padded_sequence = tokenizer.encode_plus(
                    sequence,
                    max_length=sequence_length + padding_size,
                    padding="max_length",
                    return_special_tokens_mask=True,
                )
                left_padded_input_ids = left_padded_sequence["input_ids"]
                left_padded_special_tokens_mask = left_padded_sequence["special_tokens_mask"]
                left_padded_sequence_length = len(left_padded_input_ids)

                assert sequence_length + padding_size == left_padded_sequence_length
                assert [padding_idx] * padding_size + input_ids == left_padded_input_ids
                assert [1] * padding_size + special_tokens_mask == left_padded_special_tokens_mask

                if "token_type_ids" in tokenizer.model_input_names:
                    token_type_ids = encoded_sequence["token_type_ids"]
                    left_padded_token_type_ids = left_padded_sequence["token_type_ids"]
                    right_padded_token_type_ids = right_padded_sequence["token_type_ids"]

                    assert token_type_ids + [token_type_padding_idx] * padding_size == right_padded_token_type_ids
                    assert [token_type_padding_idx] * padding_size + token_type_ids == left_padded_token_type_ids

                if "attention_mask" in tokenizer.model_input_names:
                    attention_mask = encoded_sequence["attention_mask"]
                    right_padded_attention_mask = right_padded_sequence["attention_mask"]
                    left_padded_attention_mask = left_padded_sequence["attention_mask"]

                    assert attention_mask + [0] * padding_size == right_padded_attention_mask
                    assert [0] * padding_size + attention_mask == left_padded_attention_mask

    def test_separate_tokenizers(self):
        # This tests that tokenizers don't impact others. Unfortunately the case where it fails is when
        # we're loading an S3 configuration from a pre-trained identifier, and we have no way of testing those today.

        tokenizers = self.get_tokenizers(random_argument=True)
        new_tokenizers = self.get_tokenizers(random_argument=False)

        for tokenizer, new_tokenizer in zip(tokenizers, new_tokenizers):
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                assert tokenizer.init_kwargs["random_argument"] is True
                assert tokenizer.init_kwargs["random_argument"] is True
                assert new_tokenizer.init_kwargs["random_argument"] is False

    def test_get_vocab(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_dict = tokenizer.get_vocab()
                self.assertIsInstance(vocab_dict, dict)
                self.assertGreaterEqual(len(tokenizer), len(vocab_dict))

                vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
                self.assertEqual(len(vocab), len(tokenizer))

                tokenizer.add_tokens(["asdfasdfasdfasdf"])
                vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
                self.assertEqual(len(vocab), len(tokenizer))

    def test_conversion_reversible(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab = tokenizer.get_vocab()
                for word, ind in vocab.items():
                    if word == tokenizer.unk_token:
                        continue
                    self.assertEqual(tokenizer.convert_tokens_to_ids(word), ind)
                    self.assertEqual(tokenizer.convert_ids_to_tokens(ind), word)

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                # Test not batched
                encoded_sequences_1 = tokenizer.encode_plus(sequences[0])
                encoded_sequences_2 = tokenizer(sequences[0])
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test not batched pairs
                encoded_sequences_1 = tokenizer.encode_plus(sequences[0], sequences[1])
                encoded_sequences_2 = tokenizer(sequences[0], sequences[1])
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched
                encoded_sequences_1 = tokenizer.batch_encode_plus(sequences)
                encoded_sequences_2 = tokenizer(sequences)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched pairs
                encoded_sequences_1 = tokenizer.batch_encode_plus(list(zip(sequences, sequences)))
                encoded_sequences_2 = tokenizer(sequences, sequences)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

    def test_batch_encode_plus_batch_sequence_length(self):
        # Tests that all encoded values have the correct size
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                encoded_sequences = [tokenizer.encode_plus(sequence) for sequence in sequences]
                encoded_sequences_batch = tokenizer.batch_encode_plus(sequences, padding=False)
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

                maximum_length = len(
                    max([encoded_sequence["input_ids"] for encoded_sequence in encoded_sequences], key=len)
                )

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences_padded = [
                    tokenizer.encode_plus(sequence, max_length=maximum_length, padding="max_length")
                    for sequence in sequences
                ]

                encoded_sequences_batch_padded = tokenizer.batch_encode_plus(sequences, padding=True)
                self.assertListEqual(
                    encoded_sequences_padded,
                    self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch_padded),
                )

                # check 'longest' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode_plus(sequences, padding=True)
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode_plus(
                    sequences, max_length=maximum_length + 10, padding="longest"
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

                # check 'no_padding' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer.batch_encode_plus(sequences, padding=False)
                encoded_sequences_batch_padded_2 = tokenizer.batch_encode_plus(
                    sequences, max_length=maximum_length + 10, padding=False
                )
                for key in encoded_sequences_batch_padded_1.keys():
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

    @require_tokenizers
    def test_added_token_are_matched_longest_first(self):
        if not self.test_slow_tokenizer:
            self.skipTest("This test is only for slow tokenizers")
            return
        tokenizers = self.get_tokenizers(fast=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                try:
                    tokenizer.add_tokens([AddedToken("extra_id_1")])
                    tokenizer.add_tokens([AddedToken("extra_id_100")])
                except Exception:
                    # Canine cannot add tokens which are not codepoints
                    self.skipTest("Cannot add those Added tokens")

                # XXX: This used to split on `extra_id_1` first we're matching
                # longest first now.
                tokens = tokenizer.tokenize("This is some extra_id_100")
                self.assertIn("extra_id_100", tokens)

        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokenizer.add_tokens([AddedToken("extra_id_100")])
                tokenizer.add_tokens([AddedToken("extra_id_1")])

                tokens = tokenizer.tokenize("This is some extra_id_100")
                self.assertIn("extra_id_100", tokens)

    @require_tokenizers
    def test_added_token_serializable(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                new_token = AddedToken("new_token", lstrip=True)
                tokenizer.add_special_tokens({"additional_special_tokens": [new_token]})

                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    tokenizer.save_pretrained(tmp_dir_name)
                    tokenizer.from_pretrained(tmp_dir_name)

    def test_batch_encode_plus_padding(self):
        # Test that padded sequences are equivalent between batch_encode_plus and encode_plus

        # Right padding tests
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                max_length = 100

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences = [
                    tokenizer.encode_plus(sequence, max_length=max_length, padding="max_length")
                    for sequence in sequences
                ]
                encoded_sequences_batch = tokenizer.batch_encode_plus(
                    sequences, max_length=max_length, padding="max_length"
                )
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

        # Left padding tests
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokenizer.padding_side = "left"
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                max_length = 100

                # check correct behaviour if no pad_token_id exists and add it eventually
                self._check_no_pad_token_padding(tokenizer, sequences)

                encoded_sequences = [
                    tokenizer.encode_plus(sequence, max_length=max_length, padding="max_length")
                    for sequence in sequences
                ]
                encoded_sequences_batch = tokenizer.batch_encode_plus(
                    sequences, max_length=max_length, padding="max_length"
                )
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_encode_plus_format_to_encode_plus(encoded_sequences_batch)
                )

    def test_pretokenized_inputs(self):
        # Test when inputs are pretokenized

        tokenizers = self.get_tokenizers(do_lower_case=False)  # , add_prefix_space=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                if hasattr(tokenizer, "add_prefix_space") and not tokenizer.add_prefix_space:
                    continue

                # Prepare a sequence from our tokenizer vocabulary
                sequence, ids = self.get_clean_sequence(tokenizer, with_prefix_space=True, max_length=20)
                # sequence = " " + sequence  # To be sure the byte-level tokenizers are feeling good
                token_sequence = sequence.split()
                # sequence_no_prefix_space = sequence.strip()

                # Test encode for pretokenized inputs
                output = tokenizer.encode(token_sequence, is_split_into_words=True, add_special_tokens=False)
                output_sequence = tokenizer.encode(sequence, add_special_tokens=False)
                self.assertEqual(output, output_sequence)

                output = tokenizer.encode(token_sequence, is_split_into_words=True, add_special_tokens=True)
                output_sequence = tokenizer.encode(sequence, add_special_tokens=True)
                self.assertEqual(output, output_sequence)

                # Test encode_plus for pretokenized inputs
                output = tokenizer.encode_plus(token_sequence, is_split_into_words=True, add_special_tokens=False)
                output_sequence = tokenizer.encode_plus(sequence, add_special_tokens=False)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])
                output = tokenizer.encode_plus(token_sequence, is_split_into_words=True, add_special_tokens=True)
                output_sequence = tokenizer.encode_plus(sequence, add_special_tokens=True)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

                # Test batch_encode_plus for pretokenized inputs
                sequence_batch = [sequence.strip()] * 2 + [sequence.strip() + " " + sequence.strip()]
                token_sequence_batch = [s.split() for s in sequence_batch]
                sequence_batch_cleaned_up_spaces = [" " + " ".join(s) for s in token_sequence_batch]

                output = tokenizer.batch_encode_plus(
                    token_sequence_batch, is_split_into_words=True, add_special_tokens=False
                )
                output_sequence = tokenizer.batch_encode_plus(
                    sequence_batch_cleaned_up_spaces, add_special_tokens=False
                )
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])
                output = tokenizer.batch_encode_plus(
                    token_sequence_batch, is_split_into_words=True, add_special_tokens=True
                )
                output_sequence = tokenizer.batch_encode_plus(
                    sequence_batch_cleaned_up_spaces, add_special_tokens=True
                )
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

                # Test encode for pretokenized inputs pairs
                output = tokenizer.encode(
                    token_sequence, token_sequence, is_split_into_words=True, add_special_tokens=False
                )
                output_sequence = tokenizer.encode(sequence, sequence, add_special_tokens=False)
                self.assertEqual(output, output_sequence)
                output = tokenizer.encode(
                    token_sequence, token_sequence, is_split_into_words=True, add_special_tokens=True
                )
                output_sequence = tokenizer.encode(sequence, sequence, add_special_tokens=True)
                self.assertEqual(output, output_sequence)

                # Test encode_plus for pretokenized inputs pairs
                output = tokenizer.encode_plus(
                    token_sequence, token_sequence, is_split_into_words=True, add_special_tokens=False
                )
                output_sequence = tokenizer.encode_plus(sequence, sequence, add_special_tokens=False)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])
                output = tokenizer.encode_plus(
                    token_sequence, token_sequence, is_split_into_words=True, add_special_tokens=True
                )
                output_sequence = tokenizer.encode_plus(sequence, sequence, add_special_tokens=True)
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

                # Test batch_encode_plus for pretokenized inputs pairs
                sequence_pair_batch = [(sequence.strip(), sequence.strip())] * 2 + [
                    (sequence.strip() + " " + sequence.strip(), sequence.strip())
                ]
                token_sequence_pair_batch = [tuple(s.split() for s in pair) for pair in sequence_pair_batch]
                sequence_pair_batch_cleaned_up_spaces = [
                    tuple(" " + " ".join(s) for s in pair) for pair in token_sequence_pair_batch
                ]

                output = tokenizer.batch_encode_plus(
                    token_sequence_pair_batch, is_split_into_words=True, add_special_tokens=False
                )
                output_sequence = tokenizer.batch_encode_plus(
                    sequence_pair_batch_cleaned_up_spaces, add_special_tokens=False
                )
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])
                output = tokenizer.batch_encode_plus(
                    token_sequence_pair_batch, is_split_into_words=True, add_special_tokens=True
                )
                output_sequence = tokenizer.batch_encode_plus(
                    sequence_pair_batch_cleaned_up_spaces, add_special_tokens=True
                )
                for key in output.keys():
                    self.assertEqual(output[key], output_sequence[key])

    def test_prepare_for_model(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                string_sequence = "Testing the prepare_for_model method."
                ids = tokenizer.encode(string_sequence, add_special_tokens=False)
                prepared_input_dict = tokenizer.prepare_for_model(ids, add_special_tokens=True)

                input_dict = tokenizer.encode_plus(string_sequence, add_special_tokens=True)

                self.assertEqual(input_dict, prepared_input_dict)

    def test_batch_encode_plus_overflowing_tokens(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            string_sequences = ["Testing the prepare_for_model method.", "Test"]

            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            tokenizer.batch_encode_plus(
                string_sequences, return_overflowing_tokens=True, truncation=True, padding=True, max_length=3
            )

    @is_pt_tf_cross_test
    def test_batch_encode_plus_tensors(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequences = [
                    "Testing batch encode plus",
                    "Testing batch encode plus with different sequence lengths",
                    "Testing batch encode plus with different sequence lengths correctly pads",
                ]

                # A Tensor cannot be build by sequences which are not the same size
                self.assertRaises(ValueError, tokenizer.batch_encode_plus, sequences, return_tensors="pt")
                self.assertRaises(ValueError, tokenizer.batch_encode_plus, sequences, return_tensors="tf")

                if tokenizer.pad_token_id is None:
                    self.assertRaises(
                        ValueError,
                        tokenizer.batch_encode_plus,
                        sequences,
                        padding=True,
                        return_tensors="pt",
                    )
                    self.assertRaises(
                        ValueError,
                        tokenizer.batch_encode_plus,
                        sequences,
                        padding="longest",
                        return_tensors="tf",
                    )
                else:
                    pytorch_tensor = tokenizer.batch_encode_plus(sequences, padding=True, return_tensors="pt")
                    tensorflow_tensor = tokenizer.batch_encode_plus(sequences, padding="longest", return_tensors="tf")
                    encoded_sequences = tokenizer.batch_encode_plus(sequences, padding=True)

                    for key in encoded_sequences.keys():
                        pytorch_value = pytorch_tensor[key].tolist()
                        tensorflow_value = tensorflow_tensor[key].numpy().tolist()
                        encoded_value = encoded_sequences[key]

                        self.assertEqual(pytorch_value, tensorflow_value, encoded_value)

    def _check_no_pad_token_padding(self, tokenizer, sequences):
        # if tokenizer does not have pad_token_id, an error should be thrown
        if tokenizer.pad_token_id is None:
            with self.assertRaises(ValueError):
                if isinstance(sequences, list):
                    tokenizer.batch_encode_plus(sequences, padding="longest")
                else:
                    tokenizer.encode_plus(sequences, padding=True)

            # add pad_token_id to pass subsequent tests
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    def check_subword_sampling(
        self,
        tokenizer: PreTrainedTokenizer,
        text: str = None,
    ) -> None:
        """
        Check if the tokenizer generates different results when subword regularization is enabled.

        Subword regularization augments training data with subword sampling.
        This has a random component.

        Args:
            tokenizer: The tokenizer to check.
            text: The text to use for the checks.
        """
        text = "This is a test for subword regularization." if text is None else text
        if self.test_sentencepiece_ignore_case:
            text = text.lower()

        tokens_list = []
        for _ in range(5):
            tokens_list.append(tokenizer.tokenize(text))

        # the list of different pairs of tokens_list
        combinations = itertools.combinations(tokens_list, 2)

        # check of sampling is done
        subword_sampling_found = False
        for combination in combinations:
            if combination[0] != combination[1]:
                subword_sampling_found = True
        self.assertTrue(subword_sampling_found)

        # check if converting back to original text works
        for tokens in tokens_list:
            if self.test_sentencepiece_ignore_case:
                self.assertEqual(text, tokenizer.convert_tokens_to_string(tokens).lower())
            else:
                self.assertEqual(text, tokenizer.convert_tokens_to_string(tokens))

    @require_torch
    @slow
    def test_torch_encode_plus_sent_to_model(self):
        import torch

        from transformers import MODEL_MAPPING, TOKENIZER_MAPPING

        MODEL_TOKENIZER_MAPPING = merge_model_tokenizer_mappings(MODEL_MAPPING, TOKENIZER_MAPPING)

        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):

                if tokenizer.__class__ not in MODEL_TOKENIZER_MAPPING:
                    return

                config_class, model_class = MODEL_TOKENIZER_MAPPING[tokenizer.__class__]
                config = config_class()

                if config.is_encoder_decoder or config.pad_token_id is None:
                    return

                model = model_class(config)

                # Make sure the model contains at least the full vocabulary size in its embedding matrix
                is_using_common_embeddings = hasattr(model.get_input_embeddings(), "weight")
                assert (
                    (model.get_input_embeddings().weight.shape[0] >= len(tokenizer))
                    if is_using_common_embeddings
                    else True
                )

                # Build sequence
                first_ten_tokens = list(tokenizer.get_vocab().keys())[:10]
                sequence = " ".join(first_ten_tokens)
                encoded_sequence = tokenizer.encode_plus(sequence, return_tensors="pt")

                # Ensure that the BatchEncoding.to() method works.
                encoded_sequence.to(model.device)

                batch_encoded_sequence = tokenizer.batch_encode_plus([sequence, sequence], return_tensors="pt")
                # This should not fail

                with torch.no_grad():  # saves some time
                    model(**encoded_sequence)
                    model(**batch_encoded_sequence)

        # if self.test_rust_tokenizer:
        #     fast_tokenizer = self.get_rust_tokenizer()
        #     encoded_sequence_fast = fast_tokenizer.encode_plus(sequence, return_tensors="pt")
        #     batch_encoded_sequence_fast = fast_tokenizer.batch_encode_plus([sequence, sequence], return_tensors="pt")
        #     # This should not fail
        #     model(**encoded_sequence_fast)
        #     model(**batch_encoded_sequence_fast)

    @require_tf
    @slow
    def test_tf_encode_plus_sent_to_model(self):
        from transformers import TF_MODEL_MAPPING, TOKENIZER_MAPPING

        MODEL_TOKENIZER_MAPPING = merge_model_tokenizer_mappings(TF_MODEL_MAPPING, TOKENIZER_MAPPING)

        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.__class__ not in MODEL_TOKENIZER_MAPPING:
                    return

                config_class, model_class = MODEL_TOKENIZER_MAPPING[tokenizer.__class__]
                config = config_class()

                if config.is_encoder_decoder or config.pad_token_id is None:
                    return

                model = model_class(config)

                # Make sure the model contains at least the full vocabulary size in its embedding matrix
                assert model.config.vocab_size >= len(tokenizer)

                # Build sequence
                first_ten_tokens = list(tokenizer.get_vocab().keys())[:10]
                sequence = " ".join(first_ten_tokens)
                encoded_sequence = tokenizer.encode_plus(sequence, return_tensors="tf")
                batch_encoded_sequence = tokenizer.batch_encode_plus([sequence, sequence], return_tensors="tf")

                # This should not fail
                model(encoded_sequence)
                model(batch_encoded_sequence)

    # TODO: Check if require_torch is the best to test for numpy here ... Maybe move to require_flax when available
    @require_torch
    @slow
    def test_np_encode_plus_sent_to_model(self):
        from transformers import MODEL_MAPPING, TOKENIZER_MAPPING

        MODEL_TOKENIZER_MAPPING = merge_model_tokenizer_mappings(MODEL_MAPPING, TOKENIZER_MAPPING)

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.__class__ not in MODEL_TOKENIZER_MAPPING:
                    return

                config_class, model_class = MODEL_TOKENIZER_MAPPING[tokenizer.__class__]
                config = config_class()

                if config.is_encoder_decoder or config.pad_token_id is None:
                    return

                # Build sequence
                first_ten_tokens = list(tokenizer.get_vocab().keys())[:10]
                sequence = " ".join(first_ten_tokens)
                encoded_sequence = tokenizer.encode_plus(sequence, return_tensors="np")
                batch_encoded_sequence = tokenizer.batch_encode_plus([sequence, sequence], return_tensors="np")

                # TODO: add forward through JAX/Flax when PR is merged
                # This is currently here to make flake8 happy !
                if encoded_sequence is None:
                    raise ValueError("Cannot convert list to numpy tensor on  encode_plus()")

                if batch_encoded_sequence is None:
                    raise ValueError("Cannot convert list to numpy tensor on  batch_encode_plus()")

                if self.test_rust_tokenizer:
                    fast_tokenizer = self.get_rust_tokenizer()
                    encoded_sequence_fast = fast_tokenizer.encode_plus(sequence, return_tensors="np")
                    batch_encoded_sequence_fast = fast_tokenizer.batch_encode_plus(
                        [sequence, sequence], return_tensors="np"
                    )

                    # TODO: add forward through JAX/Flax when PR is merged
                    # This is currently here to make flake8 happy !
                    if encoded_sequence_fast is None:
                        raise ValueError("Cannot convert list to numpy tensor on  encode_plus() (fast)")

                    if batch_encoded_sequence_fast is None:
                        raise ValueError("Cannot convert list to numpy tensor on  batch_encode_plus() (fast)")

    @require_torch
    def test_prepare_seq2seq_batch(self):
        if not self.test_seq2seq:
            return

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Longer text that will definitely require truncation.
                src_text = [
                    " UN Chief Says There Is No Military Solution in Syria",
                    " Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for Syria is that 'there is no military solution' to the nearly five-year conflict and more weapons will only worsen the violence and misery for millions of people.",
                ]
                tgt_text = [
                    "Şeful ONU declară că nu există o soluţie militară în Siria",
                    "Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar al Rusiei "
                    'pentru Siria este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi că noi arme nu '
                    "vor face decât să înrăutăţească violenţele şi mizeria pentru milioane de oameni.",
                ]
                try:
                    batch = tokenizer.prepare_seq2seq_batch(
                        src_texts=src_text,
                        tgt_texts=tgt_text,
                        max_length=3,
                        max_target_length=10,
                        return_tensors="pt",
                        src_lang="en_XX",  # this should be ignored (for all but mbart) but not cause an error
                    )
                except NotImplementedError:
                    return
                self.assertEqual(batch.input_ids.shape[1], 3)
                self.assertEqual(batch.labels.shape[1], 10)
                # max_target_length will default to max_length if not specified
                batch = tokenizer.prepare_seq2seq_batch(
                    src_text, tgt_texts=tgt_text, max_length=3, return_tensors="pt"
                )
                self.assertEqual(batch.input_ids.shape[1], 3)
                self.assertEqual(batch.labels.shape[1], 3)

                batch_encoder_only = tokenizer.prepare_seq2seq_batch(
                    src_texts=src_text, max_length=3, max_target_length=10, return_tensors="pt"
                )
                self.assertEqual(batch_encoder_only.input_ids.shape[1], 3)
                self.assertEqual(batch_encoder_only.attention_mask.shape[1], 3)
                self.assertNotIn("decoder_input_ids", batch_encoder_only)

    def test_is_fast(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                # Check is_fast is set correctly
                self.assertTrue(tokenizer_r.is_fast)

                if self.test_slow_tokenizer:
                    tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                    self.assertFalse(tokenizer_p.is_fast)

    def test_fast_only_inputs(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                # Ensure None raise an error
                self.assertRaises(TypeError, tokenizer_r.tokenize, None)
                self.assertRaises(TypeError, tokenizer_r.encode, None)
                self.assertRaises(TypeError, tokenizer_r.encode_plus, None)
                self.assertRaises(TypeError, tokenizer_r.batch_encode_plus, None)

    def test_alignement_methods(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

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
                self.assertEqual(
                    batch_encoding.word_to_tokens(last_batch_index, last_word_index).end, last_token_index + 1
                )

                # Assert token_to_chars
                self.assertEqual(encoding.token_to_chars(0).start, 0)
                self.assertEqual(encoding.token_to_chars(0, 0).start, 0)
                self.assertEqual(encoding.token_to_chars(last_token_index).end, last_char_index + 1)
                self.assertEqual(encoding.token_to_chars(0, last_token_index).end, last_char_index + 1)
                self.assertEqual(batch_encoding.token_to_chars(1, 0).start, 0)
                self.assertEqual(batch_encoding.token_to_chars(0, last_token_index).end, last_char_index + 1)
                self.assertEqual(
                    batch_encoding.token_to_chars(last_batch_index, last_token_index).end, last_char_index + 1
                )

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
                self.assertEqual(
                    batch_encoding.word_to_chars(last_batch_index, last_word_index).end, last_char_index + 1
                )

                # Assert token_to_sequence
                self.assertEqual(encoding.token_to_sequence(num_tokens // 2), 0)
                self.assertEqual(encoding.token_to_sequence(0, num_tokens // 2), 0)
                self.assertEqual(batch_encoding.token_to_sequence(1, num_tokens // 2), 0)
                self.assertEqual(batch_encoding.token_to_sequence(0, num_tokens // 2), 0)
                self.assertEqual(batch_encoding.token_to_sequence(last_batch_index, num_tokens // 2), 0)

                # Pair of input sequences

                words = ["Wonderful", "no", "inspiration", "example", "with", "subtoken"]
                text = " ".join(words)
                pair_words = ["Amazing", "example", "full", "of", "inspiration"]
                pair_text = " ".join(pair_words)
                batch_size = 3
                index_word_in_first_seq = words.index("inspiration")
                index_word_in_pair_seq = pair_words.index("inspiration")
                index_char_in_first_seq = text.find("inspiration")
                index_char_in_pair_seq = pair_text.find("inspiration")

                pair_encoding = tokenizer_r.encode_plus(text, pair_text, add_special_tokens=False)

                pair_batch_encoding = tokenizer_r.batch_encode_plus(
                    [(text, pair_text)] * batch_size, add_special_tokens=False
                )
                num_tokens = len(encoding["input_ids"])

                last_word_index = len(words) - 1
                last_token_index = num_tokens - 1
                last_batch_index = batch_size - 1
                last_char_index = len(text) - 1

                # Assert word_to_tokens
                self.assertNotEqual(
                    pair_encoding.word_to_tokens(index_word_in_first_seq, sequence_index=0).start,
                    pair_encoding.word_to_tokens(index_word_in_pair_seq, sequence_index=1).start,
                )
                self.assertEqual(
                    pair_encoding["input_ids"][
                        pair_encoding.word_to_tokens(index_word_in_first_seq, sequence_index=0).start
                    ],
                    pair_encoding["input_ids"][
                        pair_encoding.word_to_tokens(index_word_in_pair_seq, sequence_index=1).start
                    ],
                )
                self.assertNotEqual(
                    pair_batch_encoding.word_to_tokens(1, index_word_in_first_seq, sequence_index=0).start,
                    pair_batch_encoding.word_to_tokens(1, index_word_in_pair_seq, sequence_index=1).start,
                )
                self.assertEqual(
                    pair_batch_encoding["input_ids"][1][
                        pair_batch_encoding.word_to_tokens(1, index_word_in_first_seq, sequence_index=0).start
                    ],
                    pair_batch_encoding["input_ids"][1][
                        pair_batch_encoding.word_to_tokens(1, index_word_in_pair_seq, sequence_index=1).start
                    ],
                )

                # Assert char_to_token
                self.assertNotEqual(
                    pair_encoding.char_to_token(index_char_in_first_seq, sequence_index=0),
                    pair_encoding.char_to_token(index_char_in_pair_seq, sequence_index=1),
                )
                self.assertEqual(
                    pair_encoding["input_ids"][pair_encoding.char_to_token(index_char_in_first_seq, sequence_index=0)],
                    pair_encoding["input_ids"][pair_encoding.char_to_token(index_char_in_pair_seq, sequence_index=1)],
                )
                self.assertNotEqual(
                    pair_batch_encoding.char_to_token(1, index_char_in_first_seq, sequence_index=0),
                    pair_batch_encoding.char_to_token(1, index_char_in_pair_seq, sequence_index=1),
                )
                self.assertEqual(
                    pair_batch_encoding["input_ids"][1][
                        pair_batch_encoding.char_to_token(1, index_char_in_first_seq, sequence_index=0)
                    ],
                    pair_batch_encoding["input_ids"][1][
                        pair_batch_encoding.char_to_token(1, index_char_in_pair_seq, sequence_index=1)
                    ],
                )

                # Assert char_to_word
                self.assertNotEqual(
                    pair_encoding.char_to_word(index_char_in_first_seq, sequence_index=0),
                    pair_encoding.char_to_word(index_char_in_pair_seq, sequence_index=1),
                )
                self.assertEqual(
                    words[pair_encoding.char_to_word(index_char_in_first_seq, sequence_index=0)],
                    pair_words[pair_encoding.char_to_word(index_char_in_pair_seq, sequence_index=1)],
                )
                self.assertNotEqual(
                    pair_batch_encoding.char_to_word(1, index_char_in_first_seq, sequence_index=0),
                    pair_batch_encoding.char_to_word(1, index_char_in_pair_seq, sequence_index=1),
                )
                self.assertEqual(
                    words[pair_batch_encoding.char_to_word(1, index_char_in_first_seq, sequence_index=0)],
                    pair_words[pair_batch_encoding.char_to_word(1, index_char_in_pair_seq, sequence_index=1)],
                )

                # Assert word_to_chars
                self.assertNotEqual(
                    pair_encoding.word_to_chars(index_word_in_first_seq, sequence_index=0).start,
                    pair_encoding.word_to_chars(index_word_in_pair_seq, sequence_index=1).start,
                )
                self.assertEqual(
                    text[pair_encoding.word_to_chars(index_word_in_first_seq, sequence_index=0).start],
                    pair_text[pair_encoding.word_to_chars(index_word_in_pair_seq, sequence_index=1).start],
                )
                self.assertNotEqual(
                    pair_batch_encoding.word_to_chars(1, index_word_in_first_seq, sequence_index=0).start,
                    pair_batch_encoding.word_to_chars(1, index_word_in_pair_seq, sequence_index=1).start,
                )
                self.assertEqual(
                    text[pair_batch_encoding.word_to_chars(1, index_word_in_first_seq, sequence_index=0).start],
                    pair_text[pair_batch_encoding.word_to_chars(1, index_word_in_pair_seq, sequence_index=1).start],
                )

                # Assert token_to_sequence
                pair_encoding = tokenizer_r.encode_plus(text, pair_text, add_special_tokens=True)

                pair_sequence_ids = [
                    pair_encoding.token_to_sequence(i) for i in range(len(pair_encoding["input_ids"]))
                ]
                self.assertIn(0, pair_sequence_ids)
                self.assertIn(1, pair_sequence_ids)
                if tokenizer_r.num_special_tokens_to_add(pair=True):
                    self.assertIn(None, pair_sequence_ids)

                pair_batch_encoding = tokenizer_r.batch_encode_plus(
                    [(text, pair_text)] * batch_size, add_special_tokens=True
                )
                pair_batch_sequence_ids = [
                    pair_batch_encoding.token_to_sequence(1, i)
                    for i in range(len(pair_batch_encoding["input_ids"][0]))
                ]
                self.assertIn(0, pair_batch_sequence_ids)
                self.assertIn(1, pair_batch_sequence_ids)
                if tokenizer_r.num_special_tokens_to_add(pair=True):
                    self.assertIn(None, pair_batch_sequence_ids)

    def test_tokenization_python_rust_equals(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

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

    def test_num_special_tokens_to_add_equal(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                # Check we have the same number of added_tokens for both pair and non-pair inputs.
                self.assertEqual(
                    tokenizer_r.num_special_tokens_to_add(False), tokenizer_p.num_special_tokens_to_add(False)
                )
                self.assertEqual(
                    tokenizer_r.num_special_tokens_to_add(True), tokenizer_p.num_special_tokens_to_add(True)
                )

    def test_max_length_equal(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                # Check we have the correct max_length for both pair and non-pair inputs.
                self.assertEqual(tokenizer_r.max_len_single_sentence, tokenizer_p.max_len_single_sentence)
                self.assertEqual(tokenizer_r.max_len_sentences_pair, tokenizer_p.max_len_sentences_pair)

    def test_special_tokens_map_equal(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                # Assert the set of special tokens match.
                self.assertSequenceEqual(
                    tokenizer_p.special_tokens_map.items(),
                    tokenizer_r.special_tokens_map.items(),
                )

    def test_add_tokens(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                vocab_size = len(tokenizer_r)
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
                self.assertIn("<testtoken3>", tokenizer_r.special_tokens_map["additional_special_tokens"])
                self.assertIsInstance(tokenizer_r.special_tokens_map["additional_special_tokens"], list)
                self.assertGreaterEqual(len(tokenizer_r.special_tokens_map["additional_special_tokens"]), 2)

                self.assertEqual(len(tokenizer_r), vocab_size + 8)

    def test_offsets_mapping(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

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

    def test_batch_encode_dynamic_overflowing(self):
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
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            tokenizer = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name}, {tokenizer.__class__.__name__})"):

                if is_torch_available():
                    returned_tensor = "pt"
                elif is_tf_available():
                    returned_tensor = "tf"
                else:
                    returned_tensor = "jax"

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

    def test_compare_pretokenized_inputs(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                if hasattr(tokenizer_p, "add_prefix_space") and not tokenizer_p.add_prefix_space:
                    continue  # Too hard to test for now

                # Input string
                pretokenized_input_simple = "This is a sample input".split()
                pretokenized_input_pair = "This is a sample pair".split()

                # Test encode for pretokenized inputs
                output_r = tokenizer_r.encode(
                    pretokenized_input_simple, is_split_into_words=True, add_special_tokens=False
                )
                output_p = tokenizer_p.encode(
                    pretokenized_input_simple, is_split_into_words=True, add_special_tokens=False
                )
                self.assertEqual(output_p, output_r)

                kwargs = {
                    "is_split_into_words": True,
                    # "return_token_type_ids": True,  # Use the defaults for each tokenizers
                    # "return_attention_mask": True,  # Use the defaults for each tokenizers
                    "return_overflowing_tokens": False,
                    "return_special_tokens_mask": True,
                    "return_offsets_mapping": False,  # Not implemented in python tokenizers
                    # "add_special_tokens": False,
                }
                batch_kwargs = {
                    "is_split_into_words": True,
                    # "return_token_type_ids": True,  # Use the defaults for each tokenizers
                    # "return_attention_mask": True,  # Use the defaults for each tokenizers
                    "return_overflowing_tokens": False,
                    "return_special_tokens_mask": True,
                    "return_offsets_mapping": False,  # Not implemented in python tokenizers
                    # "add_special_tokens": False,
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
                output_r = tokenizer_r.encode(
                    pretokenized_input_simple, pretokenized_input_pair, is_split_into_words=True
                )
                output_p = tokenizer_p.encode(
                    pretokenized_input_simple, pretokenized_input_pair, is_split_into_words=True
                )
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

    def test_create_token_type_ids(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
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

    def test_build_inputs_with_special_tokens(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                # # Input string
                # input_simple = tokenizer_p.tokenize("This is a sample input", add_special_tokens=False)
                # input_pair = tokenizer_p.tokenize("This is a sample pair", add_special_tokens=False)

                # # Generate output
                # output_r = tokenizer_r.build_inputs_with_special_tokens(input_simple)
                # output_p = tokenizer_p.build_inputs_with_special_tokens(input_simple)
                # self.assertEqual(output_p, output_r)

                # # Generate pair output
                # output_r = tokenizer_r.build_inputs_with_special_tokens(input_simple, input_pair)
                # output_p = tokenizer_p.build_inputs_with_special_tokens(input_simple, input_pair)
                # self.assertEqual(output_p, output_r)

                # Input tokens id
                input_simple = tokenizer_p.encode("This is a sample input", add_special_tokens=False)
                input_pair = tokenizer_p.encode("This is a sample pair", add_special_tokens=False)

                # Generate output
                output_r = tokenizer_r.build_inputs_with_special_tokens(input_simple)
                output_p = tokenizer_p.build_inputs_with_special_tokens(input_simple)
                self.assertEqual(output_p, output_r)

                # Generate pair output
                output_r = tokenizer_r.build_inputs_with_special_tokens(input_simple, input_pair)
                output_p = tokenizer_p.build_inputs_with_special_tokens(input_simple, input_pair)
                self.assertEqual(output_p, output_r)

    def test_padding(self, max_length=50):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                self.assertEqual(tokenizer_p.pad_token_id, tokenizer_r.pad_token_id)
                pad_token_id = tokenizer_p.pad_token_id

                # Encode - Simple input
                input_r = tokenizer_r.encode("This is a simple input", max_length=max_length, pad_to_max_length=True)
                input_p = tokenizer_p.encode("This is a simple input", max_length=max_length, pad_to_max_length=True)
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)
                input_r = tokenizer_r.encode("This is a simple input", max_length=max_length, padding="max_length")
                input_p = tokenizer_p.encode("This is a simple input", max_length=max_length, padding="max_length")
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.encode("This is a simple input", padding="longest")
                input_p = tokenizer_p.encode("This is a simple input", padding=True)
                self.assert_padded_input_match(input_r, input_p, len(input_r), pad_token_id)

                # Encode - Pair input
                input_r = tokenizer_r.encode(
                    "This is a simple input", "This is a pair", max_length=max_length, pad_to_max_length=True
                )
                input_p = tokenizer_p.encode(
                    "This is a simple input", "This is a pair", max_length=max_length, pad_to_max_length=True
                )
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)
                input_r = tokenizer_r.encode(
                    "This is a simple input", "This is a pair", max_length=max_length, padding="max_length"
                )
                input_p = tokenizer_p.encode(
                    "This is a simple input", "This is a pair", max_length=max_length, padding="max_length"
                )
                self.assert_padded_input_match(input_r, input_p, max_length, pad_token_id)
                input_r = tokenizer_r.encode("This is a simple input", "This is a pair", padding=True)
                input_p = tokenizer_p.encode("This is a simple input", "This is a pair", padding="longest")
                self.assert_padded_input_match(input_r, input_p, len(input_r), pad_token_id)

                # Encode_plus - Simple input
                input_r = tokenizer_r.encode_plus(
                    "This is a simple input", max_length=max_length, pad_to_max_length=True
                )
                input_p = tokenizer_p.encode_plus(
                    "This is a simple input", max_length=max_length, pad_to_max_length=True
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])
                input_r = tokenizer_r.encode_plus(
                    "This is a simple input", max_length=max_length, padding="max_length"
                )
                input_p = tokenizer_p.encode_plus(
                    "This is a simple input", max_length=max_length, padding="max_length"
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])

                input_r = tokenizer_r.encode_plus("This is a simple input", padding="longest")
                input_p = tokenizer_p.encode_plus("This is a simple input", padding=True)
                self.assert_padded_input_match(
                    input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]), pad_token_id
                )

                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])

                # Encode_plus - Pair input
                input_r = tokenizer_r.encode_plus(
                    "This is a simple input", "This is a pair", max_length=max_length, pad_to_max_length=True
                )
                input_p = tokenizer_p.encode_plus(
                    "This is a simple input", "This is a pair", max_length=max_length, pad_to_max_length=True
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])
                input_r = tokenizer_r.encode_plus(
                    "This is a simple input", "This is a pair", max_length=max_length, padding="max_length"
                )
                input_p = tokenizer_p.encode_plus(
                    "This is a simple input", "This is a pair", max_length=max_length, padding="max_length"
                )
                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])
                input_r = tokenizer_r.encode_plus("This is a simple input", "This is a pair", padding="longest")
                input_p = tokenizer_p.encode_plus("This is a simple input", "This is a pair", padding=True)
                self.assert_padded_input_match(
                    input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]), pad_token_id
                )
                self.assertSequenceEqual(input_r["attention_mask"], input_p["attention_mask"])

                # Batch_encode_plus - Simple input
                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    max_length=max_length,
                    pad_to_max_length=True,
                )
                input_p = tokenizer_p.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    max_length=max_length,
                    pad_to_max_length=True,
                )
                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    max_length=max_length,
                    padding="max_length",
                )
                input_p = tokenizer_p.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    max_length=max_length,
                    padding="max_length",
                )
                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    max_length=max_length,
                    padding="longest",
                )
                input_p = tokenizer_p.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"],
                    max_length=max_length,
                    padding=True,
                )
                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"], padding="longest"
                )
                input_p = tokenizer_p.batch_encode_plus(
                    ["This is a simple input 1", "This is a simple input 2"], padding=True
                )
                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

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
                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

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
                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                # Using pad on single examples after tokenization
                input_r = tokenizer_r.encode_plus("This is a input 1")
                input_r = tokenizer_r.pad(input_r)

                input_p = tokenizer_r.encode_plus("This is a input 1")
                input_p = tokenizer_r.pad(input_p)

                self.assert_padded_input_match(
                    input_r["input_ids"], input_p["input_ids"], len(input_r["input_ids"]), pad_token_id
                )

                # Using pad on single examples after tokenization
                input_r = tokenizer_r.encode_plus("This is a input 1")
                input_r = tokenizer_r.pad(input_r, max_length=max_length, padding="max_length")

                input_p = tokenizer_r.encode_plus("This is a input 1")
                input_p = tokenizer_r.pad(input_p, max_length=max_length, padding="max_length")

                self.assert_padded_input_match(input_r["input_ids"], input_p["input_ids"], max_length, pad_token_id)

                # Using pad after tokenization
                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input whilch should be padded"]
                )
                input_r = tokenizer_r.pad(input_r)

                input_p = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input whilch should be padded"]
                )
                input_p = tokenizer_r.pad(input_p)

                self.assert_batch_padded_input_match(input_r, input_p, len(input_r["input_ids"][0]), pad_token_id)

                # Using pad after tokenization
                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input whilch should be padded"]
                )
                input_r = tokenizer_r.pad(input_r, max_length=max_length, padding="max_length")

                input_p = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input whilch should be padded"]
                )
                input_p = tokenizer_r.pad(input_p, max_length=max_length, padding="max_length")

                self.assert_batch_padded_input_match(input_r, input_p, max_length, pad_token_id)

    def test_padding_different_model_input_name(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                self.assertEqual(tokenizer_p.pad_token_id, tokenizer_r.pad_token_id)
                pad_token_id = tokenizer_p.pad_token_id

                input_r = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input whilch should be padded"]
                )
                input_p = tokenizer_r.batch_encode_plus(
                    ["This is a input 1", "This is a much longer input whilch should be padded"]
                )

                # rename encoded batch to "inputs"
                input_r["inputs"] = input_r[tokenizer_r.model_input_names[0]]
                del input_r[tokenizer_r.model_input_names[0]]

                input_p["inputs"] = input_p[tokenizer_p.model_input_names[0]]
                del input_p[tokenizer_p.model_input_names[0]]

                # Renaming `input_ids` to `inputs`
                tokenizer_r.model_input_names = ["inputs"] + tokenizer_r.model_input_names[1:]
                tokenizer_p.model_input_names = ["inputs"] + tokenizer_p.model_input_names[1:]

                input_r = tokenizer_r.pad(input_r, padding="longest")
                input_p = tokenizer_r.pad(input_p, padding="longest")

                max_length = len(input_p["inputs"][0])
                self.assert_batch_padded_input_match(
                    input_r, input_p, max_length, pad_token_id, model_main_input_name="inputs"
                )

    def test_save_pretrained(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it save with the same files + the tokenizer.json file for the fast one
                self.assertTrue(any("tokenizer.json" in f for f in tokenizer_r_files))
                tokenizer_r_files = tuple(f for f in tokenizer_r_files if "tokenizer.json" not in f)
                self.assertSequenceEqual(tokenizer_r_files, tokenizer_p_files)

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))
                    # self.assertEqual(getattr(tokenizer_rp, key), getattr(tokenizer_pp, key))
                    # self.assertEqual(getattr(tokenizer_rp, key + "_id"), getattr(tokenizer_pp, key + "_id"))

                shutil.rmtree(tmpdirname2)

                # Save tokenizer rust, legacy_format=True
                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2, legacy_format=True)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it save with the same files
                self.assertSequenceEqual(tokenizer_r_files, tokenizer_p_files)

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

                # Save tokenizer rust, legacy_format=False
                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2, legacy_format=False)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it saved the tokenizer.json file
                self.assertTrue(any("tokenizer.json" in f for f in tokenizer_r_files))

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

    def test_embeded_special_tokens(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                sentence = "A, <mask> AllenNLP sentence."
                tokens_r = tokenizer_r.encode_plus(
                    sentence,
                    add_special_tokens=True,
                )
                tokens_p = tokenizer_p.encode_plus(
                    sentence,
                    add_special_tokens=True,
                )

                for key in tokens_p.keys():
                    self.assertEqual(tokens_r[key], tokens_p[key])

                if "token_type_ids" in tokens_r:
                    self.assertEqual(sum(tokens_r["token_type_ids"]), sum(tokens_p["token_type_ids"]))

                tokens_r = tokenizer_r.convert_ids_to_tokens(tokens_r["input_ids"])
                tokens_p = tokenizer_p.convert_ids_to_tokens(tokens_p["input_ids"])
                self.assertSequenceEqual(tokens_r, tokens_p)

    def test_compare_add_special_tokens(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                simple_num_special_tokens_to_add = tokenizer_r.num_special_tokens_to_add(pair=False)
                # pair_num_special_tokens_to_add = tokenizer_r.num_special_tokens_to_add(pair=True)

                for text in ["", " "]:
                    # tokenize()
                    no_special_tokens = tokenizer_r.tokenize(text, add_special_tokens=False)
                    with_special_tokens = tokenizer_r.tokenize(text, add_special_tokens=True)
                    self.assertEqual(
                        len(no_special_tokens), len(with_special_tokens) - simple_num_special_tokens_to_add
                    )

                    # encode()
                    no_special_tokens = tokenizer_r.encode(text, add_special_tokens=False)
                    with_special_tokens = tokenizer_r.encode(text, add_special_tokens=True)
                    self.assertEqual(
                        len(no_special_tokens), len(with_special_tokens) - simple_num_special_tokens_to_add
                    )

                    # encode_plus()
                    no_special_tokens = tokenizer_r.encode_plus(text, add_special_tokens=False)
                    with_special_tokens = tokenizer_r.encode_plus(text, add_special_tokens=True)
                    for key in no_special_tokens.keys():
                        self.assertEqual(
                            len(no_special_tokens[key]),
                            len(with_special_tokens[key]) - simple_num_special_tokens_to_add,
                        )

                    # # batch_encode_plus
                    no_special_tokens = tokenizer_r.batch_encode_plus([text, text], add_special_tokens=False)
                    with_special_tokens = tokenizer_r.batch_encode_plus([text, text], add_special_tokens=True)
                    for key in no_special_tokens.keys():
                        for i_no, i_with in zip(no_special_tokens[key], with_special_tokens[key]):
                            self.assertEqual(len(i_no), len(i_with) - simple_num_special_tokens_to_add)

    def test_compare_prepare_for_model(self):
        if not self.test_slow_tokenizer:
            # as we don't have a slow version, we can't compare the outputs between slow and fast versions
            return

        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                string_sequence = "Asserting that both tokenizers are equal"
                python_output = tokenizer_p.prepare_for_model(
                    tokenizer_p.encode(string_sequence, add_special_tokens=False)
                )
                rust_output = tokenizer_r.prepare_for_model(
                    tokenizer_r.encode(string_sequence, add_special_tokens=False)
                )
                for key in python_output:
                    self.assertEqual(python_output[key], rust_output[key])

    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):

                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.rust_tokenizer_class.from_pretrained(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )
                r_output = tokenizer_r.encode("Hey this is a <special> token")

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

                self.assertTrue(special_token_id in r_output)

                if self.test_slow_tokenizer:
                    tokenizer_cr = self.rust_tokenizer_class.from_pretrained(
                        pretrained_name, additional_special_tokens=added_tokens, **kwargs, from_slow=True
                    )
                    tokenizer_p = self.tokenizer_class.from_pretrained(
                        pretrained_name, additional_special_tokens=added_tokens, **kwargs
                    )

                    p_output = tokenizer_p.encode("Hey this is a <special> token")

                    cr_output = tokenizer_cr.encode("Hey this is a <special> token")

                    self.assertEqual(p_output, r_output)
                    self.assertEqual(cr_output, r_output)
                    self.assertTrue(special_token_id in p_output)
                    self.assertTrue(special_token_id in cr_output)

    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(self):
        tokenizer_list = []
        if self.test_slow_tokenizer:
            tokenizer_list.append((self.tokenizer_class, self.get_tokenizer()))

        if self.test_rust_tokenizer:
            tokenizer_list.append((self.rust_tokenizer_class, self.get_rust_tokenizer()))

        for tokenizer_class, tokenizer_utils in tokenizer_list:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tokenizer_utils.save_pretrained(tmp_dir)

                with open(os.path.join(tmp_dir, "special_tokens_map.json"), encoding="utf-8") as json_file:
                    special_tokens_map = json.load(json_file)

                with open(os.path.join(tmp_dir, "tokenizer_config.json"), encoding="utf-8") as json_file:
                    tokenizer_config = json.load(json_file)

                special_tokens_map["additional_special_tokens"] = ["an_additional_special_token"]
                tokenizer_config["additional_special_tokens"] = ["an_additional_special_token"]

                with open(os.path.join(tmp_dir, "special_tokens_map.json"), "w", encoding="utf-8") as outfile:
                    json.dump(special_tokens_map, outfile)
                with open(os.path.join(tmp_dir, "tokenizer_config.json"), "w", encoding="utf-8") as outfile:
                    json.dump(tokenizer_config, outfile)

                # the following checks allow us to verify that our test works as expected, i.e. that the tokenizer takes
                # into account the new value of additional_special_tokens given in the "tokenizer_config.json" and
                # "special_tokens_map.json" files
                tokenizer_without_change_in_init = tokenizer_class.from_pretrained(
                    tmp_dir,
                )
                self.assertIn(
                    "an_additional_special_token", tokenizer_without_change_in_init.additional_special_tokens
                )
                self.assertIn("an_additional_special_token", tokenizer_without_change_in_init.get_vocab())
                self.assertEqual(
                    ["an_additional_special_token"],
                    tokenizer_without_change_in_init.convert_ids_to_tokens(
                        tokenizer_without_change_in_init.convert_tokens_to_ids(["an_additional_special_token"])
                    ),
                )

                # Now we test that we can change the value of additional_special_tokens in the from_pretrained
                new_added_tokens = [AddedToken("a_new_additional_special_token", lstrip=True)]
                tokenizer = tokenizer_class.from_pretrained(
                    tmp_dir,
                    additional_special_tokens=new_added_tokens,
                )

                self.assertIn("a_new_additional_special_token", tokenizer.additional_special_tokens)
                self.assertEqual(
                    ["a_new_additional_special_token"],
                    tokenizer.convert_ids_to_tokens(
                        tokenizer.convert_tokens_to_ids(["a_new_additional_special_token"])
                    ),
                )

    def test_training_new_tokenizer(self):
        # This feature only exists for fast tokenizers
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_rust_tokenizer()
        new_tokenizer = tokenizer.train_new_from_iterator(SMALL_TRAINING_CORPUS, 100)

        # Test we can use the new tokenizer with something not seen during training
        inputs = new_tokenizer(["This is the first sentence", "This sentence is different 🤗."])
        self.assertEqual(len(inputs["input_ids"]), 2)
        decoded_input = new_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        expected_result = "This is the first sentence"

        if tokenizer.backend_tokenizer.normalizer is not None:
            expected_result = tokenizer.backend_tokenizer.normalizer.normalize_str(expected_result)
        self.assertEqual(expected_result, decoded_input)

        # We check that the parameters of the tokenizer remained the same
        # Check we have the same number of added_tokens for both pair and non-pair inputs.
        self.assertEqual(tokenizer.num_special_tokens_to_add(False), new_tokenizer.num_special_tokens_to_add(False))
        self.assertEqual(tokenizer.num_special_tokens_to_add(True), new_tokenizer.num_special_tokens_to_add(True))

        # Check we have the correct max_length for both pair and non-pair inputs.
        self.assertEqual(tokenizer.max_len_single_sentence, new_tokenizer.max_len_single_sentence)
        self.assertEqual(tokenizer.max_len_sentences_pair, new_tokenizer.max_len_sentences_pair)

        # Assert the set of special tokens match as we didn't ask to change them
        self.assertSequenceEqual(
            tokenizer.all_special_tokens_extended,
            new_tokenizer.all_special_tokens_extended,
        )

        self.assertDictEqual(tokenizer.special_tokens_map, new_tokenizer.special_tokens_map)

    def test_training_new_tokenizer_with_special_tokens_change(self):
        # This feature only exists for fast tokenizers
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_rust_tokenizer()
        # Test with a special tokens map
        class_signature = inspect.signature(tokenizer.__class__)
        if "cls_token" in class_signature.parameters:
            new_tokenizer = tokenizer.train_new_from_iterator(
                SMALL_TRAINING_CORPUS, 100, special_tokens_map={tokenizer.cls_token: "<cls>"}
            )
            cls_id = new_tokenizer.get_vocab()["<cls>"]
            self.assertEqual(new_tokenizer.cls_token, "<cls>")
            self.assertEqual(new_tokenizer.cls_token_id, cls_id)

        # Create a new mapping from the special tokens defined in the original tokenizer
        special_tokens_list = SpecialTokensMixin.SPECIAL_TOKENS_ATTRIBUTES.copy()
        special_tokens_list.remove("additional_special_tokens")
        special_tokens_map = {}
        for token in special_tokens_list:
            # Get the private one to avoid unnecessary warnings.
            if getattr(tokenizer, f"_{token}") is not None:
                special_token = getattr(tokenizer, token)
                special_tokens_map[special_token] = f"{special_token}a"

        # Train new tokenizer
        new_tokenizer = tokenizer.train_new_from_iterator(
            SMALL_TRAINING_CORPUS, 100, special_tokens_map=special_tokens_map
        )

        # Check the changes
        for token in special_tokens_list:
            # Get the private one to avoid unnecessary warnings.
            if getattr(tokenizer, f"_{token}") is None:
                continue
            special_token = getattr(tokenizer, token)
            if special_token in special_tokens_map:
                new_special_token = getattr(new_tokenizer, token)
                self.assertEqual(special_tokens_map[special_token], new_special_token)

                new_id = new_tokenizer.get_vocab()[new_special_token]
                self.assertEqual(getattr(new_tokenizer, f"{token}_id"), new_id)

        # Check if the AddedToken / string format has been kept
        for special_token in tokenizer.all_special_tokens_extended:
            if isinstance(special_token, AddedToken) and special_token.content not in special_tokens_map:
                # The special token must appear identically in the list of the new tokenizer.
                self.assertTrue(
                    special_token in new_tokenizer.all_special_tokens_extended,
                    f"'{special_token}' should be in {new_tokenizer.all_special_tokens_extended}",
                )
            elif isinstance(special_token, AddedToken):
                # The special token must appear in the list of the new tokenizer as an object of type AddedToken with
                # the same parameters as the old AddedToken except the content that the user has requested to change.
                special_token_str = special_token.content
                new_special_token_str = special_tokens_map[special_token_str]

                find = False
                for candidate in new_tokenizer.all_special_tokens_extended:
                    if (
                        isinstance(candidate, AddedToken)
                        and candidate.content == new_special_token_str
                        and candidate.lstrip == special_token.lstrip
                        and candidate.rstrip == special_token.rstrip
                        and candidate.normalized == special_token.normalized
                        and candidate.single_word == special_token.single_word
                    ):
                        find = True
                        break
                self.assertTrue(
                    find,
                    (
                        f"'{new_special_token_str}' doesn't appear in the list "
                        f"'{new_tokenizer.all_special_tokens_extended}' as an AddedToken with the same parameters as "
                        f"'{special_token}' in the list {tokenizer.all_special_tokens_extended}"
                    ),
                )
            elif special_token not in special_tokens_map:
                # The special token must appear identically in the list of the new tokenizer.
                self.assertTrue(
                    special_token in new_tokenizer.all_special_tokens_extended,
                    f"'{special_token}' should be in {new_tokenizer.all_special_tokens_extended}",
                )

            else:
                # The special token must appear in the list of the new tokenizer as an object of type string.
                self.assertTrue(special_tokens_map[special_token] in new_tokenizer.all_special_tokens_extended)

        # Test we can use the new tokenizer with something not seen during training
        inputs = new_tokenizer(["This is the first sentence", "This sentence is different 🤗."])
        self.assertEqual(len(inputs["input_ids"]), 2)
        decoded_input = new_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        expected_result = "This is the first sentence"

        if tokenizer.backend_tokenizer.normalizer is not None:
            expected_result = tokenizer.backend_tokenizer.normalizer.normalize_str(expected_result)
        self.assertEqual(expected_result, decoded_input)

    def test_tokenizer_mismatch_warning(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                with self.assertLogs("transformers", level="WARNING") as cm:
                    try:
                        if self.tokenizer_class == BertTokenizer:
                            AlbertTokenizer.from_pretrained(pretrained_name)
                        else:
                            BertTokenizer.from_pretrained(pretrained_name)
                    except (TypeError, AttributeError):
                        # Some tokenizers cannot be loaded into the target tokenizer at all and errors are returned,
                        # here we just check that the warning has been logged before the error is raised
                        pass
                    finally:
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "The tokenizer class you load from this checkpoint is not the same type as the class this function is called from."
                            )
                        )
                    try:
                        if self.rust_tokenizer_class == BertTokenizerFast:
                            AlbertTokenizerFast.from_pretrained(pretrained_name)
                        else:
                            BertTokenizerFast.from_pretrained(pretrained_name)
                    except (TypeError, AttributeError):
                        # Some tokenizers cannot be loaded into the target tokenizer at all and errors are returned,
                        # here we just check that the warning has been logged before the error is raised
                        pass
                    finally:
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "The tokenizer class you load from this checkpoint is not the same type as the class this function is called from."
                            )
                        )

    @require_torch
    def test_saving_tokenizer_trainer(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Save the fast tokenizer files in a temporary directory
                    tokenizer_old = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs, use_fast=True)
                    tokenizer_old.save_pretrained(tmp_dir, legacy_format=False)  # save only fast version

                    # Initialize toy model for the trainer
                    model = nn.Module()

                    # Load tokenizer from a folder without legacy files
                    tokenizer = self.rust_tokenizer_class.from_pretrained(tmp_dir)
                    training_args = TrainingArguments(output_dir=tmp_dir, do_train=True, no_cuda=True)
                    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)

                    # Should not raise an error
                    trainer.save_model(os.path.join(tmp_dir, "checkpoint"))
                    self.assertIn("tokenizer.json", os.listdir(os.path.join(tmp_dir, "checkpoint")))


@is_staging_test
class TokenizerPushToHubTester(unittest.TestCase):
    vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "bla", "blou"]

    @classmethod
    def setUpClass(cls):
        cls._api = HfApi(endpoint=ENDPOINT_STAGING)
        cls._token = cls._api.login(username=USER, password=PASS)

    @classmethod
    def tearDownClass(cls):
        try:
            cls._api.delete_repo(token=cls._token, name="test-tokenizer")
        except HTTPError:
            pass

        try:
            cls._api.delete_repo(token=cls._token, name="test-tokenizer-org", organization="valid_org")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_file = os.path.join(tmp_dir, "vocab.txt")
            with open(vocab_file, "w", encoding="utf-8") as vocab_writer:
                vocab_writer.write("".join([x + "\n" for x in self.vocab_tokens]))
            tokenizer = BertTokenizer(vocab_file)
            tokenizer.save_pretrained(
                os.path.join(tmp_dir, "test-tokenizer"), push_to_hub=True, use_auth_token=self._token
            )

            new_tokenizer = BertTokenizer.from_pretrained(f"{USER}/test-tokenizer")
            self.assertDictEqual(new_tokenizer.vocab, tokenizer.vocab)

    def test_push_to_hub_in_organization(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            vocab_file = os.path.join(tmp_dir, "vocab.txt")
            with open(vocab_file, "w", encoding="utf-8") as vocab_writer:
                vocab_writer.write("".join([x + "\n" for x in self.vocab_tokens]))
            tokenizer = BertTokenizer(vocab_file)
            tokenizer.save_pretrained(
                os.path.join(tmp_dir, "test-tokenizer-org"),
                push_to_hub=True,
                use_auth_token=self._token,
                organization="valid_org",
            )

            new_tokenizer = BertTokenizer.from_pretrained("valid_org/test-tokenizer-org")
            self.assertDictEqual(new_tokenizer.vocab, tokenizer.vocab)


class TrieTest(unittest.TestCase):
    def test_trie(self):
        trie = Trie()
        trie.add("Hello 友達")
        self.assertEqual(trie.data, {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}})
        trie.add("Hello")
        trie.data
        self.assertEqual(trie.data, {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}})

    def test_trie_split(self):
        trie = Trie()
        self.assertEqual(trie.split("[CLS] This is a extra_id_100"), ["[CLS] This is a extra_id_100"])
        trie.add("[CLS]")
        trie.add("extra_id_1")
        trie.add("extra_id_100")
        self.assertEqual(trie.split("[CLS] This is a extra_id_100"), ["[CLS]", " This is a ", "extra_id_100"])
