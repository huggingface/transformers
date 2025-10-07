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

import copy
import functools
import inspect
import itertools
import json
import os
import pickle
import re
import shutil
import tempfile
import traceback
import unittest
from collections import OrderedDict
from itertools import takewhile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from parameterized import parameterized

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
    is_mlx_available,
    is_torch_available,
    logging,
)
from transformers.testing_utils import (
    check_json_file_has_correct_format,
    get_tests_dir,
    require_jinja,
    require_read_token,
    require_tokenizers,
    require_torch,
    run_test_in_subprocess,
    slow,
)
from transformers.tokenization_utils import AddedToken

from .test_tokenizers_backend_mixin import TokenizersBackendTesterMixin
from .test_sentencepiece_backend_mixin import SentencePieceBackendTesterMixin


if is_torch_available():
    import torch
    import torch.nn as nn


if TYPE_CHECKING:
    from transformers import PreTrainedConfig, PreTrainedModel


def use_cache_if_possible(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        use_cache = kwargs.pop("use_cache", True)

        underline_func = func
        if "functools" in str(func):
            underline_func = func.__wrapped__

        if not use_cache:
            return underline_func(*args, **kwargs)
        if any(not arg.__hash__ for arg in args):
            return underline_func(*args, **kwargs)
        elif any(not kwarg.__hash__ for kwarg in kwargs.values()):
            return underline_func(*args, **kwargs)

        cached = func(*args, **kwargs)
        copied = copy.deepcopy(cached)

        if hasattr(copied, "_tokenizer") and "tests.models.clip.test_tokenization_clip.CLIPTokenizationTest" in str(
            args[0]
        ):
            copied._tokenizer = cached._tokenizer

        if hasattr(copied, "sp_model"):
            copied.sp_model = cached.sp_model

        return copied

    return wrapper


logger = logging.get_logger(__name__)

NON_ENGLISH_TAGS = ["chinese", "dutch", "french", "finnish", "german", "multilingual"]


def filter_non_english(_, pretrained_name: str):
    """Filter all the model for non-english language"""
    return not any(lang in pretrained_name for lang in NON_ENGLISH_TAGS)


def filter_roberta_detectors(_, pretrained_name: str):
    return "detector" not in pretrained_name


def merge_model_tokenizer_mappings(
    model_mapping: dict["PreTrainedConfig", "PreTrainedModel"],
    tokenizer_mapping: dict["PreTrainedConfig", tuple["PreTrainedTokenizer", "PreTrainedTokenizerFast"]],
) -> dict[
    Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
    tuple["PreTrainedConfig", "PreTrainedModel"],
]:
    configurations = list(model_mapping.keys())
    model_tokenizer_mapping = OrderedDict([])

    for configuration in configurations:
        if configuration in model_mapping and configuration in tokenizer_mapping:
            model = model_mapping[configuration]
            tokenizer = tokenizer_mapping[configuration][0]
            tokenizer_fast = tokenizer_mapping[configuration][1]

            if tokenizer is not None:
                if configuration.__name__.startswith(tokenizer.__name__.replace("Tokenizer", "")):
                    model_tokenizer_mapping.update({tokenizer: (configuration, model)})
            if tokenizer_fast is not None:
                if configuration.__name__.startswith(tokenizer_fast.__name__.replace("TokenizerFast", "")):
                    model_tokenizer_mapping.update({tokenizer_fast: (configuration, model)})

    return model_tokenizer_mapping

def check_subword_sampling(
    tokenizer: PreTrainedTokenizer,
    text: Optional[str] = None,
    test_sentencepiece_ignore_case: bool = True,
) -> None:
    """
    Check if the tokenizer generates different results when subword regularization is enabled.

    Subword regularization augments training data with subword sampling.
    This has a random component.

    Args:
        tokenizer: The tokenizer to check.
        text: The text to use for the checks.
        test_sentencepiece_ignore_case: See `TokenizerTesterMixin.test_sentencepiece_ignore_case`.
    """
    text = "This is a test for subword regularization." if text is None else text
    if test_sentencepiece_ignore_case:
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
    unittest.TestCase().assertTrue(subword_sampling_found)

    # check if converting back to original text works
    for tokens in tokens_list:
        if test_sentencepiece_ignore_case:
            unittest.TestCase().assertEqual(text, tokenizer.convert_tokens_to_string(tokens).lower())
        else:
            unittest.TestCase().assertEqual(text, tokenizer.convert_tokens_to_string(tokens))


class TokenizerTesterMixin:
    tokenizer_class = None
    rust_tokenizer_class = None
    test_slow_tokenizer = True
    test_rust_tokenizer = True
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    from_pretrained_filter = None
    from_pretrained_id = None
    from_pretrained_vocab_key = "vocab_file"
    test_seq2seq = True

    # set to True to test a sentencepiece tokenizer
    test_sentencepiece = False

    # set to True to ignore casing when testing a sentencepiece tokenizer
    # test_sentencepiece must also be set to True
    test_sentencepiece_ignore_case = False

    @classmethod
    def setUpClass(cls) -> None:
        # Tokenizer.filter makes it possible to filter which Tokenizer to case based on all the
        # information available in Tokenizer (name, rust class, python class, vocab key name)
        cls.from_pretrained_id = (
            [cls.from_pretrained_id] if isinstance(cls.from_pretrained_id, str) else cls.from_pretrained_id
        )

        cls.tokenizers_list = []
        if cls.test_rust_tokenizer:
            cls.tokenizers_list = [
                (
                    cls.rust_tokenizer_class,
                    pretrained_id,
                    cls.from_pretrained_kwargs if cls.from_pretrained_kwargs is not None else {},
                )
                for pretrained_id in cls.from_pretrained_id
            ]
        else:
            cls.tokenizers_list = []
        with open(f"{get_tests_dir()}/fixtures/sample_text.txt", encoding="utf-8") as f_data:
            cls._data = f_data.read().replace("\n\n", "\n").strip()

        cls.tmpdirname = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def get_input_output_texts(self, tokenizer):
        input_txt = self.get_clean_sequence(tokenizer)[0]
        return input_txt, input_txt

    def get_clean_sequence(self, tokenizer, with_prefix_space=False, max_length=20, min_length=5) -> tuple[str, list]:
        # the length of the tokenizer does not always represent the tokens that it can encode: what if there are holes?
        toks = [
            (i, tokenizer.decode([i], clean_up_tokenization_spaces=False)) for i in set(tokenizer.get_vocab().values())
        ]
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

    def get_tokenizers(self, fast=True, **kwargs) -> list[PreTrainedTokenizerBase]:
        if fast and self.test_rust_tokenizer and self.test_slow_tokenizer:
            return [self.get_tokenizer(**kwargs), self.get_rust_tokenizer(**kwargs)]
        elif fast and self.test_rust_tokenizer:
            return [self.get_rust_tokenizer(**kwargs)]
        elif self.test_slow_tokenizer:
            return [self.get_tokenizer(**kwargs)]
        else:
            raise ValueError("This tokenizer class has no tokenizer to be tested.")

    @classmethod
    def get_tokenizer(cls, pretrained_name=None, **kwargs) -> PreTrainedTokenizer:
        pretrained_name = pretrained_name or cls.tmpdirname
        return cls.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    @classmethod
    def get_rust_tokenizer(cls, pretrained_name=None, **kwargs) -> PreTrainedTokenizerFast:
        pretrained_name = pretrained_name or cls.tmpdirname
        return cls.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    def tokenizer_integration_test_util(
        self,
        expected_encoding: dict,
        model_name: str,
        revision: Optional[str] = None,
        sequences: Optional[list[str]] = None,
        decode_kwargs: Optional[dict[str, Any]] = None,
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
            (
                self.assertEqual(len(i_r), 2),
                self.assertEqual(len(i_r[0]), max_length),
                self.assertEqual(len(i_r[1]), max_length),
            )
            (
                self.assertEqual(len(i_r), 2),
                self.assertEqual(len(i_r[0]), max_length),
                self.assertEqual(len(i_r[1]), max_length),
            )

        for i_r, i_p in zip(input_r[model_main_input_name], input_p[model_main_input_name]):
            self.assert_padded_input_match(i_r, i_p, max_length, pad_token_id)

        for i_r, i_p in zip(input_r["attention_mask"], input_p["attention_mask"]):
            self.assertSequenceEqual(i_r, i_p)

    @staticmethod
    def convert_batch_to_list_format(batch_encode_plus_sequences):
        # Switch from batch_encode_plus format:   {'input_ids': [[...], [...]], ...}
        # to the list of examples/ encode_plus format: [{'input_ids': [...], ...}, {'input_ids': [...], ...}]
        return [
            {value: batch_encode_plus_sequences[value][i] for value in batch_encode_plus_sequences}
            for i in range(len(batch_encode_plus_sequences["input_ids"]))
        ]

    # TODO: this test can be combined with `test_sentencepiece_tokenize_and_convert_tokens_to_string` after the latter is extended to all tokenizers.
    def test_tokenize_special_tokens(self):
        """Test `tokenize` with special tokens."""
        tokenizers = self.get_tokenizers(fast=True, do_lower_case=True)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                SPECIAL_TOKEN_1 = "[SPECIAL_TOKEN_1]"
                SPECIAL_TOKEN_2 = "[SPECIAL_TOKEN_2]"

                # Both methods should add the token to `_additional_special_tokens` and `added_tokens_decoder`
                tokenizer.add_tokens([SPECIAL_TOKEN_1], special_tokens=True)
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": [SPECIAL_TOKEN_2]}, replace_additional_special_tokens=False
                )

                token_1 = tokenizer.tokenize(SPECIAL_TOKEN_1)
                token_2 = tokenizer.tokenize(SPECIAL_TOKEN_2)

                self.assertEqual(len(token_1), 1)
                self.assertEqual(len(token_2), 1)
                self.assertEqual(token_1[0], SPECIAL_TOKEN_1)
                # next is failing for almost all the Fast tokenizers now.
                # self.assertEqual(token_2[0], SPECIAL_TOKEN_2)

    # TODO: this test could be extended to all tokenizers - not just the sentencepiece
    def test_sentencepiece_tokenize_and_convert_tokens_to_string(self):
        self.skipTest(
            reason="Moved to SentencePieceBackendTesterMixin"
        )

    def test_sentencepiece_tokenize_and_decode(self):
        self.skipTest(
            reason="Moved to SentencePieceBackendTesterMixin"
        )

    def test_save_sentencepiece_tokenizer(self) -> None:
        self.skipTest(
            reason="Moved to SentencePieceBackendTesterMixin"
        )

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
            self.skipTest(reason="test_rust_tokenizer is set to False")

        signature = inspect.signature(self.rust_tokenizer_class.__init__)

        self.assertIn("tokenizer_file", signature.parameters)
        self.assertIsNone(signature.parameters["tokenizer_file"].default)

    def test_tokenizer_slow_store_full_signature(self):
        self.skipTest(reason="Test removed for v5, slow only")

    def test_tokenizer_fast_store_full_signature(self):
        if not self.test_rust_tokenizer:
            self.skipTest(reason="test_rust_tokenizer is set to False")

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
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

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

    def test_tokenizers_common_ids_setters(self):
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

                vocab = tokenizer.get_vocab()
                token_id_to_test_setters = next(iter(vocab.values()))
                token_to_test_setters = tokenizer.convert_ids_to_tokens(
                    token_id_to_test_setters, skip_special_tokens=False
                )

                for attr in attributes_list:
                    setattr(tokenizer, attr + "_id", None)
                    self.assertEqual(getattr(tokenizer, attr), None)
                    self.assertEqual(getattr(tokenizer, attr + "_id"), None)

                    setattr(tokenizer, attr + "_id", token_id_to_test_setters)
                    self.assertEqual(getattr(tokenizer, attr), token_to_test_setters)
                    self.assertEqual(getattr(tokenizer, attr + "_id"), token_id_to_test_setters)

                setattr(tokenizer, "additional_special_tokens_ids", [])
                self.assertListEqual(getattr(tokenizer, "additional_special_tokens"), [])
                self.assertListEqual(getattr(tokenizer, "additional_special_tokens_ids"), [])

                setattr(tokenizer, "additional_special_tokens_ids", [token_id_to_test_setters])
                self.assertListEqual(getattr(tokenizer, "additional_special_tokens"), [token_to_test_setters])
                self.assertListEqual(getattr(tokenizer, "additional_special_tokens_ids"), [token_id_to_test_setters])

    @parameterized.expand([(True,), (False,)])
    def test_tokenizers_special_tokens_properties_unset(self, verbose):
        self.skipTest(reason="Test removed for v5, test_tokenizers_common_ids_setters already tests this")

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

                sample_text = " He is very happy, UNwant\u00e9d,running"
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

                sample_text = " He is very happy, UNwant\u00e9d,running"
                tokenizer.add_tokens(["bim", "bambam"])
                additional_special_tokens = tokenizer.additional_special_tokens
                additional_special_tokens.append("new_additional_special_token")
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": additional_special_tokens}, replace_additional_special_tokens=False
                )
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

                sample_text = " He is very happy, UNwant\u00e9d,running"
                tokenizer.add_tokens(["bim", "bambam"])
                additional_special_tokens = tokenizer.additional_special_tokens
                additional_special_tokens.append("new_additional_special_token")
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": additional_special_tokens}, replace_additional_special_tokens=False
                )
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
        self.skipTest(
            reason="Moved to SentencePieceBackendTesterMixin, tokenizers should test this, not transformers"
        )

    def test_added_tokens_do_lower_case(self):
        self.skipTest(
            reason="Moved to SentencePieceBackendTesterMixin, tokenizers should test this, not transformers"
                )

    # TODO @ArthurZ Nuke this
    def test_add_tokens_tokenizer(self):
        self.skipTest(
            reason="Moved to SentencePieceBackendTesterMixin, tokenizers should test this, not transformers"
        )

    def test_add_special_tokens(self):
        self.skipTest(
            reason="Moved to SentencePieceBackendTesterMixin, tokenizers should test this, not transformers"
        )

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
        self.skipTest(reason="Test removed for v5, only tests slow tokenizer, and covered by integration tests.")

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

    @require_jinja
    def test_chat_template(self):
        dummy_template = "{% for message in messages %}{{message['role'] + message['content']}}{% endfor %}"
        dummy_conversation = [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "user message"},
            {"role": "assistant", "content": "assistant message"},
        ]
        expected_output = "systemsystem messageuseruser messageassistantassistant message"
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                output = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template=dummy_template, tokenize=False, return_dict=False
                )
                self.assertEqual(output, expected_output)  # Test we can pass chat_template arg

                # Check that no error raised when tokenize=True
                output = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template=dummy_template, tokenize=True, return_dict=False
                )
                dict_output = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template=dummy_template, tokenize=True, return_dict=True
                )
                self.assertEqual(dict_output["input_ids"], output)  # Test return_dict behaviour matches

                tokenizer.chat_template = dummy_template
                self.assertEqual(tokenizer.chat_template, dummy_template)  # Test property setter
                output = tokenizer.apply_chat_template(dummy_conversation, tokenize=False, return_dict=False)
                self.assertEqual(output, expected_output)  # Test chat_template attribute is used if no arg is passed
                # Check that no error raised
                tokenizer.apply_chat_template(dummy_conversation, tokenize=True, return_dict=False)

                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    save_files = tokenizer.save_pretrained(tmp_dir_name, save_jinja_files=False)
                    # Check we aren't saving a chat_template.jinja file
                    self.assertFalse(any(file.endswith("chat_template.jinja") for file in save_files))
                    new_tokenizer = tokenizer.from_pretrained(tmp_dir_name)

                self.assertEqual(new_tokenizer.chat_template, dummy_template)  # Test template has persisted
                output = new_tokenizer.apply_chat_template(dummy_conversation, tokenize=False, return_dict=False)
                self.assertEqual(output, expected_output)  # Test output is the same after reloading
                # Check that no error raised
                new_tokenizer.apply_chat_template(dummy_conversation, tokenize=True, return_dict=False)

                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    save_files = tokenizer.save_pretrained(tmp_dir_name)
                    # Check we are saving a chat_template.jinja file
                    self.assertTrue(any(file.endswith("chat_template.jinja") for file in save_files))
                    chat_template_file = Path(tmp_dir_name) / "chat_template.jinja"
                    self.assertTrue(chat_template_file.is_file())
                    self.assertEqual(chat_template_file.read_text(), dummy_template)
                    config_dict = json.loads((Path(tmp_dir_name) / "tokenizer_config.json").read_text())
                    # Assert the chat template is not in the config when it's saved as a separate file
                    self.assertNotIn("chat_template", config_dict)
                    new_tokenizer = tokenizer.from_pretrained(tmp_dir_name)

                self.assertEqual(new_tokenizer.chat_template, dummy_template)  # Test template has persisted
                output = new_tokenizer.apply_chat_template(dummy_conversation, tokenize=False, return_dict=False)
                self.assertEqual(output, expected_output)  # Test output is the same after reloading
                # Check that no error raised
                new_tokenizer.apply_chat_template(dummy_conversation, tokenize=True, return_dict=False)

    @require_jinja
    def test_chat_template_save_loading(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            signature = inspect.signature(tokenizer.__init__)
            if "chat_template" not in {*signature.parameters.keys()}:
                self.skipTest("tokenizer doesn't accept chat templates at input")
            tokenizer.chat_template = "test template"
            with tempfile.TemporaryDirectory() as tmpdirname:
                tokenizer.save_pretrained(tmpdirname)
                self.assertTrue(Path(tmpdirname, "chat_template.jinja").is_file())
                self.assertFalse(Path(tmpdirname, "chat_template.json").is_file())
                self.assertFalse(Path(tmpdirname, "additional_chat_templates").is_dir())
                reloaded_tokenizer = self.tokenizer_class.from_pretrained(tmpdirname)
                self.assertEqual(tokenizer.chat_template, reloaded_tokenizer.chat_template)
                # When we save as single files, tokenizers and tokenizers share a chat template, which means
                # the reloaded tokenizer should get the chat template as well
                self.assertEqual(reloaded_tokenizer.chat_template, reloaded_tokenizer.tokenizer.chat_template)

            with tempfile.TemporaryDirectory() as tmpdirname:
                tokenizer.chat_template = {"default": "a", "secondary": "b"}
                tokenizer.save_pretrained(tmpdirname)
                self.assertTrue(Path(tmpdirname, "chat_template.jinja").is_file())
                self.assertFalse(Path(tmpdirname, "chat_template.json").is_file())
                self.assertTrue(Path(tmpdirname, "additional_chat_templates").is_dir())
                reloaded_tokenizer = self.tokenizer_class.from_pretrained(tmpdirname)
                self.assertEqual(tokenizer.chat_template, reloaded_tokenizer.chat_template)
                # When we save as single files, tokenizers and tokenizers share a chat template, which means
                # the reloaded tokenizer should get the chat template as well
                self.assertEqual(reloaded_tokenizer.chat_template, reloaded_tokenizer.tokenizer.chat_template)

            with tempfile.TemporaryDirectory() as tmpdirname:
                tokenizer.chat_template = {"default": "a", "secondary": "b"}
                tokenizer.save_pretrained(tmpdirname, save_jinja_files=False)
                self.assertFalse(Path(tmpdirname, "chat_template.jinja").is_file())
                self.assertFalse(Path(tmpdirname, "chat_template.json").is_file())
                self.assertFalse(Path(tmpdirname, "additional_chat_templates").is_dir())
                reloaded_tokenizer = self.tokenizer_class.from_pretrained(tmpdirname)
                self.assertEqual(tokenizer.chat_template, reloaded_tokenizer.chat_template)
                # When we save as single files, tokenizers and tokenizers share a chat template, which means
                # the reloaded tokenizer should get the chat template as well
                self.assertEqual(reloaded_tokenizer.chat_template, reloaded_tokenizer.tokenizer.chat_template)

    @require_jinja
    def test_chat_template_batched(self):
        dummy_template = "{% for message in messages %}{{message['role'] + message['content']}}{% endfor %}"
        dummy_conversations = [
            [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "user message"},
                {"role": "assistant", "content": "assistant message"},
            ],
            [
                {"role": "system", "content": "system message 2"},
                {"role": "user", "content": "user message 2"},
                {"role": "assistant", "content": "assistant message 2"},
            ],
        ]
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                output = tokenizer.apply_chat_template(
                    dummy_conversations, chat_template=dummy_template, tokenize=False
                )
                self.assertEqual(
                    output,
                    [
                        "systemsystem messageuseruser messageassistantassistant message",
                        "systemsystem message 2useruser message 2assistantassistant message 2",
                    ],
                )
                one_element_output = tokenizer.apply_chat_template(
                    dummy_conversations[:1], chat_template=dummy_template, tokenize=False
                )
                self.assertEqual(
                    one_element_output, ["systemsystem messageuseruser messageassistantassistant message"]
                )  # Assert that list structure is retained even with one element
                tokenizer.apply_chat_template(
                    dummy_conversations, chat_template=dummy_template, tokenize=True
                )  # Check that no error raised

    @require_jinja
    def test_jinja_loopcontrols(self):
        break_template = """
        {%- for message in messages %}
            {{- message.role + " " + message.content }}
            {%- if loop.first %}
                {%- break %}
            {%- endif %}
        {%- endfor %}""".strip()

        dummy_conversation = [
            {"role": "system", "content": "1"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "3"},
        ]

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                break_output = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template=break_template, tokenize=False
                )
                self.assertEqual(break_output, "system 1")  # Loop should break after first iter

    @require_jinja
    def test_jinja_strftime(self):
        strftime_template = """{{- strftime_now("%Y-%m-%d") }}""".strip()

        dummy_conversation = [
            {"role": "system", "content": "1"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "3"},
        ]

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                strftime_output = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template=strftime_template, tokenize=False
                )

                # Assert that we get a date formatted as expected
                self.assertEqual(len(strftime_output), 10)
                self.assertEqual(len(strftime_output.split("-")), 3)

    @require_torch
    @require_jinja
    def test_chat_template_return_assistant_tokens_mask(self):
        dummy_template = (
            "{% for message in messages %}"
            "{% if (message['role'] != 'assistant') %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% elif (message['role'] == 'assistant')%}"
            "{{'<|im_start|>' + message['role'] + '\n'}}"
            "{% generation %}"
            "{{message['content'] + '<|im_end|>'}}"
            "{% endgeneration %}"
            "{{'\n'}}"
            "{% endif %}"
            "{% endfor %}"
        )
        conversations = [
            [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "user message"},
                {"role": "assistant", "content": "start turn 1 assistant message. end turn 1"},
                {"role": "user", "content": "user message 2"},
                {"role": "assistant", "content": "start turn 2 assistant message. end turn 2"},
            ],
            [
                {"role": "system", "content": "system message 3"},
                {"role": "user", "content": "user message 3"},
                {"role": "assistant", "content": "start turn 3 assistant message. end turn 3"},
                {"role": "user", "content": "user message 4"},
                {"role": "assistant", "content": "start turn 4 assistant message. end turn 4"},
            ],
        ]

        # These are the prefix and suffix strings of all the assistant messages. Used to find the assistant substring
        # in the entire chat string, and then find the corresponding tokens in the tokenized output.
        assistant_prefix_suffix = [
            [("start turn 1", "end turn 1<|im_end|>"), ("start turn 2", "end turn 2<|im_end|>")],
            [("start turn 3", "end turn 3<|im_end|>"), ("start turn 4", "end turn 4<|im_end|>")],
        ]
        for tokenizer, pretrained_name, _ in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                if not self.test_rust_tokenizer:
                    self.skipTest(reason="No fast tokenizer defined")

                tokenizer_r = self.get_rust_tokenizer(pretrained_name)
                self._check_no_pad_token_padding(tokenizer_r, conversations)

                tokenizer_r.padding_side = "right"

                # check batched
                output = tokenizer_r.apply_chat_template(
                    conversations,
                    chat_template=dummy_template,
                    tokenize=True,
                    return_assistant_tokens_mask=True,
                    return_dict=True,
                )

                output_pt = tokenizer_r.apply_chat_template(
                    conversations,
                    chat_template=dummy_template,
                    tokenize=True,
                    padding=True,
                    return_assistant_tokens_mask=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                self.assertEqual(type(output_pt["assistant_masks"]), torch.Tensor)
                self.assertEqual(output_pt["assistant_masks"].shape, output_pt["input_ids"].shape)

                for i, conv in enumerate(conversations):
                    chat_string = tokenizer_r.apply_chat_template(conv, tokenize=False, chat_template=dummy_template)
                    assistant_start = output.char_to_token(i, chat_string.index(assistant_prefix_suffix[i][0][0]))
                    assistant_end = output.char_to_token(
                        i,
                        chat_string.index(assistant_prefix_suffix[i][0][1])
                        + len(assistant_prefix_suffix[i][0][1])
                        - 1,
                    )

                    assistant_start2 = output.char_to_token(i, chat_string.index(assistant_prefix_suffix[i][1][0]))
                    assistant_end2 = output.char_to_token(
                        i,
                        chat_string.index(assistant_prefix_suffix[i][1][1])
                        + len(assistant_prefix_suffix[i][1][1])
                        - 1,
                    )

                    # assert 1 in first assistant message
                    self.assertEqual(
                        output["assistant_masks"][i][assistant_start : assistant_end + 1],
                        [1] * (assistant_end - assistant_start + 1),
                    )
                    self.assertTrue(
                        (output_pt["assistant_masks"][i, assistant_start : assistant_end + 1] == 1).all(),
                    )

                    # assert 1 second assistant message
                    self.assertEqual(
                        output["assistant_masks"][i][assistant_start2 : assistant_end2 + 1],
                        [1] * (assistant_end2 - assistant_start2 + 1),
                    )
                    self.assertTrue(
                        (output_pt["assistant_masks"][i, assistant_start2 : assistant_end2 + 1] == 1).all(),
                    )

                    # assert 0 in user/system indices
                    self.assertEqual(output["assistant_masks"][i][:assistant_start], [0] * assistant_start)
                    self.assertTrue((output_pt["assistant_masks"][i, :assistant_start] == 0).all())

                    self.assertEqual(
                        output["assistant_masks"][i][assistant_end + 1 : assistant_start2],
                        [0] * (assistant_start2 - assistant_end - 1),
                    )
                    self.assertTrue(
                        (output_pt["assistant_masks"][i, assistant_end + 1 : assistant_start2] == 0).all(),
                    )

                # check not batched
                output = tokenizer_r.apply_chat_template(
                    conversations[0],
                    chat_template=dummy_template,
                    tokenize=True,
                    return_assistant_tokens_mask=True,
                    return_dict=True,
                )
                output_pt = tokenizer_r.apply_chat_template(
                    conversations[0],
                    chat_template=dummy_template,
                    tokenize=True,
                    return_assistant_tokens_mask=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                self.assertEqual(type(output_pt["assistant_masks"]), torch.Tensor)
                self.assertEqual(output_pt["assistant_masks"].shape, output_pt["input_ids"].shape)

                chat_string = tokenizer_r.apply_chat_template(
                    conversations[0], tokenize=False, chat_template=dummy_template
                )
                assistant_start = output.char_to_token(0, chat_string.index(assistant_prefix_suffix[0][0][0]))
                assistant_end = output.char_to_token(
                    0, chat_string.index(assistant_prefix_suffix[0][0][1]) + len(assistant_prefix_suffix[0][0][1]) - 1
                )
                assistant_start2 = output.char_to_token(0, chat_string.index(assistant_prefix_suffix[0][1][0]))
                assistant_end2 = output.char_to_token(
                    0, chat_string.index(assistant_prefix_suffix[0][1][1]) + len(assistant_prefix_suffix[0][1][1]) - 1
                )

                # assert 1 in assistant indices
                self.assertEqual(
                    output["assistant_masks"][assistant_start : assistant_end + 1],
                    [1] * (assistant_end - assistant_start + 1),
                )
                self.assertTrue(
                    (output_pt["assistant_masks"][assistant_start : assistant_end + 1] == 1).all(),
                )
                self.assertEqual(
                    output["assistant_masks"][assistant_start2 : assistant_end2 + 1],
                    [1] * (assistant_end2 - assistant_start2 + 1),
                )
                self.assertTrue(
                    (output_pt["assistant_masks"][assistant_start2 : assistant_end2 + 1] == 1).all(),
                )

                # assert 0 in user/system indices
                self.assertEqual(output["assistant_masks"][:assistant_start], [0] * assistant_start)
                self.assertTrue((output_pt["assistant_masks"][0, :assistant_start] == 0).all())
                self.assertEqual(
                    output["assistant_masks"][assistant_end + 1 : assistant_start2],
                    [0] * (assistant_start2 - assistant_end - 1),
                )
                self.assertTrue(
                    (output_pt["assistant_masks"][0, assistant_end + 1 : assistant_start2] == 0).all(),
                )

    @require_jinja
    def test_chat_template_return_assistant_tokens_mask_truncated(self):
        dummy_template = (
            "{% for message in messages %}"
            "{% if (message['role'] != 'assistant') %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% elif (message['role'] == 'assistant')%}"
            "{{'<|im_start|>' + message['role'] + '\n'}}"
            "{% generation %}"
            "{{message['content'] + '<|im_end|>'}}"
            "{% endgeneration %}"
            "{{'\n'}}"
            "{% endif %}"
            "{% endfor %}"
        )
        conversations = [
            [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "user message"},
                {
                    "role": "assistant",
                    "content": (
                        "start turn assistant. long string to be truncated, long string to be truncated, "
                        "long string to be truncated, long string to be truncated, long string to be truncated"
                    ),
                },
                {"role": "user", "content": "another user message"},
            ],
            [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "user message"},
                {
                    "role": "assistant",
                    "content": (
                        "start turn assistant. long string to be truncated, long string to be truncated, "
                        "long string to be truncated, long string to be truncated, long string to be truncated"
                    ),
                },
                {"role": "user", "content": "another user message"},
            ],
        ]

        for tokenizer, pretrained_name, _ in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                if not self.test_rust_tokenizer:
                    self.skipTest(reason="No fast tokenizer defined")

                tokenizer_r = self.get_rust_tokenizer(pretrained_name)

                # Find where to truncate, as the amount of tokens is different for different tokenizers and I want the
                # truncation to happen in the middle of the assistant content.
                full_encoding = tokenizer_r.apply_chat_template(
                    conversations[0],
                    chat_template=dummy_template,
                    tokenize=True,
                    return_dict=True,
                )
                chat_string = tokenizer_r.apply_chat_template(
                    conversations[0], tokenize=False, chat_template=dummy_template
                )
                truncation_position = full_encoding.char_to_token(chat_string.index(", long string to be truncated,"))

                # check batched
                output = tokenizer_r.apply_chat_template(
                    conversations,
                    chat_template=dummy_template,
                    tokenize=True,
                    return_assistant_tokens_mask=True,
                    max_length=truncation_position,
                    truncation=True,
                    return_dict=True,
                )
                for i, conv in enumerate(conversations):
                    chat_string = tokenizer_r.apply_chat_template(conv, tokenize=False, chat_template=dummy_template)
                    assistant_start = output.char_to_token(i, chat_string.index("start turn assistant"))

                    # assert 1 from assistant_start to the end because the rest is truncated.
                    self.assertEqual(
                        output["assistant_masks"][i][assistant_start:],
                        [1] * (len(output["assistant_masks"][i]) - assistant_start),
                    )

                # check not batched
                output = tokenizer_r.apply_chat_template(
                    conversations[0],
                    chat_template=dummy_template,
                    tokenize=True,
                    return_assistant_tokens_mask=True,
                    return_dict=True,
                    max_length=truncation_position,
                    truncation=True,
                )

                chat_string = tokenizer_r.apply_chat_template(
                    conversations[0], tokenize=False, chat_template=dummy_template
                )
                assistant_start = output.char_to_token(0, chat_string.index("start turn assistant"))

                # assert 1 from assistant_start to the end because the rest is truncated.
                self.assertEqual(
                    output["assistant_masks"][assistant_start:],
                    [1] * (len(output["assistant_masks"]) - assistant_start),
                )

    @require_jinja
    def test_continue_final_message(self):
        dummy_template = """
        {%- for message in messages %}
            {{- "<|im_start|>" + message['role'] + "\n" + message['content'] + "<|im_end|>" + "\n"}}
        {%- endfor %}"""
        dummy_conversation = [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "user message"},
            {"role": "assistant", "content": "assistant message"},
        ]
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                output = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template=dummy_template, tokenize=False, continue_final_message=False
                )
                self.assertEqual(
                    output,
                    "<|im_start|>system\nsystem message<|im_end|>\n<|im_start|>user\nuser message<|im_end|>\n<|im_start|>assistant\nassistant message<|im_end|>\n",
                )
                prefill_output = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template=dummy_template, tokenize=False, continue_final_message=True
                )
                # Assert that the final message is unterminated
                self.assertEqual(
                    prefill_output,
                    "<|im_start|>system\nsystem message<|im_end|>\n<|im_start|>user\nuser message<|im_end|>\n<|im_start|>assistant\nassistant message",
                )

    @require_jinja
    def test_continue_final_message_with_trim(self):
        """Regression test for chat templates with trimming: https://github.com/huggingface/transformers/pull/34214"""

        dummy_template = """
        {%- for message in messages %}
            {{- "<|im_start|>" + message['role'] + "\n" + message['content'] | trim + "<|im_end|>" + "\n"}}
        {%- endfor %}"""
        dummy_conversation = [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "user message"},
            {"role": "assistant", "content": "assistant message "},  # Note the trailing whitespace
        ]
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                output = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template=dummy_template, tokenize=False, continue_final_message=False
                )
                self.assertEqual(
                    output,
                    "<|im_start|>system\nsystem message<|im_end|>\n<|im_start|>user\nuser message<|im_end|>\n<|im_start|>assistant\nassistant message<|im_end|>\n",
                )
                prefill_output = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template=dummy_template, tokenize=False, continue_final_message=True
                )
                # Assert that the final message is unterminated
                self.assertEqual(
                    prefill_output,
                    "<|im_start|>system\nsystem message<|im_end|>\n<|im_start|>user\nuser message<|im_end|>\n<|im_start|>assistant\nassistant message",
                )

    @require_jinja
    def test_continue_final_message_with_decoy_earlier_message(self):
        """Regression test for chat templates where an earlier message has similar content to the final message
        https://github.com/huggingface/transformers/issues/35433"""

        dummy_template = """
        {%- for message in messages %}
            {{- "<|im_start|>" + message['role'] + "\n" + message['content'] | trim + "<|im_end|>" + "\n"}}
        {%- endfor %}"""
        dummy_conversation = [
            {"role": "user", "content": "hi 0"},
            {"role": "assistant", "content": "bye: 0"},
            {"role": "user", "content": "hi 1"},
            {"role": "assistant", "content": "bye: "},
        ]
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                prefill_output = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template=dummy_template, tokenize=False, continue_final_message=True
                )
                # Assert that the final message is unterminated
                self.assertEqual(
                    prefill_output,
                    "<|im_start|>user\nhi 0<|im_end|>\n<|im_start|>assistant\nbye: 0<|im_end|>\n<|im_start|>user\nhi 1<|im_end|>\n<|im_start|>assistant\nbye:",
                )

    @require_jinja
    def test_chat_template_dict(self):
        dummy_template_1 = "{{'a'}}"
        dummy_template_2 = "{{'b'}}"
        dummy_conversation = [
            {"role": "user", "content": "user message"},
        ]
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                tokenizer.chat_template = {"template1": dummy_template_1, "template2": dummy_template_2}
                output1 = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template=dummy_template_1, tokenize=False
                )
                output1_via_dict = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template="template1", tokenize=False
                )
                self.assertEqual(output1, output1_via_dict)
                output2 = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template=dummy_template_2, tokenize=False
                )
                output2_via_dict = tokenizer.apply_chat_template(
                    dummy_conversation, chat_template="template2", tokenize=False
                )
                self.assertEqual(output2, output2_via_dict)

    @require_jinja
    def test_chat_template_dict_saving(self):
        dummy_template_1 = "{{'a'}}"
        dummy_template_2 = "{{'b'}}"
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                for save_jinja_files in (True, False):
                    tokenizer.chat_template = {"default": dummy_template_1, "template2": dummy_template_2}
                    with tempfile.TemporaryDirectory() as tmp_dir_name:
                        # Test that save_jinja_files is ignored when there's a dict of multiple templates
                        tokenizer.save_pretrained(tmp_dir_name, save_jinja_files=save_jinja_files)
                        if save_jinja_files:
                            config_dict = json.load(open(os.path.join(tmp_dir_name, "tokenizer_config.json")))
                            self.assertNotIn("chat_template", config_dict)
                            self.assertTrue(os.path.exists(os.path.join(tmp_dir_name, "chat_template.jinja")))
                            self.assertTrue(
                                os.path.exists(os.path.join(tmp_dir_name, "additional_chat_templates/template2.jinja"))
                            )
                        else:
                            config_dict = json.load(open(os.path.join(tmp_dir_name, "tokenizer_config.json")))
                            # Assert that chat templates are correctly serialized as lists of dictionaries
                            self.assertEqual(
                                config_dict["chat_template"],
                                [
                                    {"name": "default", "template": "{{'a'}}"},
                                    {"name": "template2", "template": "{{'b'}}"},
                                ],
                            )
                            self.assertFalse(os.path.exists(os.path.join(tmp_dir_name, "chat_template.jinja")))
                        new_tokenizer = tokenizer.from_pretrained(tmp_dir_name)
                    # Assert that the serialized list is correctly reconstructed as a single dict
                    self.assertEqual(new_tokenizer.chat_template, tokenizer.chat_template)

    @require_jinja
    def test_chat_template_file_priority(self):
        dummy_template1 = "a"
        dummy_template2 = "b"
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    tokenizer.chat_template = dummy_template1
                    tokenizer.save_pretrained(tmp_dir_name, save_jinja_files=False)
                    with Path(tmp_dir_name, "chat_template.jinja").open("w") as f:
                        f.write(dummy_template2)
                    new_tokenizer = tokenizer.from_pretrained(tmp_dir_name)
                # Assert the file template clobbers any template in the config
                self.assertEqual(new_tokenizer.chat_template, dummy_template2)

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

                self.assertGreater(
                    total_length, 4, "Issue with the testing sequence, please update it, it's too short"
                )

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_1 = seq_0 * model_max_length

                sequence1 = tokenizer(seq_1, add_special_tokens=False)
                total_length1 = len(sequence1["input_ids"])
                self.assertGreater(
                    total_length1,
                    model_max_length,
                    "Issue with the testing sequence, please update it, it's too short",
                )

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
                                "Token indices sequence length is longer than the specified maximum sequence length"
                                " for this model"
                            )
                        )

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer([seq_1], padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length"
                                " for this model"
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
                self.assertGreater(len(seq0_tokens), 2 + stride)

                seq_1 = "This is another sentence to be encoded."
                seq1_tokens = tokenizer.encode(seq_1, add_special_tokens=False)
                if abs(len(seq0_tokens) - len(seq1_tokens)) <= 2:
                    seq1_tokens = seq1_tokens + seq1_tokens
                    seq_1 = tokenizer.decode(seq1_tokens, clean_up_tokenization_spaces=False)
                seq1_tokens = tokenizer.encode(seq_1, add_special_tokens=False)

                self.assertGreater(len(seq1_tokens), 2 + stride)

                smallest = seq1_tokens if len(seq0_tokens) > len(seq1_tokens) else seq0_tokens

                # We are not using the special tokens - a bit too hard to test all the tokenizers with this
                # TODO try this again later
                sequence = tokenizer.encode(seq_0, seq_1, add_special_tokens=False)  # , add_prefix_space=False)

                # Test with max model input length
                model_max_length = tokenizer.model_max_length
                self.assertEqual(model_max_length, 100)
                seq_2 = seq_0 * model_max_length
                self.assertGreater(len(seq_2), model_max_length)

                sequence1 = tokenizer(seq_1, add_special_tokens=False)
                total_length1 = len(sequence1["input_ids"])
                sequence2 = tokenizer(seq_2, seq_1, add_special_tokens=False)
                total_length2 = len(sequence2["input_ids"])
                self.assertLess(
                    total_length1, model_max_length - 10, "Issue with the testing sequence, please update it."
                )
                self.assertGreater(
                    total_length2, model_max_length, "Issue with the testing sequence, please update it."
                )

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
                                "Token indices sequence length is longer than the specified maximum sequence length"
                                " for this model"
                            )
                        )

                        tokenizer.deprecation_warnings = {}
                        with self.assertLogs("transformers", level="WARNING") as cm:
                            output = tokenizer([seq_1], [seq_2], padding=padding_state, truncation=False)
                            self.assertNotEqual(len(output["input_ids"][0]), model_max_length)
                        self.assertEqual(len(cm.records), 1)
                        self.assertTrue(
                            cm.records[0].message.startswith(
                                "Token indices sequence length is longer than the specified maximum sequence length"
                                " for this model"
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


    def test_special_tokens_mask(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence_0 = "Encode this."
                # Testing single inputs
                encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False)
                encoded_sequence_dict = tokenizer.encode_plus(
                    sequence_0,
                    add_special_tokens=True,
                    return_special_tokens_mask=True,  # , add_prefix_space=False
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

    def test_padding_side_in_kwargs(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                if self.test_rust_tokenizer:
                    tokenizer_r = self.get_rust_tokenizer(pretrained_name, padding_side="left", **kwargs)
                    self.assertEqual(tokenizer_r.padding_side, "left")

                    tokenizer_r = self.get_rust_tokenizer(pretrained_name, padding_side="right", **kwargs)
                    self.assertEqual(tokenizer_r.padding_side, "right")

                    self.assertRaises(
                        ValueError,
                        self.rust_tokenizer_class.from_pretrained,
                        pretrained_name,
                        padding_side="unauthorized",
                        **kwargs,
                    )

                if self.test_slow_tokenizer:
                    tokenizer_p = self.get_tokenizer(pretrained_name, padding_side="left", **kwargs)
                    self.assertEqual(tokenizer_p.padding_side, "left")

                    tokenizer_p = self.get_tokenizer(pretrained_name, padding_side="right", **kwargs)
                    self.assertEqual(tokenizer_p.padding_side, "right")

                    self.assertRaises(
                        ValueError,
                        self.tokenizer_class.from_pretrained,
                        pretrained_name,
                        padding_side="unauthorized",
                        **kwargs,
                    )

    def test_truncation_side_in_kwargs(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                if self.test_rust_tokenizer:
                    tokenizer_r = self.get_rust_tokenizer(pretrained_name, truncation_side="left", **kwargs)
                    self.assertEqual(tokenizer_r.truncation_side, "left")

                    tokenizer_r = self.get_rust_tokenizer(pretrained_name, truncation_side="right", **kwargs)
                    self.assertEqual(tokenizer_r.truncation_side, "right")

                    self.assertRaises(
                        ValueError,
                        self.rust_tokenizer_class.from_pretrained,
                        pretrained_name,
                        truncation_side="unauthorized",
                        **kwargs,
                    )

                if self.test_slow_tokenizer:
                    tokenizer_p = self.get_tokenizer(pretrained_name, truncation_side="left", **kwargs)
                    self.assertEqual(tokenizer_p.truncation_side, "left")

                    tokenizer_p = self.get_tokenizer(pretrained_name, truncation_side="right", **kwargs)
                    self.assertEqual(tokenizer_p.truncation_side, "right")

                    self.assertRaises(
                        ValueError,
                        self.tokenizer_class.from_pretrained,
                        pretrained_name,
                        truncation_side="unauthorized",
                        **kwargs,
                    )

    def test_encode_basic_padding(self):
        """Test basic left/right padding behavior using encode() method with max_length strategy."""
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
                self.assertEqual(sequence_length + padding_size, padded_sequence_length)
                self.assertEqual(encoded_sequence + [padding_idx] * padding_size, padded_sequence)

                # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                tokenizer.padding_side = "left"
                encoded_sequence = tokenizer.encode(sequence)
                sequence_length = len(encoded_sequence)
                padded_sequence = tokenizer.encode(
                    sequence, max_length=sequence_length + padding_size, padding="max_length"
                )
                padded_sequence_length = len(padded_sequence)
                self.assertEqual(sequence_length + padding_size, padded_sequence_length)
                self.assertEqual([padding_idx] * padding_size + encoded_sequence, padded_sequence)

    def test_right_and_left_truncation(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                sequence = "This is a test sequence"

                # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
                truncation_size = 3
                tokenizer.truncation_side = "right"
                encoded_sequence = tokenizer.encode(sequence, add_special_tokens=False)
                sequence_length = len(encoded_sequence)
                # Remove EOS/BOS tokens
                truncated_sequence = tokenizer.encode(
                    sequence, max_length=sequence_length - truncation_size, truncation=True, add_special_tokens=False
                )
                truncated_sequence_length = len(truncated_sequence)
                self.assertEqual(sequence_length, truncated_sequence_length + truncation_size)
                self.assertEqual(encoded_sequence[:-truncation_size], truncated_sequence)

                # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the truncation flag set to True
                tokenizer.truncation_side = "left"
                sequence_length = len(encoded_sequence)
                truncated_sequence = tokenizer.encode(
                    sequence, max_length=sequence_length - truncation_size, truncation=True, add_special_tokens=False
                )
                truncated_sequence_length = len(truncated_sequence)
                self.assertEqual(sequence_length, truncated_sequence_length + truncation_size)
                self.assertEqual(encoded_sequence[truncation_size:], truncated_sequence)

                # RIGHT & LEFT PADDING - Check that nothing is done for 'longest' and 'no_truncation'
                sequence_length = len(encoded_sequence)

                tokenizer.truncation_side = "right"
                truncated_sequence_right = tokenizer.encode(sequence, truncation=True, add_special_tokens=False)
                truncated_sequence_right_length = len(truncated_sequence_right)
                self.assertEqual(sequence_length, truncated_sequence_right_length)
                self.assertEqual(encoded_sequence, truncated_sequence_right)

                tokenizer.truncation_side = "left"
                truncated_sequence_left = tokenizer.encode(
                    sequence, truncation="longest_first", add_special_tokens=False
                )
                truncated_sequence_left_length = len(truncated_sequence_left)
                self.assertEqual(sequence_length, truncated_sequence_left_length)
                self.assertEqual(encoded_sequence, truncated_sequence_left)

                tokenizer.truncation_side = "right"
                truncated_sequence_right = tokenizer.encode(sequence, add_special_tokens=False)
                truncated_sequence_right_length = len(truncated_sequence_right)
                self.assertEqual(sequence_length, truncated_sequence_right_length)
                self.assertEqual(encoded_sequence, truncated_sequence_right)

                tokenizer.truncation_side = "left"
                truncated_sequence_left = tokenizer.encode(sequence, truncation=False, add_special_tokens=False)
                truncated_sequence_left_length = len(truncated_sequence_left)
                self.assertEqual(sequence_length, truncated_sequence_left_length)
                self.assertEqual(encoded_sequence, truncated_sequence_left)

    def test_padding_to_multiple_of(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.pad_token is None:
                    self.skipTest(reason="No padding token.")
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

    def test_padding_with_attention_mask(self):
        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.pad_token is None:
                    self.skipTest(reason="No padding token.")
                if "attention_mask" not in tokenizer.model_input_names:
                    self.skipTest(reason="This model does not use attention mask.")

                features = [
                    {"input_ids": [1, 2, 3, 4, 5, 6], "attention_mask": [1, 1, 1, 1, 1, 0]},
                    {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 0]},
                ]
                padded_features = tokenizer.pad(features)
                if tokenizer.padding_side == "right":
                    self.assertListEqual(padded_features["attention_mask"], [[1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0]])
                else:
                    self.assertListEqual(padded_features["attention_mask"], [[1, 1, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0]])

    @parameterized.expand([(True,), (False,)])
    def test_encode_plus_with_padding(self, use_padding_as_call_kwarg: bool):
        """
        This test checks that padding works as expected when tokenizing a sequence.
        Padding is expected to have no effect when the input is a single sequence and
        the padding-strategy is not `max_length`. Otherwise it pads to the specified max-length
        using tokenizer classes `padding_side` attribute. Also, we check that passing `padding_side`
        as call time kwarg works same way as when one sets `tokenizer.padding_side` attribute.
        """
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
                not_padded_sequence = tokenizer.encode_plus(
                    sequence,
                    padding=True,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                self.assertEqual(sequence_length, not_padded_sequence_length)
                self.assertEqual(input_ids, not_padded_input_ids)
                self.assertEqual(special_tokens_mask, not_padded_special_tokens_mask)

                not_padded_sequence = tokenizer.encode_plus(
                    sequence,
                    padding=False,
                    return_special_tokens_mask=True,
                )
                not_padded_input_ids = not_padded_sequence["input_ids"]

                not_padded_special_tokens_mask = not_padded_sequence["special_tokens_mask"]
                not_padded_sequence_length = len(not_padded_input_ids)

                self.assertEqual(sequence_length, not_padded_sequence_length)
                self.assertEqual(input_ids, not_padded_input_ids)
                self.assertEqual(special_tokens_mask, not_padded_special_tokens_mask)

                # Test right padding
                tokenizer_kwargs_right = {
                    "max_length": sequence_length + padding_size,
                    "padding": "max_length",
                    "return_special_tokens_mask": True,
                }

                if not use_padding_as_call_kwarg:
                    tokenizer.padding_side = "right"
                else:
                    tokenizer_kwargs_right["padding_side"] = "right"

                right_padded_sequence = tokenizer.encode_plus(sequence, **tokenizer_kwargs_right)
                right_padded_input_ids = right_padded_sequence["input_ids"]

                right_padded_special_tokens_mask = right_padded_sequence["special_tokens_mask"]
                right_padded_sequence_length = len(right_padded_input_ids)

                self.assertEqual(sequence_length + padding_size, right_padded_sequence_length)
                self.assertEqual(input_ids + [padding_idx] * padding_size, right_padded_input_ids)
                self.assertEqual(special_tokens_mask + [1] * padding_size, right_padded_special_tokens_mask)

                # Test left padding
                tokenizer_kwargs_left = {
                    "max_length": sequence_length + padding_size,
                    "padding": "max_length",
                    "return_special_tokens_mask": True,
                }

                if not use_padding_as_call_kwarg:
                    tokenizer.padding_side = "left"
                else:
                    tokenizer_kwargs_left["padding_side"] = "left"

                left_padded_sequence = tokenizer.encode_plus(sequence, **tokenizer_kwargs_left)
                left_padded_input_ids = left_padded_sequence["input_ids"]
                left_padded_special_tokens_mask = left_padded_sequence["special_tokens_mask"]
                left_padded_sequence_length = len(left_padded_input_ids)

                self.assertEqual(sequence_length + padding_size, left_padded_sequence_length)
                self.assertEqual([padding_idx] * padding_size + input_ids, left_padded_input_ids)
                self.assertEqual([1] * padding_size + special_tokens_mask, left_padded_special_tokens_mask)

                if "token_type_ids" in tokenizer.model_input_names:
                    token_type_ids = encoded_sequence["token_type_ids"]
                    left_padded_token_type_ids = left_padded_sequence["token_type_ids"]
                    right_padded_token_type_ids = right_padded_sequence["token_type_ids"]

                    self.assertEqual(
                        token_type_ids + [token_type_padding_idx] * padding_size, right_padded_token_type_ids
                    )
                    self.assertEqual(
                        [token_type_padding_idx] * padding_size + token_type_ids, left_padded_token_type_ids
                    )

                if "attention_mask" in tokenizer.model_input_names:
                    attention_mask = encoded_sequence["attention_mask"]
                    right_padded_attention_mask = right_padded_sequence["attention_mask"]
                    left_padded_attention_mask = left_padded_sequence["attention_mask"]

                    self.assertEqual(attention_mask + [0] * padding_size, right_padded_attention_mask)
                    self.assertEqual([0] * padding_size + attention_mask, left_padded_attention_mask)

    def test_padding_warning_message_fast_tokenizer(self):
        self.skipTest(reason="Test removed for v5, no value.")

    def test_separate_tokenizers(self):
        self.skipTest(reason="Test removed for v5, this test was testing random arguments.")

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
        # Tests that all call wrap to encode_plus
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
                encoded_sequences_1 = tokenizer(sequences)
                encoded_sequences_2 = tokenizer(sequences)
                self.assertEqual(encoded_sequences_1, encoded_sequences_2)

                # Test batched pairs
                encoded_sequences_1 = tokenizer(list(zip(sequences, sequences)))
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
                encoded_sequences_batch = tokenizer(sequences, padding=False)
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_to_list_format(encoded_sequences_batch)
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

                encoded_sequences_batch_padded = tokenizer(sequences, padding=True)
                self.assertListEqual(
                    encoded_sequences_padded,
                    self.convert_batch_to_list_format(encoded_sequences_batch_padded),
                )

                # check 'longest' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer(sequences, padding=True)
                encoded_sequences_batch_padded_2 = tokenizer(
                    sequences, max_length=maximum_length + 10, padding="longest"
                )
                for key in encoded_sequences_batch_padded_1:
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )

                # check 'no_padding' is unsensitive to a max length
                encoded_sequences_batch_padded_1 = tokenizer(sequences, padding=False)
                encoded_sequences_batch_padded_2 = tokenizer(
                    sequences, max_length=maximum_length + 10, padding=False
                )
                for key in encoded_sequences_batch_padded_1:
                    self.assertListEqual(
                        encoded_sequences_batch_padded_1[key],
                        encoded_sequences_batch_padded_2[key],
                    )


    @require_tokenizers
    def test_added_token_serializable(self):
        self.skipTest(reason="Completely covered by test_save_and_load_tokenizer")

    def test_batch_encode_plus_padding(self):
        # Test that padded sequences are equivalent between batch and individual encoding

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
                encoded_sequences_batch = tokenizer(
                    sequences, max_length=max_length, padding="max_length"
                )
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_to_list_format(encoded_sequences_batch)
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
                encoded_sequences_batch = tokenizer(
                    sequences, max_length=max_length, padding="max_length"
                )
                self.assertListEqual(
                    encoded_sequences, self.convert_batch_to_list_format(encoded_sequences_batch)
                )

    def test_pretokenized_inputs(self):
        # Test when inputs are pretokenized
        # All methods (encode, encode_plus, __call__) go through the same code path,
        # so we only test __call__ 

        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if hasattr(tokenizer, "add_prefix_space") and not tokenizer.add_prefix_space:
                    continue

                # Prepare a sequence from our tokenizer vocabulary
                sequence, ids = self.get_clean_sequence(tokenizer, with_prefix_space=True, max_length=20)
                token_sequence = sequence.split()

                # Test single sequence
                output = tokenizer(token_sequence, is_split_into_words=True, add_special_tokens=False)
                output_sequence = tokenizer(sequence, add_special_tokens=False)
                for key in output:
                    self.assertEqual(output[key], output_sequence[key])

                output = tokenizer(token_sequence, is_split_into_words=True, add_special_tokens=True)
                output_sequence = tokenizer(sequence, add_special_tokens=True)
                for key in output:
                    self.assertEqual(output[key], output_sequence[key])

                # Test sequence pairs
                output = tokenizer(
                    token_sequence, token_sequence, is_split_into_words=True, add_special_tokens=False
                )
                output_sequence = tokenizer(sequence, sequence, add_special_tokens=False)
                for key in output:
                    self.assertEqual(output[key], output_sequence[key])

                output = tokenizer(
                    token_sequence, token_sequence, is_split_into_words=True, add_special_tokens=True
                )
                output_sequence = tokenizer(sequence, sequence, add_special_tokens=True)
                for key in output:
                    self.assertEqual(output[key], output_sequence[key])

                # Test batched inputs
                sequence_batch = [sequence.strip()] * 2 + [sequence.strip() + " " + sequence.strip()]
                token_sequence_batch = [s.split() for s in sequence_batch]
                sequence_batch_cleaned_up_spaces = [" " + " ".join(s) for s in token_sequence_batch]

                output = tokenizer(
                    token_sequence_batch, is_split_into_words=True, add_special_tokens=False
                )
                output_sequence = tokenizer(
                    sequence_batch_cleaned_up_spaces, add_special_tokens=False
                )
                for key in output:
                    self.assertEqual(output[key], output_sequence[key])

                output = tokenizer(
                    token_sequence_batch, is_split_into_words=True, add_special_tokens=True
                )
                output_sequence = tokenizer(
                    sequence_batch_cleaned_up_spaces, add_special_tokens=True
                )
                for key in output:
                    self.assertEqual(output[key], output_sequence[key])

                # Test batch_encode_plus for pretokenized inputs pairs
                sequence_pair_batch = [(sequence.strip(), sequence.strip())] * 2 + [
                    (sequence.strip() + " " + sequence.strip(), sequence.strip())
                ]
                token_sequence_pair_batch = [tuple(s.split() for s in pair) for pair in sequence_pair_batch]
                sequence_pair_batch_cleaned_up_spaces = [
                    tuple(" " + " ".join(s) for s in pair) for pair in token_sequence_pair_batch
                ]

                output = tokenizer(
                    token_sequence_pair_batch, is_split_into_words=True, add_special_tokens=False
                )
                output_sequence = tokenizer(
                    sequence_pair_batch_cleaned_up_spaces, add_special_tokens=False
                )
                for key in output:
                    self.assertEqual(output[key], output_sequence[key])
                output = tokenizer(
                    token_sequence_pair_batch, is_split_into_words=True, add_special_tokens=True
                )
                output_sequence = tokenizer(
                    sequence_pair_batch_cleaned_up_spaces, add_special_tokens=True
                )
                for key in output:
                    self.assertEqual(output[key], output_sequence[key])

    def test_prepare_for_model(self):
        self.skipTest(reason="Test removed for v5, only tests slow")

    def test_batch_encode_plus_overflowing_tokens(self):
        self.skipTest(reason="Smoke test, covered by test_maximum_encoding_* and test_batch_encode_dynamic_overflowing")

    def _check_no_pad_token_padding(self, tokenizer, sequences):
        # if tokenizer does  v have pad_token_id, an error should be thrown
        if tokenizer.pad_token_id is None:
            with self.assertRaises(ValueError):
                if isinstance(sequences, list):
                    tokenizer(sequences, padding="longest")
                else:
                    tokenizer.encode_plus(sequences, padding=True)

            # add pad_token_id to pass subsequent tests
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})

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
                    self.skipTest(f"{tokenizer.__class__.__name__} is not in the MODEL_TOKENIZER")

                config_class, model_class = MODEL_TOKENIZER_MAPPING[tokenizer.__class__]
                config = config_class()

                if config.is_encoder_decoder or config.pad_token_id is None:
                    self.skipTest(reason="Model is not an encoder-decoder model or has no set pad token id")

                model = model_class(config)

                # Make sure the model contains at least the full vocabulary size in its embedding matrix
                is_using_common_embeddings = hasattr(model.get_input_embeddings(), "weight")
                if is_using_common_embeddings:
                    self.assertGreaterEqual(model.get_input_embeddings().weight.shape[0], len(tokenizer))

                # Build sequence
                first_ten_tokens = list(tokenizer.get_vocab().keys())[:10]
                sequence = " ".join(first_ten_tokens)
                encoded_sequence = tokenizer.encode_plus(sequence, return_tensors="pt")

                # Ensure that the BatchEncoding.to() method works.
                encoded_sequence.to(model.device)

                batch_encoded_sequence = tokenizer([sequence, sequence], return_tensors="pt")
                # This should not fail

                with torch.no_grad():  # saves some time
                    model(**encoded_sequence)
                    model(**batch_encoded_sequence)

    @require_torch
    @slow
    def test_np_encode_plus_sent_to_model(self):
        from transformers import MODEL_MAPPING, TOKENIZER_MAPPING

        MODEL_TOKENIZER_MAPPING = merge_model_tokenizer_mappings(MODEL_MAPPING, TOKENIZER_MAPPING)

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                if tokenizer.__class__ not in MODEL_TOKENIZER_MAPPING:
                    self.skipTest(f"{tokenizer.__class__.__name__} is not in the MODEL_TOKENIZER_MAPPING")

                config_class, model_class = MODEL_TOKENIZER_MAPPING[tokenizer.__class__]
                config = config_class()

                if config.is_encoder_decoder or config.pad_token_id is None:
                    self.skipTest("Model is not an encoder-decoder model or has no set pad token id")

                # Build sequence
                first_ten_tokens = list(tokenizer.get_vocab().keys())[:10]
                sequence = " ".join(first_ten_tokens)
                encoded_sequence = tokenizer.encode_plus(sequence, return_tensors="np")
                batch_encoded_sequence = tokenizer([sequence, sequence], return_tensors="np")

                # This is currently here to make ruff happy !
                if encoded_sequence is None:
                    raise ValueError("Cannot convert list to numpy tensor on  encode_plus()")

                if batch_encoded_sequence is None:
                    raise ValueError("Cannot convert list to numpy tensor on  batch_encode_plus()")

                fast_tokenizer = self.get_rust_tokenizer()
                encoded_sequence_fast = fast_tokenizer.encode_plus(sequence, return_tensors="np")
                batch_encoded_sequence_fast = fast_tokenizer(
                    [sequence, sequence], return_tensors="np"
                )

                # This is currently here to make ruff happy !
                if encoded_sequence_fast is None:
                    raise ValueError("Cannot convert list to numpy tensor on  encode_plus() (fast)")

                if batch_encoded_sequence_fast is None:
                    raise ValueError("Cannot convert list to numpy tensor on  batch_encode_plus() (fast)")

    @require_torch
    def test_prepare_seq2seq_batch(self):
        if not self.test_seq2seq:
            self.skipTest(reason="test_seq2seq is set to False")

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Longer text that will definitely require truncation.
                src_text = [
                    " UN Chief Says There Is No Military Solution in Syria",
                    " Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for"
                    " Syria is that 'there is no military solution' to the nearly five-year conflict and more weapons"
                    " will only worsen the violence and misery for millions of people.",
                ]
                tgt_text = [
                    "Şeful ONU declară că nu există o soluţie militară în Siria",
                    "Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar al"
                    ' Rusiei pentru Siria este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi'
                    " că noi arme nu vor face decât să înrăutăţească violenţele şi mizeria pentru milioane de oameni.",
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
                    self.skipTest(reason="Encountered NotImplementedError calling prepare_seq2seq_batch")
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
        self.skipTest(reason="Test removed for v5, we will only have fast.")

    def test_fast_only_inputs(self):
        self.skipTest(reason="Test removed for v5, inputs can be None.")

    def test_alignment_methods(self):
        self.skipTest(
            reason="This test is now in TokenizersBackendTesterMixin - it tests tokenizers-backend API, not transformers code"
        )

    def test_tokenization_python_rust_equals(self):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_num_special_tokens_to_add_equal(self):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_max_length_equal(self):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_special_tokens_map_equal(self):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_add_tokens(self):
        self.skipTest(
            reason="Moved to SentencePieceBackendTesterMixin, tokenizers should test this, not transformers"
        )

    def test_offsets_mapping(self):
        self.skipTest(
            reason="This test is now in TokenizersBackendTesterMixin - it tests tokenizers-backend API, not transformers code"
        )

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
            tokenizer = self.get_rust_tokenizer(pretrained_name, **kwargs)

            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name}, {tokenizer.__class__.__name__})"):
                if is_torch_available():
                    returned_tensor = "pt"
                else:
                    self.skipTest(reason="No expected framework (PT) found")

                if not tokenizer.pad_token or tokenizer.pad_token_id < 0:
                    self.skipTest(reason="This tokenizer has no padding token set, or pad_token_id < 0")

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
                tokens = tokenizer(
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
                tokens = tokenizer(
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
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_create_token_type_ids(self):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_build_inputs_with_special_tokens(self):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_padding(self, max_length=50):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_padding_different_model_input_name(self):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_save_pretrained(self):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_embedded_special_tokens(self):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_compare_add_special_tokens(self):
        self.skipTest(
            reason="Moved to SentencePieceBackendTesterMixin, tokenizers should test this, not transformers"
        )

    def test_compare_prepare_for_model(self):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_special_tokens_initialization(self):
        self.skipTest(
            reason="Moved to SentencePieceBackendTesterMixin, tokenizers should test this, not transformers"
        )

    def test_special_tokens_initialization_with_non_empty_additional_special_tokens(self):
        self.skipTest(reason="Test removed for v5, only tests slow tokenizers")

    def test_training_new_tokenizer(self):
        self.skipTest(
            reason="This test is now in TokenizersBackendTesterMixin - it tests tokenizers-backend API, not transformers code"
        )

    def test_training_new_tokenizer_with_special_tokens_change(self):
        self.skipTest(
            reason="This test is now in TokenizersBackendTesterMixin - it tests tokenizers-backend API, not transformers code"
        )

    def test_tokenizer_mismatch_warning(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                with self.assertLogs("transformers", level="WARNING") as cm:
                    try:
                        if self.tokenizer_class == BertTokenizer:
                            AlbertTokenizer.from_pretrained(pretrained_name)
                        else:
                            BertTokenizer.from_pretrained(pretrained_name)
                    except OSError as e:
                        # Some tokenizer will raised an error before reaching the logged warning because there are no
                        # corresponding files to load
                        error_message = str(e)
                    except (TypeError, AttributeError):
                        # Some tokenizers cannot be loaded into the target tokenizer at all and errors are returned,
                        # here we just check that the warning has been logged before the error is raised
                        pass
                    finally:
                        logged_msg_target = (
                            "The tokenizer class you load from this checkpoint is not the same type as the class "
                            "this function is called from."
                        )
                        raised_error_msg_target = "Can't load tokenizer for"
                        self.assertTrue(
                            cm.records[0].message.startswith(logged_msg_target)
                            if len(cm.records) > 0
                            else False or raised_error_msg_target in error_message
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
                                "The tokenizer class you load from this checkpoint is not the same type as the class"
                                " this function is called from."
                            )
                        )

    @require_torch
    def test_saving_tokenizer_trainer(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Save the fast tokenizer files in a temporary directory
                    tokenizer_old = self.get_rust_tokenizer(pretrained_name, **kwargs, use_fast=True)
                    tokenizer_old.save_pretrained(tmp_dir, legacy_format=False)  # save only fast version

                    # Initialize toy model for the trainer
                    model = nn.Module()

                    # Load tokenizer from a folder without legacy files
                    tokenizer = self.rust_tokenizer_class.from_pretrained(tmp_dir)
                    training_args = TrainingArguments(output_dir=tmp_dir, do_train=True, use_cpu=True)
                    trainer = Trainer(model=model, args=training_args, processing_class=tokenizer)

                    # Should not raise an error
                    trainer.save_model(os.path.join(tmp_dir, "checkpoint"))
                    self.assertIn("tokenizer.json", os.listdir(os.path.join(tmp_dir, "checkpoint")))


    def test_save_slow_from_fast_and_reload_fast(self):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_split_special_tokens(self):
        self.skipTest(reason="Test removed for v5, was comparing slow vs fast tokenizers")

    def test_added_tokens_serialization(self):
        new_eos = AddedToken("[NEW_EOS]", rstrip=False, lstrip=True, normalized=False, special=True)
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                # Test loading a fast tokenizer from the hub with a new eos token
                tokenizer_fast = self.get_rust_tokenizer(pretrained_name, eos_token=new_eos)
                self.assertEqual(tokenizer_fast._special_tokens_map["eos_token"], new_eos)
                self.assertIn(new_eos, list(tokenizer_fast.added_tokens_decoder.values()))

                EXPECTED_ADDED_TOKENS_DECODER = tokenizer_fast.added_tokens_decoder
                
                # Test saving and reloading the fast tokenizer
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tokenizer_fast.save_pretrained(tmp_dir)
                    
                    with self.subTest("Fast -> Fast: saving fast tokenizer locally and reloading"):
                        tokenizer = self.rust_tokenizer_class.from_pretrained(tmp_dir)
                        self.assertTrue(str(new_eos) not in tokenizer.additional_special_tokens)
                        self.assertIn(new_eos, tokenizer.added_tokens_decoder.values())
                        self.assertEqual(tokenizer.added_tokens_decoder[tokenizer.eos_token_id], new_eos)
                        self.assertTrue(all(item in tokenizer.added_tokens_decoder.items() for item in EXPECTED_ADDED_TOKENS_DECODER.items()))

    def test_special_token_addition(self):
        self.skipTest(
            reason="Moved to SentencePieceBackendTesterMixin, tokenizers should test this, not transformers"
        )

    def test_tokenizer_initialization_with_conflicting_key(self):
        with self.assertRaises(AttributeError, msg="conflicts with the method"):
            self.get_rust_tokenizer(add_special_tokens=True)

        with self.assertRaises(AttributeError, msg="conflicts with the method"):
            self.get_rust_tokenizer(get_vocab=True)

    @parameterized.expand([(True,), (False,)])
    def test_rust_tokenizer_add_prefix_space(self, add_prefix_space):
        self.skipTest(
            reason="This test is now in TokenizersBackendTesterMixin - it tests tokenizers-backend API, not transformers code"
        )

    def test_empty_input_string(self):
        empty_input_string = ""
        tokenizer_return_type = []
        output_tensor_type = []

        if is_torch_available():
            import numpy as np
            import torch

            tokenizer_return_type.append("pt")
            output_tensor_type.append(torch.int64)
            tokenizer_return_type.append("np")
            output_tensor_type.append(np.int64)

        if is_mlx_available():
            import mlx.core as mx

            tokenizer_return_type.append("mlx")
            output_tensor_type.append(mx.int32)

        if len(tokenizer_return_type) == 0:
            self.skipTest(reason="No expected framework from PT, or MLX found")

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                for return_type, target_type in zip(tokenizer_return_type, output_tensor_type):
                    output = tokenizer(empty_input_string, return_tensors=return_type)
                    self.assertEqual(output.input_ids.dtype, target_type)


@require_tokenizers
class TokenizersBackendCommonTest(unittest.TestCase, TokenizersBackendTesterMixin):
    """
    A single test class that runs all tokenizers-backend tests once.
    Uses BertTokenizerFast as a representative fast tokenizer.
    """

    rust_tokenizer_class = BertTokenizerFast
    from_pretrained_id = "google-bert/bert-base-uncased"
    from_pretrained_kwargs = {}


class SentencePieceBackendCommonTest(unittest.TestCase, SentencePieceBackendTesterMixin):
    """
    A single test class that runs all SentencePiece-backend tests once.
    Uses T5Tokenizer as a representative SentencePiece tokenizer.
    """

    tokenizer_class = AlbertTokenizer
    rust_tokenizer_class = AlbertTokenizerFast
    from_pretrained_id = "albert/albert-base-v1"
    from_pretrained_kwargs = {}
    test_sentencepiece = True
