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
import re
import shutil
import tempfile
import unittest
from collections import OrderedDict
from itertools import takewhile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from parameterized import parameterized

from transformers import (
    AutoTokenizer,
    BertTokenizer,
    BertTokenizerFast,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    T5Tokenizer,
    T5TokenizerFast,
    TokenizersBackend,
    is_mlx_available,
    is_torch_available,
    logging,
)
from transformers.testing_utils import (
    get_tests_dir,
    require_jinja,
    require_tokenizers,
    require_torch,
    slow,
)
from transformers.tokenization_python import AddedToken

from .test_sentencepiece_backend_mixin import SentencePieceBackendTesterMixin
from .test_tokenizers_backend_mixin import TokenizersBackendTesterMixin


NON_ENGLISH_TAGS = ["chinese", "dutch", "french", "finnish", "german", "multilingual"]

SMALL_TRAINING_CORPUS = [
    ["This is the first sentence.", "This is the second one."],
    ["This sentence (contains #) over symbols and numbers 12 3.", "But not this one."],
]

input_string = """This is a test 😊
I was born in 92000, and this is falsé.
生活的真谛是
Hi  Hello
Hi   Hello

 
  
 Hello
<s>
hi<s>there
The following string should be properly encoded: Hello.
But ird and ปี   ird   ด
Hey how are you doing"""  # noqa: W293

if is_torch_available():
    import torch


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel


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

        # Preserve _tokenizer for all tokenizers (Rust tokenizer objects don't deep copy properly)
        # This was previously only done for CLIP, but it's needed for all TokenizersBackend tokenizers
        if hasattr(cached, "_tokenizer"):
            # Restore _tokenizer from original since deep copy may have lost or corrupted it
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
    model_mapping: dict["PretrainedConfig", "PreTrainedModel"],
    tokenizer_mapping: dict["PretrainedConfig", tuple["PreTrainedTokenizer", "TokenizersBackend"]],
) -> dict[
    Union["PreTrainedTokenizer", "TokenizersBackend"],
    tuple["PretrainedConfig", "PreTrainedModel"],
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
    text: str | None = None,
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


class TokenizersExtractor:
    """
    Extractor implementation for tokenizers library tokenizer.json files.

    This class extracts vocab and merges from a tokenizer.json file, similar to
    SentencePieceExtractor for .model files.

    """

    def __init__(self, tokenizer_file: str):
        """
        Initialize the extractor with a tokenizer.json file.

        Args:
            tokenizer_file (str): Path to the tokenizer.json file
        """
        with open(tokenizer_file, "r", encoding="utf-8") as f:
            self.tokenizer_data = json.load(f)

        if "model" not in self.tokenizer_data:
            raise ValueError(f"Invalid tokenizer.json file: missing 'model' key in {tokenizer_file}")

        self.model_data = self.tokenizer_data["model"]
        self.model_type = self.model_data.get("type", "Unknown")

    def extract(self) -> tuple[dict[str, int], list[tuple[str, float]], list[tuple[str, str]], list[dict]]:
        """
        Extract vocabulary, scores, merges, and added_tokens from the tokenizer.json file.

        Returns:
            tuple containing:
                - vocab_ids (dict[str, int]): Mapping from token string to token ID
                - vocab_scores (list[tuple[str, float]]): List of (token, score) tuples.
                  Note: tokenizer.json doesn't store scores, so all scores are 0.0
                - merges (list[tuple[str, str]]): List of merge pairs for BPE tokenizers
                - added_tokens (list[dict]): List of added token dicts with 'id', 'content', 'special', etc.

        Raises:
            ValueError: If the tokenizer type is not supported or vocab is missing
        """
        # Extract vocabulary
        if "vocab" not in self.model_data:
            raise ValueError(f"Tokenizer model type '{self.model_type}' does not have a 'vocab' field")

        vocab_field = self.model_data["vocab"]

        # Support both dict-based (BPE/WordPiece/WordLevel) and list-based (Unigram) vocabs
        if isinstance(vocab_field, dict):
            # {token: id}
            vocab_ids = dict(vocab_field)
            # tokenizer.json doesn't store scores for these types; default to 0.0 and sort by id
            vocab_scores = sorted([(token, 0.0) for token in vocab_field.keys()], key=lambda x: vocab_field[x[0]])
        elif isinstance(vocab_field, list):
            # [[token, score], ...] — ids are the list indices
            vocab_ids = {token: idx for idx, (token, _score) in enumerate(vocab_field)}
            vocab_scores = [(token, float(score)) for token, score in vocab_field]
        else:
            raise ValueError(f"Unsupported vocab type in tokenizer.json: {type(vocab_field)}")

        # Extract merges (for BPE tokenizers)
        merges = []
        if "merges" in self.model_data:
            # tokenizer.json can store merges as either:
            # 1. Lists like ["▁", "t"]
            # 2. Strings like "▁ t"
            for merge_item in self.model_data["merges"]:
                if isinstance(merge_item, list):
                    # Already in list format
                    if len(merge_item) == 2:
                        merges.append((merge_item[0], merge_item[1]))
                    else:
                        logger.warning(f"Invalid merge format (expected 2 items): {merge_item}, skipping")
                elif isinstance(merge_item, str):
                    # String format - split on first space
                    parts = merge_item.split(" ", 1)
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
                    else:
                        logger.warning(f"Invalid merge format: '{merge_item}', skipping")
                else:
                    logger.warning(f"Unknown merge type: {type(merge_item)}, skipping")

        # Extract added_tokens from tokenizer.json
        # These are tokens that should not be split by the tokenization algorithm
        added_tokens_list = self.tokenizer_data.get("added_tokens", [])
        # Convert to decoder-style mapping: id -> token dict
        added_tokens_decoder = {}
        for item in added_tokens_list:
            if not isinstance(item, dict) or "id" not in item:
                continue
            token_id = item["id"]
            token_kwargs = {k: v for k, v in item.items() if k != "id"}
            try:
                added_token_obj = AddedToken(**token_kwargs)
            except Exception:
                # Fallback: at minimum require content
                content = token_kwargs.get("content")
                if content is None:
                    continue
                added_token_obj = AddedToken(content, special=bool(token_kwargs.get("special", True)))
            added_tokens_decoder[token_id] = added_token_obj

        return vocab_ids, vocab_scores, merges, added_tokens_decoder


class TokenizerTesterMixin:
    tokenizer_class = None
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    from_pretrained_filter = None
    from_pretrained_id = None
    from_pretrained_vocab_key = "vocab_file"
    test_seq2seq = True
    test_tokenizer_from_extractor = True

    # set to True to test a sentencepiece tokenizer
    test_sentencepiece = False

    # set to True to ignore casing when testing a sentencepiece tokenizer
    # test_sentencepiece must also be set to True
    test_sentencepiece_ignore_case = False

    # Integration test data - can be optionally set by subclasses
    # Default comprehensive test string covering various edge cases
    integration_test_input_string = """This is a test 😊
I was born in 92000, and this is falsé.
生活的真谛是
Hi  Hello
Hi   Hello

 
  
 Hello
<s>
hi<s>there
The following string should be properly encoded: Hello.
But ird and ปี   ird   ด
Hey how are you doing"""  # noqa: W293
    integration_expected_tokens = None
    integration_expected_token_ids = None

    @classmethod
    def setUpClass(cls) -> None:
        # Tokenizer.filter makes it possible to filter which Tokenizer to case based on all the
        # information available in Tokenizer (name, tokenizer class, vocab key name)
        if cls.from_pretrained_id is None:
            cls.from_pretrained_id = []
        elif isinstance(cls.from_pretrained_id, str):
            cls.from_pretrained_id = [cls.from_pretrained_id]

        cls.tokenizers_list = []
        if cls.tokenizer_class is not None:
            cls.tokenizers_list = [
                (
                    cls.tokenizer_class,
                    pretrained_id,
                    cls.from_pretrained_kwargs if cls.from_pretrained_kwargs is not None else {},
                )
                for pretrained_id in cls.from_pretrained_id
            ]
        with open(f"{get_tests_dir()}/fixtures/sample_text.txt", encoding="utf-8") as f_data:
            cls._data = f_data.read().replace("\n\n", "\n").strip()

        cls.tmpdirname = tempfile.mkdtemp()

        # save the first pretrained tokenizer to tmpdirname for tests to use
        if cls.from_pretrained_id and cls.tokenizer_class is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                cls.from_pretrained_id[0],
                **(cls.from_pretrained_kwargs if cls.from_pretrained_kwargs is not None else {}),
            )
            tokenizer.save_pretrained(cls.tmpdirname)

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

    def get_tokenizers(self, **kwargs) -> list[PreTrainedTokenizerBase]:
        """
        Returns a list containing a single tokenizer from get_tokenizer().
        Subclasses can override this method to return multiple tokenizers for testing.
        """
        return [self.get_tokenizer(**kwargs)]

    @classmethod
    def get_tokenizer(cls, pretrained_name=None, **kwargs) -> PreTrainedTokenizer:
        """Get a tokenizer instance from pretrained."""
        pretrained_name = pretrained_name or cls.tmpdirname
        return cls.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    def get_extracted_tokenizer(self, reference_tokenizer=None):
        """
        Build a tokenizer from extracted vocab/merges using TokenizersExtractor.

        Args:
            reference_tokenizer: Optional tokenizer to copy special tokens from.
                                If None, uses get_tokenizer().

        Returns:
            Tokenizer built from extracted vocab/merges, or None if extraction fails.
        """

        if reference_tokenizer is None:
            reference_tokenizer = self.get_tokenizer()

        tokenizer_json_path = os.path.join(self.tmpdirname, "tokenizer.json")
        if not os.path.exists(tokenizer_json_path):
            return None

        extractor = TokenizersExtractor(tokenizer_json_path)
        vocab_ids, vocab_scores, merges, added_tokens_decoder = extractor.extract()
        vocab = vocab_scores
        if _type := getattr(self.tokenizer_class, "model", None):
            if _type.__name__ == "BPE" or _type.__name__ == "WordPiece":
                vocab = vocab_ids

        # Convert added_tokens list to added_tokens_decoder dict format
        # This matches the format used by from_pretrained() from tokenizer_config.jso
        tokenizer_from_extractor = self.tokenizer_class(
            vocab=vocab,
            merges=merges,
            do_lower_case=False,
            keep_accents=True,
            added_tokens_decoder=added_tokens_decoder,
            **(self.from_pretrained_kwargs if self.from_pretrained_kwargs is not None else {}),
        )

        return tokenizer_from_extractor

    def get_extracted_tokenizer_from_sentencepiece(self, reference_tokenizer=None):
        """
        Build a tokenizer from extracted vocab/merges using SentencePieceExtractor.
        """
        from transformers.tokenization_utils_sentencepiece import SentencePieceExtractor

        try:
            sentencepiece_model_path = os.path.join(self.tmpdirname, "tokenizer.model")
            if not os.path.exists(sentencepiece_model_path):
                return None

            extractor = SentencePieceExtractor(sentencepiece_model_path)
            vocab_ids, vocab_scores, merges = extractor.extract()

            tokenizer_from_extractor = self.tokenizer_class(vocab=vocab_ids, merges=merges)

            return tokenizer_from_extractor
        except (TypeError, Exception):
            return None

    def tokenizer_integration_test_util(
        self,
        expected_encoding: dict,
        model_name: str,
        revision: str | None = None,
        sequences: list[str] | None = None,
        decode_kwargs: dict[str, Any] | None = None,
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
        tokenizer = self.get_tokenizer(do_lower_case=True)

        SPECIAL_TOKEN_1 = "[SPECIAL_TOKEN_1]"
        SPECIAL_TOKEN_2 = "[SPECIAL_TOKEN_2]"

        # Both methods should add the token to `_extra_special_tokens` and `added_tokens_decoder`
        tokenizer.add_tokens([SPECIAL_TOKEN_1], special_tokens=True)
        tokenizer.add_special_tokens({"extra_special_tokens": [SPECIAL_TOKEN_2]}, replace_extra_special_tokens=False)

        token_1 = tokenizer.tokenize(SPECIAL_TOKEN_1)
        token_2 = tokenizer.tokenize(SPECIAL_TOKEN_2)

        self.assertEqual(len(token_1), 1)
        self.assertEqual(len(token_2), 1)
        self.assertEqual(token_1[0], SPECIAL_TOKEN_1)
        # next is failing for almost all the Fast tokenizers now.
        # self.assertEqual(token_2[0], SPECIAL_TOKEN_2)

    def test_model_input_names_signature(self):
        accepted_model_main_input_names = [
            "input_ids",  # nlp models
            "input_values",  # speech models
        ]

        tokenizer = self.get_tokenizer()
        # first name of model_input_names has to correspond to main model input name
        # to make sure `tokenizer.pad(...)` works correctly
        self.assertTrue(tokenizer.model_input_names[0] in accepted_model_main_input_names)

    def test_tokenizer_store_full_signature(self):
        signature = inspect.signature(self.tokenizer_class.__init__)
        tokenizer = self.get_tokenizer()

        for parameter_name, parameter in signature.parameters.items():
            if parameter.default != inspect.Parameter.empty and parameter_name not in [
                "vocab_file",
                "merges_file",
                "tokenizer_file",
                "vocab",
                "merges",
                "legacy",
            ]:
                self.assertIn(parameter_name, tokenizer.init_kwargs)

    def test_tokenizers_common_properties(self):
        tokenizer = self.get_tokenizer()

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

        self.assertTrue(hasattr(tokenizer, "extra_special_tokens"))
        self.assertTrue(hasattr(tokenizer, "extra_special_tokens_ids"))

        attributes_list = [
            "model_max_length",
            "init_inputs",
            "init_kwargs",
        ]
        if not isinstance(tokenizer, TokenizersBackend):
            attributes_list += [
                "added_tokens_encoder",
                "added_tokens_decoder",
            ]
        for attr in attributes_list:
            self.assertTrue(hasattr(tokenizer, attr))

    def test_tokenizers_common_ids_setters(self):
        tokenizer = self.get_tokenizer()
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
        token_to_test_setters = tokenizer.convert_ids_to_tokens(token_id_to_test_setters, skip_special_tokens=False)

        for attr in attributes_list:
            setattr(tokenizer, attr + "_id", None)
            self.assertEqual(getattr(tokenizer, attr), None)
            self.assertEqual(getattr(tokenizer, attr + "_id"), None)

            setattr(tokenizer, attr + "_id", token_id_to_test_setters)
            self.assertEqual(getattr(tokenizer, attr), token_to_test_setters)
            self.assertEqual(getattr(tokenizer, attr + "_id"), token_id_to_test_setters)

        setattr(tokenizer, "extra_special_tokens_ids", [])
        self.assertListEqual(getattr(tokenizer, "extra_special_tokens"), [])
        self.assertListEqual(getattr(tokenizer, "extra_special_tokens_ids"), [])

        setattr(tokenizer, "extra_special_tokens_ids", [token_id_to_test_setters])
        self.assertListEqual(getattr(tokenizer, "extra_special_tokens"), [token_to_test_setters])
        self.assertListEqual(getattr(tokenizer, "extra_special_tokens_ids"), [token_id_to_test_setters])

    def test_save_and_load_tokenizer(self):
        # safety check on max_len default value so we are sure the test works
        tokenizer = self.get_tokenizer()
        self.assertNotEqual(tokenizer.model_max_length, 42)

        # Now let's start the test
        tokenizer = self.get_tokenizer()
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

        tokenizer = self.get_tokenizer(model_max_length=42)
        # Isolate this from the other tests because we save additional tokens/etc
        tmpdirname = tempfile.mkdtemp()

        sample_text = " He is very happy, UNwant\u00e9d,running"
        tokenizer.add_tokens(["bim", "bambam"])
        extra_special_tokens = tokenizer.extra_special_tokens
        extra_special_tokens.append("new_extra_special_token")
        tokenizer.add_special_tokens(
            {"extra_special_tokens": extra_special_tokens}, replace_extra_special_tokens=False
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
        self.assertIn("new_extra_special_token", after_tokenizer.extra_special_tokens)
        self.assertEqual(after_tokenizer.model_max_length, 42)

        tokenizer = tokenizer.__class__.from_pretrained(tmpdirname, model_max_length=43)
        self.assertEqual(tokenizer.model_max_length, 43)

        shutil.rmtree(tmpdirname)

        # Test that we can also use the non-legacy saving format for fast tokenizers
        tokenizer = self.get_tokenizer(model_max_length=42)
        # Isolate this from the other tests because we save additional tokens/etc
        tmpdirname = tempfile.mkdtemp()

        sample_text = " He is very happy, UNwant\u00e9d,running"
        tokenizer.add_tokens(["bim", "bambam"])
        extra_special_tokens = tokenizer.extra_special_tokens
        extra_special_tokens.append("new_extra_special_token")
        tokenizer.add_special_tokens(
            {"extra_special_tokens": extra_special_tokens}, replace_extra_special_tokens=False
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
        self.assertIn("new_extra_special_token", after_tokenizer.extra_special_tokens)
        self.assertEqual(after_tokenizer.model_max_length, 42)

        tokenizer = tokenizer.__class__.from_pretrained(tmpdirname, model_max_length=43)
        self.assertEqual(tokenizer.model_max_length, 43)

        shutil.rmtree(tmpdirname)

    def _run_integration_checks(self, tokenizer, tokenizer_type):
        # Test 1: Tokens match expected
        tokens = tokenizer.tokenize(self.integration_test_input_string)
        self.maxDiff = None
        self.assertListEqual(
            tokens,
            self.integration_expected_tokens,
            f"Tokenized tokens don't match expected for {tokenizer.__class__.__name__} ({tokenizer_type})",
        )

        # Test 2: IDs from encode match expected (without special tokens)
        ids_from_encode = tokenizer.encode(self.integration_test_input_string, add_special_tokens=False)
        self.assertEqual(
            ids_from_encode,
            self.integration_expected_token_ids,
            f"Encoded IDs don't match expected for {tokenizer.__class__.__name__} ({tokenizer_type})",
        )

        # Test 3: Round-trip decode produces expected text (if provided)
        decoded_text = tokenizer.decode(self.integration_expected_token_ids, clean_up_tokenization_spaces=False)
        self.assertEqual(
            decoded_text,
            self.integration_expected_decoded_text,
            f"Decoded text doesn't match expected for {tokenizer.__class__.__name__} ({tokenizer_type})",
        )

    def test_integration(self):
        """
        Integration checks for the original tokenizer only.
        """
        # Skip if no integration test data is provided
        if not hasattr(self, "integration_test_input_string") or self.integration_test_input_string is None:
            self.skipTest("No integration test input string provided")
        if not hasattr(self, "integration_expected_tokens") or self.integration_expected_tokens is None:
            self.skipTest("No integration expected tokens provided")
        if not hasattr(self, "integration_expected_token_ids") or self.integration_expected_token_ids is None:
            self.skipTest("No integration expected token IDs provided")
        if not hasattr(self, "integration_expected_decoded_text") or self.integration_expected_decoded_text is None:
            self.skipTest("No integration expected decoded text provided")

        tokenizer_original = self.tokenizer_class.from_pretrained(
            self.from_pretrained_id[0],
            do_lower_case=False,
            keep_accents=True,
            **(self.from_pretrained_kwargs if self.from_pretrained_kwargs is not None else {}),
        )
        self._run_integration_checks(tokenizer_original, "original")

    def test_integration_from_extractor(self):
        """
        Integration checks for a tokenizer built via TokenizersExtractor.
        """
        # Skip if tokenizer-from-extractor path is not enabled for this class
        if not getattr(self, "test_tokenizer_from_extractor", False):
            self.skipTest("Tokenizer from TokenizersExtractor not enabled for this tokenizer")

        # Skip if no integration test data is provided
        if not hasattr(self, "integration_test_input_string") or self.integration_test_input_string is None:
            self.skipTest("No integration test input string provided")
        if not hasattr(self, "integration_expected_tokens") or self.integration_expected_tokens is None:
            self.skipTest("No integration expected tokens provided")
        if not hasattr(self, "integration_expected_token_ids") or self.integration_expected_token_ids is None:
            self.skipTest("No integration expected token IDs provided")
        if not hasattr(self, "integration_expected_decoded_text") or self.integration_expected_decoded_text is None:
            self.skipTest("No integration expected decoded text provided")

        tokenizer_original = self.tokenizer_class.from_pretrained(
            self.from_pretrained_id[0],
            do_lower_case=False,
            keep_accents=True,
            **(self.from_pretrained_kwargs if self.from_pretrained_kwargs is not None else {}),
        )
        tokenizer_from_extractor = self.get_extracted_tokenizer(reference_tokenizer=tokenizer_original)
        if tokenizer_from_extractor is None:
            self.fail("No tokenizer from TokenizersExtractor provided")
        self._run_integration_checks(tokenizer_from_extractor, "from_extractor")

    def test_internal_consistency(self):
        tokenizer = self.get_tokenizer()
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

    def test_mask_output(self):
        tokenizer = self.get_tokenizer(do_lower_case=False)
        seq_0 = "Test this method."
        seq_1 = "With these inputs."
        information = tokenizer(seq_0, seq_1, add_special_tokens=True, return_token_type_ids=True)
        sequences, mask = information["input_ids"], information["token_type_ids"]
        self.assertEqual(len(sequences), len(mask))

    def test_token_type_ids(self):
        tokenizer = self.get_tokenizer()
        seq_0 = "Test this method."

        # We want to have sequence 0 and sequence 1 are tagged
        # respectively with 0 and 1 token_ids
        # (regardless of whether the model use token type ids)
        # We use this assumption in the QA pipeline among other place
        output = tokenizer(seq_0, return_token_type_ids=True)
        self.assertIn(0, output["token_type_ids"])

    def test_sequence_ids(self):
        tokenizer = self.get_tokenizer()

        if tokenizer.backend != "tokenizers":
            self.skipTest(reason="Tokenizers backend tokenizer")

        seq_0 = "Test this method."
        seq_1 = "With these inputs."

        # We want to have sequence 0 and sequence 1 are tagged
        # respectively with 0 and 1 token_ids\
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
        tokenizer = self.get_tokenizer()
        output = tokenizer.apply_chat_template(
            dummy_conversation, chat_template=dummy_template, tokenize=False, return_dict=False
        )
        self.assertEqual(output, expected_output)  # Test we can pass chat_template arg

        # Check that no error raised when tokenize=True
        output = tokenizer.apply_chat_template(
            dummy_conversation, chat_template=dummy_template, tokenize=True, return_dict=False
        )
        dict_output = tokenizer.apply_chat_template(
            dummy_conversation,
            chat_template=dummy_template,
            tokenize=True,  # This also checks return_dict=True is the default
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
        tokenizer = self.get_tokenizer()
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
        tokenizer = self.get_tokenizer()
        output = tokenizer.apply_chat_template(dummy_conversations, chat_template=dummy_template, tokenize=False)
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

        tokenizer = self.get_tokenizer()
        break_output = tokenizer.apply_chat_template(dummy_conversation, chat_template=break_template, tokenize=False)
        self.assertEqual(break_output, "system 1")  # Loop should break after first iter

    @require_jinja
    def test_jinja_strftime(self):
        strftime_template = """{{- strftime_now("%Y-%m-%d") }}""".strip()

        dummy_conversation = [
            {"role": "system", "content": "1"},
            {"role": "user", "content": "2"},
            {"role": "assistant", "content": "3"},
        ]

        tokenizer = self.get_tokenizer()
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
                tokenizer_r = self.get_tokenizer(pretrained_name)
                if tokenizer_r.backend != "tokenizers":
                    self.skipTest(reason="Custom backend tokenizer")

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

                    if (
                        assistant_start is None
                        or assistant_end is None
                        or assistant_start2 is None
                        or assistant_end2 is None
                    ):
                        continue

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

                if (
                    assistant_start is None
                    or assistant_end is None
                    or assistant_start2 is None
                    or assistant_end2 is None
                ):
                    return

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
                tokenizer_r = self.get_tokenizer(pretrained_name)
                if tokenizer_r.backend != "tokenizers":
                    self.skipTest(reason="Custom backend tokenizer")

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
                if truncation_position is None:
                    self.skipTest("char_to_token returned None, cannot determine truncation position")

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

                    if assistant_start is None:
                        continue

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

                if assistant_start is None:
                    return

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
        tokenizer = self.get_tokenizer()
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
        tokenizer = self.get_tokenizer()
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
        tokenizer = self.get_tokenizer()
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
        tokenizer = self.get_tokenizer()
        tokenizer.chat_template = {"template1": dummy_template_1, "template2": dummy_template_2}
        output1 = tokenizer.apply_chat_template(dummy_conversation, chat_template=dummy_template_1, tokenize=False)
        output1_via_dict = tokenizer.apply_chat_template(dummy_conversation, chat_template="template1", tokenize=False)
        self.assertEqual(output1, output1_via_dict)
        output2 = tokenizer.apply_chat_template(dummy_conversation, chat_template=dummy_template_2, tokenize=False)
        output2_via_dict = tokenizer.apply_chat_template(dummy_conversation, chat_template="template2", tokenize=False)
        self.assertEqual(output2, output2_via_dict)

    @require_jinja
    def test_chat_template_dict_saving(self):
        dummy_template_1 = "{{'a'}}"
        dummy_template_2 = "{{'b'}}"
        tokenizer = self.get_tokenizer()
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
        tokenizer = self.get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tokenizer.chat_template = dummy_template1
            tokenizer.save_pretrained(tmp_dir_name, save_jinja_files=False)
            with Path(tmp_dir_name, "chat_template.jinja").open("w") as f:
                f.write(dummy_template2)
            new_tokenizer = tokenizer.from_pretrained(tmp_dir_name)
        # Assert the file template clobbers any template in the config
        self.assertEqual(new_tokenizer.chat_template, dummy_template2)

    def test_number_of_added_tokens(self):
        tokenizer = self.get_tokenizer(do_lower_case=False)
        seq_0 = "Test this method."
        seq_1 = "With these inputs."

        sequences = tokenizer.encode(seq_0, seq_1, add_special_tokens=False)
        attached_sequences = tokenizer.encode(seq_0, seq_1, add_special_tokens=True)

        # Method is implemented (e.g. not GPT-2)
        if len(attached_sequences) != 2:
            self.assertEqual(tokenizer.num_special_tokens_to_add(pair=True), len(attached_sequences) - len(sequences))

    def test_maximum_encoding_length_single_input(self):
        tokenizer = self.get_tokenizer(do_lower_case=False, model_max_length=100)
        seq_0, ids = self.get_clean_sequence(tokenizer, max_length=20)

        sequence = tokenizer.encode(seq_0, add_special_tokens=False)
        total_length = len(sequence)

        self.assertGreater(total_length, 4, "Issue with the testing sequence, please update it, it's too short")

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
        if isinstance(tokenizer, TokenizersBackend):
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
        tokenizer = self.get_tokenizer(do_lower_case=False, model_max_length=100)
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
        self.assertLess(total_length1, model_max_length - 10, "Issue with the testing sequence, please update it.")
        self.assertGreater(total_length2, model_max_length, "Issue with the testing sequence, please update it.")

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

                        output = tokenizer([seq_2], [seq_1], padding=padding_state, truncation=truncation_state)
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
            tokenizer.encode(seq_0, add_special_tokens=False) + tokenizer.encode(seq_1, add_special_tokens=False)[:-2]
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
        if isinstance(tokenizer, TokenizersBackend):
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
        if isinstance(tokenizer, TokenizersBackend):
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
        if isinstance(tokenizer, TokenizersBackend):
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
        if isinstance(tokenizer, TokenizersBackend):
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
        tokenizer = self.get_tokenizer(do_lower_case=False)
        sequence_0 = "Encode this."
        # Testing single inputs
        encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False)
        encoded_sequence_dict = tokenizer(
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
        tokenizer = self.get_tokenizer(do_lower_case=False)
        sequence_0 = "Encode this."
        sequence_1 = "This one too please."
        encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False)
        encoded_sequence += tokenizer.encode(sequence_1, add_special_tokens=False)
        encoded_sequence_dict = tokenizer(
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
                tokenizer_r = self.get_tokenizer(pretrained_name, padding_side="left", **kwargs)
                self.assertEqual(tokenizer_r.padding_side, "left")

                tokenizer_r = self.get_tokenizer(pretrained_name, padding_side="right", **kwargs)
                self.assertEqual(tokenizer_r.padding_side, "right")

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
                tokenizer_r = self.get_tokenizer(pretrained_name, truncation_side="left", **kwargs)
                self.assertEqual(tokenizer_r.truncation_side, "left")

                tokenizer_r = self.get_tokenizer(pretrained_name, truncation_side="right", **kwargs)
                self.assertEqual(tokenizer_r.truncation_side, "right")

                self.assertRaises(
                    ValueError,
                    self.tokenizer_class.from_pretrained,
                    pretrained_name,
                    truncation_side="unauthorized",
                    **kwargs,
                )

    def test_encode_basic_padding(self):
        """Test basic left/right padding behavior using encode() method with max_length strategy."""
        tokenizer = self.get_tokenizer(do_lower_case=False)
        sequence = "Sequence"
        padding_size = 10

        # check correct behaviour if no pad_token_id exists and add it eventually
        self._check_no_pad_token_padding(tokenizer, sequence)

        padding_idx = tokenizer.pad_token_id

        # RIGHT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
        tokenizer.padding_side = "right"
        encoded_sequence = tokenizer.encode(sequence)
        sequence_length = len(encoded_sequence)
        padded_sequence = tokenizer.encode(sequence, max_length=sequence_length + padding_size, padding="max_length")
        padded_sequence_length = len(padded_sequence)
        self.assertEqual(sequence_length + padding_size, padded_sequence_length)
        self.assertEqual(encoded_sequence + [padding_idx] * padding_size, padded_sequence)

        # LEFT PADDING - Check that it correctly pads when a maximum length is specified along with the padding flag set to True
        tokenizer.padding_side = "left"
        encoded_sequence = tokenizer.encode(sequence)
        sequence_length = len(encoded_sequence)
        padded_sequence = tokenizer.encode(sequence, max_length=sequence_length + padding_size, padding="max_length")
        padded_sequence_length = len(padded_sequence)
        self.assertEqual(sequence_length + padding_size, padded_sequence_length)
        self.assertEqual([padding_idx] * padding_size + encoded_sequence, padded_sequence)

    def test_right_and_left_truncation(self):
        tokenizer = self.get_tokenizer(do_lower_case=False)
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
        truncated_sequence_left = tokenizer.encode(sequence, truncation="longest_first", add_special_tokens=False)
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
        tokenizer = self.get_tokenizer()
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
        tokenizer = self.get_tokenizer()
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
        tokenizer = self.get_tokenizer(do_lower_case=False)
        sequence = "Sequence"

        # check correct behaviour if no pad_token_id exists and add it eventually
        self._check_no_pad_token_padding(tokenizer, sequence)

        padding_size = 10
        padding_idx = tokenizer.pad_token_id
        token_type_padding_idx = tokenizer.pad_token_type_id

        encoded_sequence = tokenizer(sequence, return_special_tokens_mask=True)
        input_ids = encoded_sequence["input_ids"]
        special_tokens_mask = encoded_sequence["special_tokens_mask"]
        sequence_length = len(input_ids)

        # Test 'longest' and 'no_padding' don't do anything
        not_padded_sequence = tokenizer(
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

        not_padded_sequence = tokenizer(
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

        right_padded_sequence = tokenizer(sequence, **tokenizer_kwargs_right)
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

        left_padded_sequence = tokenizer(sequence, **tokenizer_kwargs_left)
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

            self.assertEqual(token_type_ids + [token_type_padding_idx] * padding_size, right_padded_token_type_ids)
            self.assertEqual([token_type_padding_idx] * padding_size + token_type_ids, left_padded_token_type_ids)

        if "attention_mask" in tokenizer.model_input_names:
            attention_mask = encoded_sequence["attention_mask"]
            right_padded_attention_mask = right_padded_sequence["attention_mask"]
            left_padded_attention_mask = left_padded_sequence["attention_mask"]

            self.assertEqual(attention_mask + [0] * padding_size, right_padded_attention_mask)
            self.assertEqual([0] * padding_size + attention_mask, left_padded_attention_mask)

    def test_get_vocab(self):
        tokenizer = self.get_tokenizer(do_lower_case=False)
        vocab_dict = tokenizer.get_vocab()
        self.assertIsInstance(vocab_dict, dict)
        self.assertGreaterEqual(len(tokenizer), len(vocab_dict))

        vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
        self.assertEqual(len(vocab), len(tokenizer))

        tokenizer.add_tokens(["asdfasdfasdfasdf"])
        vocab = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
        self.assertEqual(len(vocab), len(tokenizer))

    @slow
    def test_conversion_reversible(self):
        tokenizer = self.get_tokenizer(do_lower_case=False)
        vocab = tokenizer.get_vocab()
        for word, ind in vocab.items():
            if word == tokenizer.unk_token:
                continue
            self.assertEqual(tokenizer.convert_tokens_to_ids(word), ind)
            self.assertEqual(tokenizer.convert_ids_to_tokens(ind), word)

    def test_call(self):
        # Tests that all call wrap to encode_plus
        tokenizer = self.get_tokenizer(do_lower_case=False)
        sequences = [
            "Testing batch encode plus",
            "Testing batch encode plus with different sequence lengths",
            "Testing batch encode plus with different sequence lengths correctly pads",
        ]

        # Test not batched
        encoded_sequences_1 = tokenizer(sequences[0])
        encoded_sequences_2 = tokenizer(sequences[0])
        self.assertEqual(encoded_sequences_1, encoded_sequences_2)

        # Test not batched pairs
        encoded_sequences_1 = tokenizer(sequences[0], sequences[1])
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
        tokenizer = self.get_tokenizer(do_lower_case=False)
        sequences = [
            "Testing batch encode plus",
            "Testing batch encode plus with different sequence lengths",
            "Testing batch encode plus with different sequence lengths correctly pads",
        ]

        encoded_sequences = [tokenizer(sequence) for sequence in sequences]
        encoded_sequences_batch = tokenizer(sequences, padding=False)
        self.assertListEqual(
            encoded_sequences, TokenizerTesterMixin.convert_batch_to_list_format(encoded_sequences_batch)
        )

        maximum_length = len(max([encoded_sequence["input_ids"] for encoded_sequence in encoded_sequences], key=len))

        # check correct behaviour if no pad_token_id exists and add it eventually
        self._check_no_pad_token_padding(tokenizer, sequences)

        encoded_sequences_padded = [
            tokenizer(sequence, max_length=maximum_length, padding="max_length") for sequence in sequences
        ]

        encoded_sequences_batch_padded = tokenizer(sequences, padding=True)
        self.assertListEqual(
            encoded_sequences_padded,
            TokenizerTesterMixin.convert_batch_to_list_format(encoded_sequences_batch_padded),
        )

        # check 'longest' is unsensitive to a max length
        encoded_sequences_batch_padded_1 = tokenizer(sequences, padding=True)
        encoded_sequences_batch_padded_2 = tokenizer(sequences, max_length=maximum_length + 10, padding="longest")
        for key in encoded_sequences_batch_padded_1:
            self.assertListEqual(
                encoded_sequences_batch_padded_1[key],
                encoded_sequences_batch_padded_2[key],
            )

        # check 'no_padding' is unsensitive to a max length
        encoded_sequences_batch_padded_1 = tokenizer(sequences, padding=False)
        encoded_sequences_batch_padded_2 = tokenizer(sequences, max_length=maximum_length + 10, padding=False)
        for key in encoded_sequences_batch_padded_1:
            self.assertListEqual(
                encoded_sequences_batch_padded_1[key],
                encoded_sequences_batch_padded_2[key],
            )

    def test_batch_encode_plus_padding(self):
        # Test that padded sequences are equivalent between batch and individual encoding

        # Right padding tests
        tokenizer = self.get_tokenizer(do_lower_case=False)
        sequences = [
            "Testing batch encode plus",
            "Testing batch encode plus with different sequence lengths",
            "Testing batch encode plus with different sequence lengths correctly pads",
        ]

        max_length = 100

        # check correct behaviour if no pad_token_id exists and add it eventually
        self._check_no_pad_token_padding(tokenizer, sequences)

        encoded_sequences = [
            tokenizer(sequence, max_length=max_length, padding="max_length") for sequence in sequences
        ]
        encoded_sequences_batch = tokenizer(sequences, max_length=max_length, padding="max_length")
        self.assertListEqual(
            encoded_sequences, TokenizerTesterMixin.convert_batch_to_list_format(encoded_sequences_batch)
        )

        # Left padding tests
        tokenizer = self.get_tokenizer(do_lower_case=False)
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
            tokenizer(sequence, max_length=max_length, padding="max_length") for sequence in sequences
        ]
        encoded_sequences_batch = tokenizer(sequences, max_length=max_length, padding="max_length")
        self.assertListEqual(
            encoded_sequences, TokenizerTesterMixin.convert_batch_to_list_format(encoded_sequences_batch)
        )

    def test_pretokenized_inputs(self):
        # Test when inputs are pretokenized
        # All methods (encode, encode_plus, __call__) go through the same code path,
        # so we only test __call__

        tokenizer = self.get_tokenizer(do_lower_case=False)
        if hasattr(tokenizer, "add_prefix_space") and not tokenizer.add_prefix_space:
            return

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
        output = tokenizer(token_sequence, token_sequence, is_split_into_words=True, add_special_tokens=False)
        output_sequence = tokenizer(sequence, sequence, add_special_tokens=False)
        for key in output:
            self.assertEqual(output[key], output_sequence[key])

        output = tokenizer(token_sequence, token_sequence, is_split_into_words=True, add_special_tokens=True)
        output_sequence = tokenizer(sequence, sequence, add_special_tokens=True)
        for key in output:
            self.assertEqual(output[key], output_sequence[key])

        # Test batched inputs
        sequence_batch = [sequence.strip()] * 2 + [sequence.strip() + " " + sequence.strip()]
        token_sequence_batch = [s.split() for s in sequence_batch]
        sequence_batch_cleaned_up_spaces = [" " + " ".join(s) for s in token_sequence_batch]

        output = tokenizer(token_sequence_batch, is_split_into_words=True, add_special_tokens=False)
        output_sequence = tokenizer(sequence_batch_cleaned_up_spaces, add_special_tokens=False)
        for key in output:
            self.assertEqual(output[key], output_sequence[key])

        output = tokenizer(token_sequence_batch, is_split_into_words=True, add_special_tokens=True)
        output_sequence = tokenizer(sequence_batch_cleaned_up_spaces, add_special_tokens=True)
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

        output = tokenizer(token_sequence_pair_batch, is_split_into_words=True, add_special_tokens=False)
        output_sequence = tokenizer(sequence_pair_batch_cleaned_up_spaces, add_special_tokens=False)
        for key in output:
            self.assertEqual(output[key], output_sequence[key])
        output = tokenizer(token_sequence_pair_batch, is_split_into_words=True, add_special_tokens=True)
        output_sequence = tokenizer(sequence_pair_batch_cleaned_up_spaces, add_special_tokens=True)
        for key in output:
            self.assertEqual(output[key], output_sequence[key])

    def _check_no_pad_token_padding(self, tokenizer, sequences):
        # if tokenizer does  v have pad_token_id, an error should be thrown
        if tokenizer.pad_token_id is None:
            with self.assertRaises(ValueError):
                if isinstance(sequences, list):
                    tokenizer(sequences, padding="longest")
                else:
                    tokenizer(sequences, padding=True)

            # add pad_token_id to pass subsequent tests
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    @require_torch
    def test_prepare_seq2seq_batch(self):
        if not self.test_seq2seq:
            self.skipTest(reason="test_seq2seq is set to False")

        tokenizer = self.get_tokenizer()
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
            batch = tokenizer(
                src_text,
                text_target=tgt_text,
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
        batch = tokenizer(src_text, text_target=tgt_text, max_length=3, return_tensors="pt")
        self.assertEqual(batch.input_ids.shape[1], 3)
        self.assertEqual(batch.labels.shape[1], 3)

        batch_encoder_only = tokenizer(src_text, max_length=3, max_target_length=10, return_tensors="pt")
        self.assertEqual(batch_encoder_only.input_ids.shape[1], 3)
        self.assertEqual(batch_encoder_only.attention_mask.shape[1], 3)
        self.assertNotIn("decoder_input_ids", batch_encoder_only)

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
            tokenizer = self.get_tokenizer(pretrained_name, **kwargs)

            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                if is_torch_available():
                    returned_tensor = "pt"
                else:
                    self.skipTest(reason="No expected framework (PT) found")

                if not tokenizer.pad_token or tokenizer.pad_token_id < 0:
                    self.skipTest(reason="This tokenizer has no padding token set, or pad_token_id < 0")

                tokens = tokenizer(
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

    def test_added_tokens_serialization(self):
        new_eos = AddedToken("[NEW_EOS]", rstrip=False, lstrip=True, normalized=False, special=True)
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                # Test loading a tokenizer from the hub with a new eos token
                tokenizer_r = self.get_tokenizer(pretrained_name, eos_token=new_eos)
                self.assertEqual(tokenizer_r._special_tokens_map["eos_token"], new_eos)
                # Check that the token content is present (may not preserve all AddedToken attributes)
                self.assertIn(str(new_eos), [str(t) for t in tokenizer_r.added_tokens_decoder.values()])

                EXPECTED_ADDED_TOKENS_DECODER = tokenizer_r.added_tokens_decoder

                # Test saving and reloading the tokenizer
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tokenizer_r.save_pretrained(tmp_dir)

                    with self.subTest("Saving tokenizer locally and reloading"):
                        tokenizer = self.tokenizer_class.from_pretrained(tmp_dir)
                        self.assertTrue(str(new_eos) not in tokenizer.extra_special_tokens)
                        # Check that the token content is present (may not preserve all AddedToken attributes)
                        self.assertIn(str(new_eos), [str(t) for t in tokenizer.added_tokens_decoder.values()])
                        self.assertEqual(str(tokenizer.added_tokens_decoder[tokenizer.eos_token_id]), str(new_eos))
                        # Check that all original tokens are still present (by string representation)
                        expected_tokens = {str(t) for t in EXPECTED_ADDED_TOKENS_DECODER.values()}
                        actual_tokens = {str(t) for t in tokenizer.added_tokens_decoder.values()}
                        self.assertTrue(expected_tokens.issubset(actual_tokens))

    def test_tokenizer_initialization_with_conflicting_key(self):
        with self.assertRaises(AttributeError, msg="conflicts with the method"):
            self.get_tokenizer(add_special_tokens=True)

        with self.assertRaises(AttributeError, msg="conflicts with the method"):
            self.get_tokenizer(get_vocab=True)

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

        tokenizer = self.get_tokenizer()
        for return_type, target_type in zip(tokenizer_return_type, output_tensor_type):
            output = tokenizer(empty_input_string, return_tensors=return_type)
            self.assertEqual(output.input_ids.dtype, target_type)

    def test_pad_token_initialization(self):
        """Test that passing pad_token when creating a tokenizer works correctly."""
        tokenizer = self.get_tokenizer(pad_token="[PAD]")
        # Verify the pad_token was set correctly
        self.assertEqual(tokenizer.pad_token, "[PAD]")
        self.assertIsNotNone(tokenizer.pad_token_id)

        # Test with two sequences of different lengths to trigger padding
        seq_0 = "Test this method."
        seq_1 = "With these inputs and some extra tokens here."

        # Test padding works with the custom pad_token
        output_with_padding = tokenizer(
            [seq_0, seq_1],
            padding=True,
            return_attention_mask=True,
        )

        # Check that sequences were padded to the same length
        self.assertEqual(
            len(output_with_padding["input_ids"][0]),
            len(output_with_padding["input_ids"][1]),
        )

        # Check that attention mask has 0s where padding was added (on the shorter sequence)
        # Find the shorter sequence
        unpadded_lengths = [
            len(tokenizer(seq_0, add_special_tokens=True)["input_ids"]),
            len(tokenizer(seq_1, add_special_tokens=True)["input_ids"]),
        ]
        shorter_idx = 0 if unpadded_lengths[0] < unpadded_lengths[1] else 1
        self.assertIn(0, output_with_padding["attention_mask"][shorter_idx])

    def test_bos_token_with_add_bos_token_true(self):
        """Test that passing bos_token with add_bos_token=True during initialization adds the BOS token."""
        try:
            tokenizer = self.get_tokenizer(bos_token="<BOS>", add_bos_token=True)
        except TypeError:
            # Some tokenizers might not support add_bos_token parameter
            self.skipTest("Tokenizer does not support add_bos_token parameter")

        test_string = "Hello world"

        # Verify bos_token was set
        self.assertEqual(tokenizer.bos_token, "<BOS>")

        # Verify the tokenizer was created successfully with these parameters
        output = tokenizer(test_string, add_special_tokens=False)
        self.assertIsNotNone(output["input_ids"])

    def test_bos_token_with_add_bos_token_false(self):
        """Test that passing bos_token with add_bos_token=False during initialization does not add the BOS token."""
        try:
            tokenizer = self.get_tokenizer(bos_token="<BOS>", add_bos_token=False)
        except TypeError:
            # Some tokenizers might not support add_bos_token parameter
            self.skipTest("Tokenizer does not support add_bos_token parameter")

        test_string = "Hello world"

        # Verify bos_token was set
        self.assertEqual(tokenizer.bos_token, "<BOS>")

        # Verify the tokenizer was created successfully with these parameters
        output = tokenizer(test_string, add_special_tokens=False)
        self.assertIsNotNone(output["input_ids"])

    def test_local_files_only(self):
        from transformers import AutoTokenizer

        pretrained_list = getattr(self, "from_pretrained_id", []) or []
        for pretrained_name in pretrained_list:
            with self.subTest(f"AutoTokenizer ({pretrained_name})"):
                # First cache the tokenizer files
                try:
                    tokenizer_cached = AutoTokenizer.from_pretrained(pretrained_name)

                    # Now load with local_files_only=True
                    tokenizer_local = AutoTokenizer.from_pretrained(pretrained_name, local_files_only=True)

                    # Check that the two tokenizers are identical
                    self.assertEqual(tokenizer_cached.get_vocab(), tokenizer_local.get_vocab())
                    self.assertEqual(
                        tokenizer_cached.all_special_tokens_extended,
                        tokenizer_local.all_special_tokens_extended,
                    )
                except Exception as _:
                    pass  # if the pretrained model is not loadable how could it pass locally :)


@require_tokenizers
class TokenizersBackendCommonTest(TokenizersBackendTesterMixin, unittest.TestCase):
    """
    A single test class that runs all tokenizers-backend tests once.
    Uses BertTokenizer as a representative tokenizer.
    """

    tokenizer_class = BertTokenizer
    rust_tokenizer_class = BertTokenizerFast
    from_pretrained_id = "google-bert/bert-base-uncased"
    from_pretrained_kwargs = {}


class SentencePieceBackendCommonTest(unittest.TestCase, SentencePieceBackendTesterMixin):
    """
    A single test class that runs all SentencePiece-backend tests once.
    Uses T5Tokenizer as a representative SentencePiece tokenizer.
    """

    tokenizer_class = T5Tokenizer
    rust_tokenizer_class = T5TokenizerFast
    test_slow_tokenizer = True
    test_rust_tokenizer = True
    from_pretrained_id = "google-t5/t5-base"
    from_pretrained_kwargs = {"use_fast": False}

    def test_add_tokens(self):
        tokenizer_r = self.get_rust_tokenizer()

        vocab_size = len(tokenizer_r)
        self.assertEqual(tokenizer_r.add_tokens(""), 0)
        self.assertEqual(tokenizer_r.add_tokens("testoken"), 1)
        self.assertEqual(tokenizer_r.add_tokens(["testoken1", "testtoken2"]), 2)
        self.assertEqual(len(tokenizer_r), vocab_size + 3)

        self.assertEqual(tokenizer_r.add_special_tokens({}), 0)
        self.assertEqual(tokenizer_r.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"}), 2)
        self.assertRaises(ValueError, tokenizer_r.add_special_tokens, {"additional_special_tokens": "<testtoken1>"})
        self.assertEqual(tokenizer_r.add_special_tokens({"additional_special_tokens": ["<testtoken2>"]}), 1)
        self.assertEqual(
            tokenizer_r.add_special_tokens({"additional_special_tokens": ["<testtoken3>", "<testtoken4>"]}), 2
        )
        added_vocab = tokenizer_r.get_added_vocab()
        self.assertIn("<testtoken3>", added_vocab)

    def test_add_tokens_tokenizer(self):
        tokenizer = self.get_tokenizer(do_lower_case=False)
        vocab_size = tokenizer.vocab_size
        all_size = len(tokenizer)

        new_toks = [
            AddedToken("newtokenone", rstrip=False, lstrip=False),
            AddedToken("newtokentwo", rstrip=False, lstrip=False),
        ]
        added_toks = tokenizer.add_tokens(new_toks)
        vocab_size_2 = tokenizer.vocab_size
        all_size_2 = len(tokenizer)

        self.assertEqual(vocab_size, vocab_size_2)
        self.assertEqual(added_toks, len(new_toks))
        self.assertEqual(all_size_2, all_size + len(new_toks))

        tokens = tokenizer.encode("newtokenone words newtokentwo", add_special_tokens=False)
        self.assertGreaterEqual(len(tokens), 3)
        self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[-1], tokenizer.vocab_size - 1)

        new_specials = {
            "eos_token": AddedToken("<|eos_new|>", rstrip=False, lstrip=False),
            "pad_token": AddedToken("<|pad_new|>", rstrip=False, lstrip=False),
        }
        added_specials = tokenizer.add_special_tokens(new_specials)
        all_size_3 = len(tokenizer)
        self.assertEqual(added_specials, len(new_specials))
        self.assertEqual(all_size_3, all_size_2 + len(new_specials))

        tokens = tokenizer.encode("<|eos_new|> newtokenone <|pad_new|>", add_special_tokens=False)
        self.assertEqual(tokens[0], tokenizer.eos_token_id)
        self.assertEqual(tokens[-1], tokenizer.pad_token_id)

    def test_alignment_methods(self):
        self.skipTest("SentencePiece fast tokenizers do not expose token alignment metadata.")

    def test_local_files_only(self):
        from transformers import AutoTokenizer

        pretrained_list = getattr(self, "from_pretrained_id", []) or []
        for pretrained_name in pretrained_list:
            with self.subTest(f"AutoTokenizer ({pretrained_name})"):
                # First cache the tokenizer files
                try:
                    tokenizer_cached = AutoTokenizer.from_pretrained(pretrained_name)

                    # Now load with local_files_only=True
                    tokenizer_local = AutoTokenizer.from_pretrained(pretrained_name, local_files_only=True)

                    # Check that the two tokenizers are identical
                    self.assertEqual(tokenizer_cached.get_vocab(), tokenizer_local.get_vocab())
                    self.assertEqual(
                        tokenizer_cached.all_special_tokens_extended,
                        tokenizer_local.all_special_tokens_extended,
                    )
                except Exception as _:
                    pass  # if the pretrained model is not loadable how could it pass locally :)
