# Copyright 2025 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

import tempfile
import unittest

import numpy as np
import torch

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.testing_utils import require_mistral_common
from transformers.tokenization_mistral_common import MistralCommonTokenizer
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy
from transformers.utils import PaddingStrategy, is_mistral_common_available


if is_mistral_common_available():
    from mistral_common.exceptions import InvalidMessageStructureException
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


IMG_URL = "https://picsum.photos/id/237/200/300"
IMG_BASE_64 = """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"""


@require_mistral_common
class TestMistralCommonTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tokenizer: MistralCommonTokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/namespace-mistralai-repo_name-Mistral-Small-3.1-24B-Instruct-2503",
            tokenizer_type="mistral",
        )
        cls.ref_tokenizer: MistralTokenizer = MistralTokenizer.from_hf_hub(
            "hf-internal-testing/namespace-mistralai-repo_name-Mistral-Small-3.1-24B-Instruct-2503"
        )
        cls.fixture_conversations = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi!"},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi!"},
                {"role": "assistant", "content": "Hello! How can I help you?"},
                {"role": "user", "content": "What is the temperature in Paris?"},
            ],
        ]
        cls.tokenized_fixture_conversations = [
            cls.ref_tokenizer.encode_chat_completion(ChatCompletionRequest.from_openai(conversation))
            for conversation in cls.fixture_conversations
        ]

        cls.ref_special_ids = {t["rank"] for t in cls.ref_tokenizer.instruct_tokenizer.tokenizer._all_special_tokens}

    def _ref_piece_to_id(self, piece: str) -> int:
        pieces = self.ref_tokenizer.instruct_tokenizer.tokenizer._model.encode(
            piece, allowed_special="all", disallowed_special=set()
        )
        assert len(pieces) == 1, f"Expected to decode 1 token, got {len(pieces)}"
        return pieces[0]

    def test_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, self.ref_tokenizer.instruct_tokenizer.tokenizer.n_words)

    def test_save_pretrained(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = self.tokenizer.save_pretrained(tmp_dir)[0]
            loaded_tokenizer = MistralCommonTokenizer.from_pretrained(tmp_file)

        self.assertIsNotNone(loaded_tokenizer)
        self.assertEqual(self.tokenizer.get_vocab(), loaded_tokenizer.get_vocab())
        self.assertEqual(
            self.tokenizer.tokenizer.instruct_tokenizer.tokenizer.version,
            loaded_tokenizer.tokenizer.instruct_tokenizer.tokenizer.version,
        )

        with self.assertRaises(
            ValueError, msg="Kwargs [unk_args] are not supported by `MistralCommonTokenizer.save_pretrained`."
        ):
            with tempfile.TemporaryDirectory() as tmp_dir:
                self.tokenizer.save_pretrained(tmp_dir, unk_args="")

    def test_encode(self):
        string = "Hello, world!"

        # Test 1:
        # encode with add_special_tokens
        expected_with_special = self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(string, bos=True, eos=True)
        tokens_with_special = self.tokenizer.encode(string, add_special_tokens=True)
        self.assertEqual(tokens_with_special, expected_with_special)

        # Test 2:
        # encode without add_special_tokens
        expected_without_special = self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(string, bos=False, eos=False)
        tokens_without_special = self.tokenizer.encode(string, add_special_tokens=False)
        self.assertEqual(tokens_without_special, expected_without_special)

        # Test 3:
        # encode with return_tensors
        tokens_with_return_tensors = self.tokenizer.encode(string, add_special_tokens=False, return_tensors="pt")
        self.assertIsInstance(tokens_with_return_tensors, torch.Tensor)
        self.assertEqual(tokens_with_return_tensors.tolist()[0], expected_without_special)

        # Test 4:
        # encode with max_length
        tokens_with_max_length = self.tokenizer.encode(string, add_special_tokens=False, max_length=3)
        self.assertEqual(tokens_with_max_length, expected_without_special[:3])

        # Test 5:
        # encode with padding
        tokens_with_padding = self.tokenizer.encode(
            string, add_special_tokens=False, padding=True, pad_to_multiple_of=6
        )
        expected_padding = [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id] * (
            6 - len(expected_without_special) % 6
        ) + expected_without_special
        self.assertEqual(tokens_with_padding, expected_padding)

        for padding in [
            False,
            True,
            "longest",
            "max_length",
            "do_not_pad",
            PaddingStrategy.LONGEST,
            PaddingStrategy.MAX_LENGTH,
            PaddingStrategy.DO_NOT_PAD,
        ]:
            tokens_with_padding = self.tokenizer.encode(string, add_special_tokens=False, padding=padding)
            self.assertEqual(tokens_with_padding, expected_without_special)

        # For truncation, we use a longer string
        string_long = (
            "Hello world! It is a beautiful day today. The sun is shining brightly and the birds are singing."
        )
        expected_long = self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(string_long, bos=False, eos=False)

        # Test 6:
        # encode with truncation
        tokens_with_truncation = self.tokenizer.encode(
            string_long, add_special_tokens=False, truncation=True, max_length=12
        )
        self.assertEqual(tokens_with_truncation, expected_long[:12])

        # Test 7:
        # encode with padding and truncation
        tokens_with_padding_and_truncation = self.tokenizer.encode(
            string_long, add_special_tokens=False, padding=True, pad_to_multiple_of=12, truncation=True, max_length=36
        )
        expected_long_padding = [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id] * (
            12 - len(expected_long) % 12
        ) + expected_long
        self.assertEqual(tokens_with_padding_and_truncation, expected_long_padding)

        # Test encode with unsupported kwargs
        with self.assertRaises(
            ValueError, msg="Kwargs [unk_args] are not supported by `MistralCommonTokenizer.encode`."
        ):
            self.tokenizer.encode("Hello, world!", add_special_tokens=True, unk_args="")

    def test_decode(self):
        string = "Hello, world!"
        string_with_space = "Hello, world !"

        tokens_ids = self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(string, bos=True, eos=True)
        tokens_ids_with_space = self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(
            string_with_space, bos=True, eos=True
        )

        # Test 1:
        # decode with and without skip_special_tokens
        self.assertEqual(self.tokenizer.decode(tokens_ids, skip_special_tokens=True), string)
        self.assertEqual(self.tokenizer.decode(tokens_ids, skip_special_tokens=False), "<s>" + string + "</s>")
        self.assertEqual(self.tokenizer.decode(tokens_ids_with_space, skip_special_tokens=True), string_with_space)

        # Test 2:
        # decode with clean_up_tokenization_spaces
        self.assertEqual(
            self.tokenizer.decode(tokens_ids_with_space, skip_special_tokens=True, clean_up_tokenization_spaces=True),
            "Hello, world!",
        )

        # Test 3:
        # decode with unsupported kwargs
        with self.assertRaises(
            ValueError, msg="Kwargs [unk_args] are not supported by `MistralCommonTokenizer.decode`."
        ):
            self.tokenizer.decode(tokens_ids, skip_special_tokens=False, unk_args="")

    def test_batch_decode(self):
        string = "Hello, world!"
        string_with_space = "Hello, world !"

        batch_tokens_ids = [
            self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(string, bos=True, eos=True),
            self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(string_with_space, bos=True, eos=True),
        ]

        # Test 1:
        # batch_decode with and without skip_special_tokens
        self.assertEqual(
            self.tokenizer.batch_decode(batch_tokens_ids, skip_special_tokens=True),
            [string, string_with_space],
        )
        self.assertEqual(
            self.tokenizer.batch_decode(batch_tokens_ids, skip_special_tokens=False),
            ["<s>" + string + "</s>", "<s>" + string_with_space + "</s>"],
        )
        self.assertEqual(
            self.tokenizer.batch_decode(batch_tokens_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True),
            ["Hello, world!", "Hello, world!"],
        )

        # Test 2:
        # batch_decode with unsupported kwargs
        with self.assertRaises(
            ValueError, msg="Kwargs [unk_args] are not supported by `MistralCommonTokenizer.batch_decode`."
        ):
            self.tokenizer.batch_decode(batch_tokens_ids, skip_special_tokens=False, unk_args="")

    def test_convert_ids_to_tokens(self):
        # Test 1:
        # with skip_special_tokens=False
        ids = self.ref_tokenizer.instruct_tokenizer.tokenizer.encode("Hello world!", bos=True, eos=True)
        expected_tokens = [self.ref_tokenizer.instruct_tokenizer.tokenizer.id_to_piece(id) for id in ids]

        tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
        self.assertEqual(tokens, expected_tokens)

        token = self.tokenizer.convert_ids_to_tokens(ids[0], skip_special_tokens=False)
        self.assertEqual(token, expected_tokens[0])

        # Test 2:
        # with skip_special_tokens=True
        expected_tokens = expected_tokens[1:-1]
        tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        self.assertEqual(tokens, expected_tokens)

        with self.assertRaises(ValueError):
            self.tokenizer.convert_ids_to_tokens(ids[0], skip_special_tokens=True)
        token = self.tokenizer.convert_ids_to_tokens(ids[1], skip_special_tokens=True)
        self.assertEqual(token, expected_tokens[0])

    def test_convert_tokens_to_ids(self):
        tokens = ["Hello", "world", "!"]
        expected_ids = [self._ref_piece_to_id(token) for token in tokens]
        # Test 1:
        # list of tokens
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        self.assertEqual(ids, expected_ids)

        # Test 2:
        # single token
        id = self.tokenizer.convert_tokens_to_ids(tokens[0])
        self.assertEqual(id, expected_ids[0])
        self.assertEqual(id, self.tokenizer.convert_tokens_to_ids(tokens[0]))

    def test_tokenize(self):
        string = "Hello world!"
        expected_tokens = [
            self.ref_tokenizer.instruct_tokenizer.tokenizer.id_to_piece(id)
            for id in self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(string, bos=False, eos=False)
        ]
        tokens = self.tokenizer.tokenize(string)
        self.assertEqual(tokens, expected_tokens)

        with self.assertRaises(
            ValueError, msg="Kwargs [add_special_tokens] are not supported by `MistralCommonTokenizer.tokenize`."
        ):
            self.tokenizer.tokenize(string, add_special_tokens=True)

    def test_get_special_tokens_mask(self):
        # Test 1:
        # with skip_special_tokens=False
        ids = self.ref_tokenizer.instruct_tokenizer.tokenizer.encode("Hello world!", bos=True, eos=True)
        expected_mask = [1 if id in self.ref_special_ids else 0 for id in ids]

        mask = self.tokenizer.get_special_tokens_mask(ids)
        self.assertEqual(mask, expected_mask)

        # Test 2:
        # already_has_special_tokens=True should raise an error
        with self.assertRaises(ValueError):
            self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)

        # Test 3:
        # token_ids_1 not None should raise an error
        with self.assertRaises(ValueError):
            self.tokenizer.get_special_tokens_mask(ids, token_ids_1=ids)

    def test_pad_batch_encoding_input(self):
        # Test 1:
        # padding and default values

        def get_batch_encoding():
            return self.tokenizer("Hello world!", return_special_tokens_mask=True)

        batch_encoding = get_batch_encoding()

        for padding in [
            False,
            True,
            "longest",
            "max_length",
            "do_not_pad",
            PaddingStrategy.LONGEST,
            PaddingStrategy.MAX_LENGTH,
            PaddingStrategy.DO_NOT_PAD,
        ]:
            padded_batch_encoding = self.tokenizer.pad(get_batch_encoding(), padding=padding)
            self.assertEqual(padded_batch_encoding, batch_encoding)

        # Test 2:
        # padding_strategy="max_length" or PaddingStrategy.MAX_LENGTH and max_length
        for padding in ["max_length", PaddingStrategy.MAX_LENGTH]:
            padded_batch_encoding = self.tokenizer.pad(get_batch_encoding(), padding=padding, max_length=12)
            self.assertEqual(
                padded_batch_encoding["input_ids"],
                [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id] * (12 - len(batch_encoding["input_ids"]))
                + batch_encoding["input_ids"],
            )
            self.assertEqual(
                padded_batch_encoding["attention_mask"],
                [0] * (12 - len(batch_encoding["input_ids"])) + batch_encoding["attention_mask"],
            )
            self.assertEqual(
                padded_batch_encoding["special_tokens_mask"],
                [1] * (12 - len(batch_encoding["input_ids"])) + batch_encoding["special_tokens_mask"],
            )

        # Test 3:
        # padding_strategy=True or "longest" or PaddingStrategy.LONGEST or "max_length" or PaddingStrategy.MAX_LENGTH and pad_to_multiple_of 16
        for padding in [True, "longest", PaddingStrategy.LONGEST]:
            padded_batch_encoding = self.tokenizer.pad(get_batch_encoding(), padding=padding, pad_to_multiple_of=16)
            self.assertEqual(
                padded_batch_encoding["input_ids"],
                [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id] * (16 - len(batch_encoding["input_ids"]))
                + batch_encoding["input_ids"],
            )
            self.assertEqual(
                padded_batch_encoding["attention_mask"],
                [0] * (16 - len(batch_encoding["input_ids"])) + batch_encoding["attention_mask"],
            )
            self.assertEqual(
                padded_batch_encoding["special_tokens_mask"],
                [1] * (16 - len(batch_encoding["input_ids"])) + batch_encoding["special_tokens_mask"],
            )

        # Test 4:
        # padding_side="right"
        right_tokenizer = MistralCommonTokenizer.from_pretrained(
            "hf-internal-testing/namespace-mistralai-repo_name-Mistral-Small-3.1-24B-Instruct-2503",
            padding_side="right",
        )
        right_paddings = [
            right_tokenizer.pad(get_batch_encoding(), padding="max_length", max_length=12),
            self.tokenizer.pad(get_batch_encoding(), padding="max_length", max_length=12, padding_side="right"),
        ]
        for padded_batch_encoding in right_paddings:
            self.assertEqual(
                padded_batch_encoding["input_ids"],
                batch_encoding["input_ids"]
                + [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id] * (12 - len(batch_encoding["input_ids"])),
            )
            self.assertEqual(
                padded_batch_encoding["attention_mask"],
                batch_encoding["attention_mask"] + [0] * (12 - len(batch_encoding["input_ids"])),
            )
            self.assertEqual(
                padded_batch_encoding["special_tokens_mask"],
                batch_encoding["special_tokens_mask"] + [1] * (12 - len(batch_encoding["input_ids"])),
            )

        # Test 5:
        # return_attention_mask=False
        padded_batch_encoding = self.tokenizer.pad(
            get_batch_encoding(), padding="max_length", max_length=12, return_attention_mask=False
        )
        self.assertEqual(
            padded_batch_encoding["input_ids"],
            [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id] * (12 - len(batch_encoding["input_ids"]))
            + batch_encoding["input_ids"],
        )
        self.assertEqual(padded_batch_encoding["attention_mask"], batch_encoding["attention_mask"])
        self.assertEqual(
            padded_batch_encoding["special_tokens_mask"],
            [1] * (12 - len(batch_encoding["input_ids"])) + batch_encoding["special_tokens_mask"],
        )

        # Test 6:
        # return_tensors="pt" or "np"
        for return_tensors in ["pt", "np"]:
            padded_batch_encoding = self.tokenizer.pad(
                get_batch_encoding(), padding="max_length", max_length=12, return_tensors=return_tensors
            )
            self.assertEqual(padded_batch_encoding["input_ids"].shape, torch.Size((12,)))
            self.assertEqual(padded_batch_encoding["attention_mask"].shape, torch.Size((12,)))
            self.assertEqual(padded_batch_encoding["special_tokens_mask"].shape, torch.Size((12,)))

    def test_list_batch_encoding_input(self):
        def get_batch_encoding():
            return self.tokenizer(["Hello world!", "Hello world! Longer sentence."], return_special_tokens_mask=True)

        # Test 1:
        # padding=True or "longest" or PaddingStrategy.LONGEST
        batch_encoding = get_batch_encoding()
        for padding in [
            True,
            "longest",
            PaddingStrategy.LONGEST,
        ]:
            padded_batch_encoding = self.tokenizer.pad(get_batch_encoding(), padding=padding)
            self.assertEqual(
                padded_batch_encoding["input_ids"],
                [
                    [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id]
                    * (len(batch_encoding["input_ids"][1]) - len(batch_encoding["input_ids"][0]))
                    + batch_encoding["input_ids"][0],
                    batch_encoding["input_ids"][1],
                ],
            )
            self.assertEqual(
                padded_batch_encoding["attention_mask"],
                [
                    [0] * (len(batch_encoding["input_ids"][1]) - len(batch_encoding["input_ids"][0]))
                    + batch_encoding["attention_mask"][0],
                    batch_encoding["attention_mask"][1],
                ],
            )
            self.assertEqual(
                padded_batch_encoding["special_tokens_mask"],
                [
                    [1] * (len(batch_encoding["input_ids"][1]) - len(batch_encoding["input_ids"][0]))
                    + batch_encoding["special_tokens_mask"][0],
                    batch_encoding["special_tokens_mask"][1],
                ],
            )

        # Test 2:
        # padding_strategy="max_length" or PaddingStrategy.MAX_LENGTH and max_length
        for padding in ["max_length", PaddingStrategy.MAX_LENGTH]:
            padded_batch_encoding = self.tokenizer.pad(get_batch_encoding(), padding=padding, max_length=12)
            self.assertEqual(
                padded_batch_encoding["input_ids"],
                [
                    [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id]
                    * (12 - len(batch_encoding["input_ids"][0]))
                    + batch_encoding["input_ids"][0],
                    [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id]
                    * (12 - len(batch_encoding["input_ids"][1]))
                    + batch_encoding["input_ids"][1],
                ],
            )
            self.assertEqual(
                padded_batch_encoding["attention_mask"],
                [
                    [0] * (12 - len(batch_encoding["input_ids"][0])) + batch_encoding["attention_mask"][0],
                    [0] * (12 - len(batch_encoding["input_ids"][1])) + batch_encoding["attention_mask"][1],
                ],
            )
            self.assertEqual(
                padded_batch_encoding["special_tokens_mask"],
                [
                    [1] * (12 - len(batch_encoding["input_ids"][0])) + batch_encoding["special_tokens_mask"][0],
                    [1] * (12 - len(batch_encoding["input_ids"][1])) + batch_encoding["special_tokens_mask"][1],
                ],
            )

        # Test 3:
        # padding_strategy=True or "longest" or PaddingStrategy.LONGEST or "max_length" or PaddingStrategy.MAX_LENGTH and pad_to_multiple_of 16
        for padding in [True, "longest", PaddingStrategy.LONGEST]:
            padded_batch_encoding = self.tokenizer.pad(get_batch_encoding(), padding=padding, pad_to_multiple_of=16)
            self.assertEqual(
                padded_batch_encoding["input_ids"],
                [
                    [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id]
                    * (16 - len(batch_encoding["input_ids"][0]))
                    + batch_encoding["input_ids"][0],
                    [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id]
                    * (16 - len(batch_encoding["input_ids"][1]))
                    + batch_encoding["input_ids"][1],
                ],
            )
            self.assertEqual(
                padded_batch_encoding["attention_mask"],
                [
                    [0] * (16 - len(batch_encoding["input_ids"][0])) + batch_encoding["attention_mask"][0],
                    [0] * (16 - len(batch_encoding["input_ids"][1])) + batch_encoding["attention_mask"][1],
                ],
            )
            self.assertEqual(
                padded_batch_encoding["special_tokens_mask"],
                [
                    [1] * (16 - len(batch_encoding["input_ids"][0])) + batch_encoding["special_tokens_mask"][0],
                    [1] * (16 - len(batch_encoding["input_ids"][1])) + batch_encoding["special_tokens_mask"][1],
                ],
            )

        # Test 4:
        # padding_side="right"
        right_tokenizer = MistralCommonTokenizer.from_pretrained(
            "hf-internal-testing/namespace-mistralai-repo_name-Mistral-Small-3.1-24B-Instruct-2503",
            padding_side="right",
        )
        right_paddings = [
            right_tokenizer.pad(get_batch_encoding(), padding="max_length", max_length=12),
            self.tokenizer.pad(get_batch_encoding(), padding="max_length", max_length=12, padding_side="right"),
        ]
        for padded_batch_encoding in right_paddings:
            self.assertEqual(
                padded_batch_encoding["input_ids"],
                [
                    batch_encoding["input_ids"][0]
                    + [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id]
                    * (12 - len(batch_encoding["input_ids"][0])),
                    batch_encoding["input_ids"][1]
                    + [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id]
                    * (12 - len(batch_encoding["input_ids"][1])),
                ],
            )
            self.assertEqual(
                padded_batch_encoding["attention_mask"],
                [
                    batch_encoding["attention_mask"][0] + [0] * (12 - len(batch_encoding["input_ids"][0])),
                    batch_encoding["attention_mask"][1] + [0] * (12 - len(batch_encoding["input_ids"][1])),
                ],
            )
            self.assertEqual(
                padded_batch_encoding["special_tokens_mask"],
                [
                    batch_encoding["special_tokens_mask"][0] + [1] * (12 - len(batch_encoding["input_ids"][0])),
                    batch_encoding["special_tokens_mask"][1] + [1] * (12 - len(batch_encoding["input_ids"][1])),
                ],
            )

        # Test 5:
        # return_attention_mask=False
        padded_batch_encoding = self.tokenizer.pad(
            get_batch_encoding(), padding="max_length", max_length=12, return_attention_mask=False
        )
        self.assertEqual(
            padded_batch_encoding["input_ids"],
            [
                [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id] * (12 - len(batch_encoding["input_ids"][0]))
                + batch_encoding["input_ids"][0],
                [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id] * (12 - len(batch_encoding["input_ids"][1]))
                + batch_encoding["input_ids"][1],
            ],
        )
        self.assertEqual(padded_batch_encoding["attention_mask"], batch_encoding["attention_mask"])
        self.assertEqual(
            padded_batch_encoding["special_tokens_mask"],
            [
                [1] * (12 - len(batch_encoding["input_ids"][0])) + batch_encoding["special_tokens_mask"][0],
                [1] * (12 - len(batch_encoding["input_ids"][1])) + batch_encoding["special_tokens_mask"][1],
            ],
        )

        # Test 6:
        # return_tensors="pt" or "np"
        for return_tensors in ["pt", "np"]:
            padded_batch_encoding = self.tokenizer.pad(
                get_batch_encoding(), padding="max_length", max_length=12, return_tensors=return_tensors
            )
            self.assertEqual(padded_batch_encoding["input_ids"].shape, torch.Size((2, 12)))
            self.assertEqual(padded_batch_encoding["attention_mask"].shape, torch.Size((2, 12)))
            self.assertEqual(padded_batch_encoding["special_tokens_mask"].shape, torch.Size((2, 12)))

    def test_truncate_sequences(self):
        # Test 1:
        # truncation_strategy="longest_first" or TruncationStrategy.LONGEST_FIRST
        text = "Hello world!"
        ids = self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=True, eos=True)
        for truncation in ["longest_first", TruncationStrategy.LONGEST_FIRST]:
            for num_tokens_to_remove in [0, 2]:
                tokens, none, overflowing_tokens = self.tokenizer.truncate_sequences(
                    ids, truncation_strategy=truncation, num_tokens_to_remove=num_tokens_to_remove
                )
                self.assertEqual(tokens, ids[:-num_tokens_to_remove] if num_tokens_to_remove > 0 else ids)
                self.assertIsNone(none)
                self.assertEqual(overflowing_tokens, ids[-num_tokens_to_remove:] if num_tokens_to_remove > 0 else [])

        # Test 2:
        # truncation_strategy="only_first" or "only_second" or TruncationStrategy.ONLY_FIRST or TruncationStrategy.ONLY_SECOND
        # Should raise a ValueError
        for truncation in ["only_first", "only_second", TruncationStrategy.ONLY_FIRST, TruncationStrategy.ONLY_SECOND]:
            with self.assertRaises(ValueError):
                self.tokenizer.truncate_sequences(ids, truncation_strategy=truncation, num_tokens_to_remove=1)

        # Test 3:
        # truncation_strategy="do_not_truncate" or TruncationStrategy.DO_NOT_TRUNCATE
        for truncation in ["do_not_truncate", TruncationStrategy.DO_NOT_TRUNCATE]:
            tokens, none, overflowing_tokens = self.tokenizer.truncate_sequences(
                ids, truncation_strategy=truncation, num_tokens_to_remove=1
            )
            self.assertEqual(tokens, ids)
            self.assertIsNone(none)
            self.assertEqual(overflowing_tokens, [])

        # Test 4:
        # pair_ids is not None
        # Should raise a ValueError
        with self.assertRaises(ValueError):
            self.tokenizer.truncate_sequences(
                ids, pair_ids=ids, truncation_strategy="longest_first", num_tokens_to_remove=1
            )

        # Test 5:
        # stride
        for stride in [0, 2]:
            tokens, none, overflowing_tokens = self.tokenizer.truncate_sequences(
                ids, truncation_strategy="longest_first", num_tokens_to_remove=2, stride=stride
            )
            self.assertEqual(tokens, ids[:-2])
            self.assertIsNone(none)
            self.assertEqual(overflowing_tokens, ids[-2 - stride :])

        # Test 6:
        # truncation_side="left"
        left_tokenizer = MistralCommonTokenizer.from_pretrained(
            "hf-internal-testing/namespace-mistralai-repo_name-Mistral-Small-3.1-24B-Instruct-2503",
            truncation_side="left",
        )
        tokens, none, overflowing_tokens = left_tokenizer.truncate_sequences(
            ids, truncation_strategy="longest_first", num_tokens_to_remove=2
        )
        self.assertEqual(tokens, ids[2:])
        self.assertIsNone(none)
        self.assertEqual(overflowing_tokens, ids[:2])

    def test_apply_chat_template_basic(self):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        expected_tokenized = self.ref_tokenizer.encode_chat_completion(ChatCompletionRequest.from_openai(conversation))

        # Test 1:
        # with tokenize
        self.assertEqual(
            self.tokenizer.apply_chat_template(conversation, tokenize=False),
            expected_tokenized.text,
        )

        # Test 2:
        # without tokenize
        self.assertEqual(self.tokenizer.apply_chat_template(conversation, tokenize=True), expected_tokenized.tokens)

        with self.assertRaises(
            ValueError, msg="Kwargs [unk_args] are not supported by `MistralCommonTokenizer.apply_chat_template`."
        ):
            self.tokenizer.apply_chat_template(conversation, tokenize=True, unk_args="")

    def test_apply_chat_template_continue_final_message(self):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
        ]

        expected_tokenized = self.ref_tokenizer.encode_chat_completion(
            ChatCompletionRequest.from_openai(conversation, continue_final_message=True)
        )

        self.assertEqual(
            self.tokenizer.apply_chat_template(conversation, tokenize=False, continue_final_message=True),
            expected_tokenized.text,
        )
        self.assertEqual(
            self.tokenizer.apply_chat_template(conversation, tokenize=True, continue_final_message=True),
            expected_tokenized.tokens,
        )

        with self.assertRaises(InvalidMessageStructureException):
            self.tokenizer.apply_chat_template(conversation, tokenize=False, continue_final_message=False)

    def test_apply_chat_template_with_tools(self):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
            {"role": "user", "content": "What is the temperature in Paris?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "azerty123",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": {"location": "Paris", "format": "text", "unit": "celsius"},
                        },
                    }
                ],
            },
            {"role": "tool", "name": "get_current_weather", "content": "22", "tool_call_id": "azerty123"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                                "required": ["location"],
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            "format": {
                                "type": "string",
                                "enum": ["text", "json"],
                                "description": "The format of the response",
                                "required": ["format"],
                            },
                        },
                    },
                },
            }
        ]

        expected_tokenized = self.ref_tokenizer.encode_chat_completion(
            ChatCompletionRequest.from_openai(conversation, tools)
        )
        self.assertEqual(
            self.tokenizer.apply_chat_template(conversation, tools=tools, tokenize=False),
            expected_tokenized.text,
        )

    def test_apply_chat_template_with_image(self):
        ref_conversation = conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": IMG_URL},
                    },
                ],
            },
        ]

        expected_tokenized = self.ref_tokenizer.encode_chat_completion(
            ChatCompletionRequest.from_openai(ref_conversation)
        )
        image_contents = [
            {
                "type": "image_url",
                "image_url": {"url": IMG_URL},
            },
            {
                "type": "image",
                "url": IMG_URL,
            },
            {"type": "image", "base64": IMG_BASE_64},
        ]
        for image_content in image_contents:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is this?"}, image_content],
                },
            ]

            output = self.tokenizer.apply_chat_template(conversation, tokenize=True)
            self.assertEqual(output, expected_tokenized.tokens)

        output_dict = self.tokenizer.apply_chat_template(conversation, tokenize=True, return_dict=True)
        self.assertEqual(output_dict["input_ids"], expected_tokenized.tokens)
        self.assertEqual(len(output_dict["pixel_values"]), len(expected_tokenized.images))
        for o, e in zip(output_dict["pixel_values"], expected_tokenized.images):
            self.assertTrue(np.allclose(o, e))

        output_dict = self.tokenizer.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="pt"
        )
        self.assertEqual(output_dict["input_ids"].tolist()[0], expected_tokenized.tokens)
        self.assertTrue(torch.allclose(output_dict["pixel_values"], torch.tensor(expected_tokenized.images)))

    def test_appsly_chat_template_with_truncation(self):
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        expected_tokenized = self.ref_tokenizer.encode_chat_completion(ChatCompletionRequest.from_openai(conversation))

        # Test 1:
        # with truncation
        self.assertEqual(
            self.tokenizer.apply_chat_template(conversation, tokenize=True, truncation=True, max_length=20),
            expected_tokenized.tokens[:20],
        )

        # Test 2:
        # without truncation
        self.assertEqual(
            self.tokenizer.apply_chat_template(conversation, tokenize=True, truncation=False, max_length=20),
            expected_tokenized.tokens,
        )

        # Test 3:
        # assert truncation is boolean
        with self.assertRaises(ValueError):
            self.tokenizer.apply_chat_template(
                conversation, tokenize=True, truncation=TruncationStrategy.LONGEST_FIRST, max_length=20
            )

    def test_batch_apply_chat_template(self):
        conversations = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi!"},
                {"role": "assistant", "content": "Hello! How can I help you?"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": IMG_URL},
                        },
                    ],
                },
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi!"},
                {"role": "assistant", "content": "Hello! How can I help you?"},
                {"role": "user", "content": "What is the temperature in Paris?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "azerty123",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": {"location": "Paris", "format": "text", "unit": "celsius"},
                            },
                        }
                    ],
                },
                {"role": "tool", "name": "get_current_weather", "content": "22", "tool_call_id": "azerty123"},
            ],
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                                "required": ["location"],
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            "format": {
                                "type": "string",
                                "enum": ["text", "json"],
                                "description": "The format of the response",
                                "required": ["format"],
                            },
                        },
                    },
                },
            }
        ]

        expected_tokenized = [
            self.ref_tokenizer.encode_chat_completion(ChatCompletionRequest.from_openai(conversation, tools=tools))
            for conversation in conversations
        ]

        text_outputs = self.tokenizer.apply_chat_template(conversations, tools=tools, tokenize=False)
        token_outputs = self.tokenizer.apply_chat_template(conversations, tools=tools, tokenize=True)

        self.assertEqual(len(text_outputs), len(token_outputs))
        self.assertEqual(len(text_outputs), len(expected_tokenized))
        for text, token, expected in zip(text_outputs, token_outputs, expected_tokenized):
            self.assertEqual(text, expected.text)
            self.assertEqual(token, expected.tokens)

        with self.assertRaises(
            ValueError,
            msg="Kwargs [unk_args] are not supported by `MistralCommonTokenizer.batch_apply_chat_template`.",
        ):
            self.tokenizer.apply_chat_template(conversations, tools=tools, tokenize=True, unk_args="")

    def test_batch_apply_images(self):
        conversations = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": IMG_URL},
                        },
                    ],
                },
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image",
                            "url": IMG_URL,
                        },
                    ],
                },
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image", "base64": IMG_BASE_64},
                    ],
                },
            ],
        ]

        ref_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": IMG_URL},
                    },
                ],
            },
        ]

        expected_tokenized = self.ref_tokenizer.encode_chat_completion(
            ChatCompletionRequest.from_openai(ref_conversation)
        )

        output = self.tokenizer.apply_chat_template(conversations, tokenize=True)
        self.assertEqual(output, [expected_tokenized.tokens] * 3)

        output = self.tokenizer.apply_chat_template(conversations, tokenize=True, return_dict=True)
        self.assertEqual(output["input_ids"], [expected_tokenized.tokens] * 3)
        self.assertEqual(len(output["pixel_values"]), len(expected_tokenized.images) * 3)
        for o, e in zip(output["pixel_values"], [expected_tokenized.images] * 3):
            self.assertTrue(np.allclose(o, e))

        output = self.tokenizer.apply_chat_template(
            conversations, tokenize=True, return_dict=True, return_tensors="pt"
        )
        self.assertEqual(output["input_ids"].tolist(), [expected_tokenized.tokens] * 3)
        self.assertEqual(output["input_ids"].shape[0], len(expected_tokenized.images) * 3)
        self.assertTrue(torch.allclose(output["pixel_values"], torch.tensor([expected_tokenized.images] * 3)))

        output = self.tokenizer.apply_chat_template(
            conversations, tokenize=True, return_dict=True, return_tensors="np"
        )
        self.assertEqual(output["input_ids"].tolist(), [expected_tokenized.tokens] * 3)
        self.assertTrue(np.allclose(output["pixel_values"], np.array([expected_tokenized.images] * 3)))

    def test_batch_apply_chat_template_with_continue_final_message(self):
        conversations = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi!"},
                {"role": "assistant", "content": "Hello! How can "},
            ],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi!"},
                {"role": "assistant", "content": "Hello! How can I help you? Ou préférez vous "},
            ],
        ]

        # Test 1:
        # with continue_final_message
        expected_tokenized = [
            self.ref_tokenizer.encode_chat_completion(
                ChatCompletionRequest.from_openai(conversation, continue_final_message=True)
            )
            for conversation in conversations
        ]

        token_outputs = self.tokenizer.apply_chat_template(conversations, tokenize=True, continue_final_message=True)

        for output, expected in zip(token_outputs, expected_tokenized):
            self.assertEqual(output, expected.tokens)

        # Test 2:
        # without continue_final_message
        with self.assertRaises(InvalidMessageStructureException):
            self.tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                continue_final_message=False,
            )

        # Test 3:
        # with continue_final_message and last role is not assistant
        with self.assertRaises(InvalidMessageStructureException):
            self.tokenizer.apply_chat_template(
                conversation=[
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hi!"},
                    ]
                ],
                tokenize=True,
                continue_final_message=True,
            )

    def test_batch_apply_chat_template_with_truncation(
        self,
    ):
        # Test 1:
        # with truncation
        token_outputs = self.tokenizer.apply_chat_template(
            self.fixture_conversations, tokenize=True, truncation=True, max_length=20
        )

        for output, expected in zip(token_outputs, self.tokenized_fixture_conversations):
            self.assertEqual(output, expected.tokens[:20])

        # Test 2:
        # without truncation
        token_outputs = self.tokenizer.apply_chat_template(
            self.fixture_conversations, tokenize=True, truncation=False, max_length=20
        )
        self.assertEqual(len(token_outputs), len(self.tokenized_fixture_conversations))
        for output, expected in zip(token_outputs, self.tokenized_fixture_conversations):
            self.assertEqual(output, expected.tokens)

        # Test 3:
        # assert truncation is boolean
        with self.assertRaises(ValueError):
            self.tokenizer.apply_chat_template(
                self.fixture_conversations, tokenize=True, truncation=TruncationStrategy.LONGEST_FIRST, max_length=20
            )

    def test_batch_apply_chat_template_with_padding(
        self,
    ):
        for padding in [True, "max_length", PaddingStrategy.LONGEST, PaddingStrategy.MAX_LENGTH]:
            if padding == PaddingStrategy.MAX_LENGTH:
                # No padding if no max length is provided
                token_outputs = self.tokenizer.apply_chat_template(self.fixture_conversations, padding=padding)
                self.assertEqual(len(token_outputs), len(self.tokenized_fixture_conversations))
                for output, expected in zip(token_outputs, self.tokenized_fixture_conversations):
                    self.assertEqual(output, expected.tokens)

            max_length = 20 if padding == PaddingStrategy.MAX_LENGTH else None

            token_outputs = self.tokenizer.apply_chat_template(
                self.fixture_conversations, tokenize=True, padding=padding, max_length=max_length
            )

            if padding != PaddingStrategy.MAX_LENGTH:
                longest = max(len(tokenized.tokens) for tokenized in self.tokenized_fixture_conversations)
                self.assertEqual(len(token_outputs), len(self.tokenized_fixture_conversations))
                for output, expected in zip(token_outputs, self.tokenized_fixture_conversations):
                    self.assertEqual(
                        output,
                        [self.tokenizer.pad_token_id] * (longest - len(expected.tokens)) + expected.tokens,
                    )
            else:
                self.assertEqual(len(token_outputs), len(self.tokenized_fixture_conversations))
                for output, expected in zip(token_outputs, self.tokenized_fixture_conversations):
                    if len(expected.tokens) < max_length:
                        self.assertEqual(
                            output,
                            [self.tokenizer.pad_token_id] * (20 - len(expected.tokens)) + expected.tokens,
                        )
                    else:
                        self.assertEqual(output, expected.tokens)

        for padding in [False, "do_not_pad", PaddingStrategy.DO_NOT_PAD]:
            token_outputs = self.tokenizer.apply_chat_template(
                self.fixture_conversations, tokenize=True, padding=padding
            )
            self.assertEqual(len(token_outputs), len(self.tokenized_fixture_conversations))
            for output, expected in zip(token_outputs, self.tokenized_fixture_conversations):
                self.assertEqual(output, expected.tokens)

    def test_batch_apply_chat_template_with_padding_and_truncation(
        self,
    ):
        max_length = 20
        for padding in [True, "max_length", PaddingStrategy.LONGEST, PaddingStrategy.MAX_LENGTH]:
            token_outputs = self.tokenizer.apply_chat_template(
                self.fixture_conversations, tokenize=True, truncation=True, padding=padding, max_length=max_length
            )
            self.assertEqual(len(token_outputs), len(self.tokenized_fixture_conversations))
            for output, expected in zip(token_outputs, self.tokenized_fixture_conversations):
                self.assertEqual(
                    output, [self.tokenizer.pad_token_id] * (20 - len(expected.tokens)) + expected.tokens[:20]
                )
        for padding in [False, "do_not_pad", PaddingStrategy.DO_NOT_PAD]:
            token_outputs = self.tokenizer.apply_chat_template(
                self.fixture_conversations, tokenize=True, truncation=True, padding=padding, max_length=max_length
            )
            self.assertEqual(len(token_outputs), len(self.tokenized_fixture_conversations))
            for output, expected in zip(token_outputs, self.tokenized_fixture_conversations):
                self.assertEqual(output, expected.tokens[:20])

    def test_batch_apply_chat_template_return_tensors(self):
        # Test 1:
        # with tokenize
        token_outputs = self.tokenizer.apply_chat_template(
            self.fixture_conversations, tokenize=True, return_tensors="pt", padding=True
        )
        self.assertIsInstance(token_outputs, torch.Tensor)
        self.assertEqual(
            token_outputs.shape,
            (len(self.fixture_conversations), max(len(t.tokens) for t in self.tokenized_fixture_conversations)),
        )

        # Test 2:
        # without tokenize, should ignore return_tensors
        token_outputs = self.tokenizer.apply_chat_template(
            self.fixture_conversations, tokenize=False, return_tensors="pt", padding=True
        )
        self.assertEqual(token_outputs, [t.text for t in self.tokenized_fixture_conversations])

    def test_batch_apply_chat_template_return_dict(self):
        # Test 1:
        # with tokenize
        token_outputs = self.tokenizer.apply_chat_template(self.fixture_conversations, tokenize=True, return_dict=True)
        self.assertIn("input_ids", token_outputs)
        self.assertIn("attention_mask", token_outputs)
        self.assertEqual(token_outputs["input_ids"], [t.tokens for t in self.tokenized_fixture_conversations])
        self.assertEqual(
            token_outputs["attention_mask"], [[1] * len(t.tokens) for t in self.tokenized_fixture_conversations]
        )

        # Test 2:
        # without tokenize, should ignore return_dict
        token_outputs = self.tokenizer.apply_chat_template(
            self.fixture_conversations, tokenize=False, return_dict=True
        )
        self.assertNotIsInstance(token_outputs, dict)
        self.assertEqual(token_outputs, [t.text for t in self.tokenized_fixture_conversations])

    def test_call(self):
        # Test 1:
        # default case
        text = "Hello world!"
        expected_tokens = self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=True, eos=True)
        tokens = self.tokenizer(text)
        self.assertIsInstance(tokens, BatchEncoding)
        self.assertEqual(tokens["input_ids"], expected_tokens)
        self.assertEqual(tokens["attention_mask"], [1] * len(expected_tokens))

        # Test 2:
        # return_attention_mask=False
        tokens = self.tokenizer(text, return_attention_mask=False)
        self.assertEqual(tokens["input_ids"], expected_tokens)
        self.assertNotIn("attention_mask", tokens)

        # Test 3:
        # return_tensors="pt"
        tokens = self.tokenizer(text, return_tensors="pt")
        self.assertIsInstance(tokens["input_ids"], torch.Tensor)
        self.assertTrue(torch.equal(tokens["input_ids"], torch.Tensor(expected_tokens).unsqueeze(0)))
        self.assertIsInstance(tokens["attention_mask"], torch.Tensor)
        self.assertTrue(torch.equal(tokens["attention_mask"], torch.ones(1, len(expected_tokens))))

        # Test 4:
        # return_special_tokens_mask=True
        tokens = self.tokenizer(text, return_special_tokens_mask=True)
        self.assertEqual(tokens["input_ids"], expected_tokens)
        self.assertEqual(tokens["attention_mask"], [1] * len(expected_tokens))
        self.assertEqual(tokens["special_tokens_mask"], [1] + [0] * (len(expected_tokens) - 2) + [1])

        # Test 5:
        # add_special_tokens=False
        expected_tokens = self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=False, eos=False)
        tokens = self.tokenizer(text, add_special_tokens=False, return_special_tokens_mask=True)
        self.assertIsInstance(tokens, BatchEncoding)
        self.assertEqual(tokens["input_ids"], expected_tokens)
        self.assertEqual(tokens["attention_mask"], [1] * len(expected_tokens))
        self.assertEqual(tokens["special_tokens_mask"], [0] * len(expected_tokens))

        with self.assertRaises(
            ValueError, msg="Kwargs [wrong_kwarg] are not supported by `MistralCommonTokenizer.__call__`."
        ):
            self.tokenizer(text, wrong_kwarg=True)

        with self.assertRaises(
            ValueError,
            msg="`text_pair`, `text_target` and `text_pair_target` are not supported by `MistralCommonTokenizer`.",
        ):
            self.tokenizer(text, text_pair="Hello world!")
        with self.assertRaises(
            ValueError,
            msg="`text_pair`, `text_target` and `text_pair_target` are not supported by `MistralCommonTokenizer`.",
        ):
            self.tokenizer(text, text_target="Hello world!")
        with self.assertRaises(
            ValueError,
            msg="`text_pair`, `text_target` and `text_pair_target` are not supported by `MistralCommonTokenizer`.",
        ):
            self.tokenizer(text, text_pair_target="Hello world!")

    def test_call_with_truncation(self):
        # Test 1:
        # truncation=True or "longest_first" or TruncationStrategy.LONGEST_FIRST
        text = "Hello world!" * 10
        expected_tokens = self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=True, eos=True)

        for truncation in [True, "longest_first", TruncationStrategy.LONGEST_FIRST]:
            tokens = self.tokenizer(text, truncation=True, max_length=10, return_special_tokens_mask=True)
            self.assertIsInstance(tokens, BatchEncoding)
            self.assertEqual(tokens["input_ids"], expected_tokens[:10])
            self.assertEqual(tokens["attention_mask"], [1] * 10)
            self.assertEqual(tokens["special_tokens_mask"], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Test 2:
        # truncation=False
        for truncation in [False, "do_not_truncate", TruncationStrategy.DO_NOT_TRUNCATE]:
            tokens = self.tokenizer(text, truncation=truncation, return_special_tokens_mask=True)
            self.assertIsInstance(tokens, BatchEncoding)
            self.assertEqual(tokens["input_ids"], expected_tokens)
            self.assertEqual(tokens["attention_mask"], [1] * len(expected_tokens))
            self.assertEqual(tokens["special_tokens_mask"], [1] + [0] * (len(expected_tokens) - 2) + [1])

        # Test 3:
        # truncation=True or "longest_first" or TruncationStrategy.LONGEST_FIRST with return_overflowing_tokens=True and stride
        for truncation in [True, "longest_first", TruncationStrategy.LONGEST_FIRST]:
            for stride in [0, 2]:
                tokens = self.tokenizer(
                    text,
                    truncation=truncation,
                    max_length=10,
                    return_overflowing_tokens=True,
                    return_special_tokens_mask=True,
                    stride=stride,
                )
                self.assertIsInstance(tokens, BatchEncoding)
                self.assertEqual(tokens["input_ids"], expected_tokens[:10])
                self.assertEqual(tokens["attention_mask"], [1] * 10)
                self.assertEqual(tokens["special_tokens_mask"], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                self.assertEqual(tokens["overflowing_tokens"], expected_tokens[10 - stride :])
                self.assertEqual(tokens["num_truncated_tokens"], len(expected_tokens) - 10)

        # Test 4:
        # truncation="only_first" or TruncationStrategy.ONLY_FIRST or "only_second" or TruncationStrategy.ONLY_SECOND
        # should raise an error
        for truncation in ["only_first", TruncationStrategy.ONLY_FIRST, "only_second", TruncationStrategy.ONLY_SECOND]:
            with self.assertRaises(
                ValueError,
                msg="Truncation strategy `only_first` and `only_second` are not supported by `MistralCommonTokenizer`.",
            ):
                self.tokenizer(text, truncation=truncation)

    def test_call_with_padding(self):
        text = "Hello world!"
        expected_tokens = self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=True, eos=True)

        # Test 1:
        # padding=False or padding=True or "do_not_pad" or PaddingStrategy.DO_NOT_PAD or padding="longest" or PaddingStrategy.LONGEST
        for padding in [False, True, "do_not_pad", PaddingStrategy.DO_NOT_PAD, "longest", PaddingStrategy.LONGEST]:
            tokens = self.tokenizer(text, padding=padding, return_special_tokens_mask=True)
            self.assertIsInstance(tokens, BatchEncoding)
            self.assertEqual(tokens["input_ids"], expected_tokens)
            self.assertEqual(tokens["attention_mask"], [1] * len(expected_tokens))
            self.assertEqual(tokens["special_tokens_mask"], [1] + [0] * (len(expected_tokens) - 2) + [1])

        # Test 2:
        # padding="max_length" or PaddingStrategy.MAX_LENGTH
        for padding in ["max_length", PaddingStrategy.MAX_LENGTH]:
            tokens = self.tokenizer(text, padding=padding, max_length=20, return_special_tokens_mask=True)
            self.assertIsInstance(tokens, BatchEncoding)
            num_padding = 20 - len(expected_tokens)
            self.assertEqual(tokens["input_ids"], num_padding * [self.tokenizer.pad_token_id] + expected_tokens)
            self.assertEqual(tokens["attention_mask"], num_padding * [0] + [1] * len(expected_tokens))
            self.assertEqual(
                tokens["special_tokens_mask"], num_padding * [1] + [1] + [0] * (len(expected_tokens) - 2) + [1]
            )

        # Test 3:
        # pad_to_multiple_of
        tokens = self.tokenizer(
            text, padding=True, max_length=20, pad_to_multiple_of=16, return_special_tokens_mask=True
        )
        self.assertIsInstance(tokens, BatchEncoding)
        num_padding = 16 - len(expected_tokens)
        self.assertEqual(tokens["input_ids"], num_padding * [self.tokenizer.pad_token_id] + expected_tokens)
        self.assertEqual(tokens["attention_mask"], num_padding * [0] + [1] * len(expected_tokens))
        self.assertEqual(
            tokens["special_tokens_mask"], num_padding * [1] + [1] + [0] * (len(expected_tokens) - 2) + [1]
        )

        # Test 4:
        # padding="max_length" and padding_side="right"
        tokens = self.tokenizer(
            text, padding="max_length", max_length=20, padding_side="right", return_special_tokens_mask=True
        )
        self.assertIsInstance(tokens, BatchEncoding)
        num_padding = 20 - len(expected_tokens)
        self.assertEqual(tokens["input_ids"], expected_tokens + num_padding * [self.tokenizer.pad_token_id])
        self.assertEqual(tokens["attention_mask"], [1] * len(expected_tokens) + num_padding * [0])
        self.assertEqual(
            tokens["special_tokens_mask"], [1] + [0] * (len(expected_tokens) - 2) + [1] + num_padding * [1]
        )

    def test_batch_call(self):
        # Test 1:
        # default case
        text = ["Hello world!", "Hello world! Longer"]
        expected_tokens = [self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(t, bos=True, eos=True) for t in text]
        tokens = self.tokenizer(text)
        self.assertIsInstance(tokens, BatchEncoding)
        self.assertEqual(tokens["input_ids"], expected_tokens)
        self.assertEqual(tokens["attention_mask"], [[1] * len(t) for t in expected_tokens])

        # Test 2:
        # return_attention_mask=False
        tokens = self.tokenizer(text, return_attention_mask=False)
        self.assertEqual(tokens["input_ids"], expected_tokens)
        self.assertNotIn("attention_mask", tokens)

        # Test 3:
        # return_tensors="pt"
        tokens = self.tokenizer(text, return_tensors="pt", padding="longest", return_special_tokens_mask=True)
        self.assertIsInstance(tokens["input_ids"], torch.Tensor)
        self.assertEqual(tokens["input_ids"].shape, torch.Size([2, len(expected_tokens[1])]))
        self.assertTrue(
            torch.equal(
                tokens["input_ids"][0],
                torch.Tensor(
                    (len(expected_tokens[1]) - len(expected_tokens[0]))
                    * [self.ref_tokenizer.instruct_tokenizer.tokenizer.pad_id]
                    + expected_tokens[0]
                ),
            )
        )
        self.assertIsInstance(tokens["attention_mask"], torch.Tensor)
        self.assertEqual(tokens["attention_mask"].shape, torch.Size([2, len(expected_tokens[1])]))
        self.assertTrue(
            torch.equal(
                tokens["attention_mask"][0],
                torch.Tensor(
                    [0] * (len(expected_tokens[1]) - len(expected_tokens[0])) + [1] * len(expected_tokens[0])
                ),
            )
        )
        self.assertTrue(torch.equal(tokens["attention_mask"][1], torch.Tensor([1] * len(expected_tokens[1]))))
        self.assertIsInstance(tokens["special_tokens_mask"], torch.Tensor)
        self.assertEqual(tokens["special_tokens_mask"].shape, torch.Size([2, len(expected_tokens[1])]))
        self.assertTrue(
            torch.equal(
                tokens["special_tokens_mask"][0],
                torch.Tensor(
                    (len(expected_tokens[1]) - len(expected_tokens[0])) * [1]
                    + [1]
                    + [0] * (len(expected_tokens[0]) - 2)
                    + [1]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                tokens["special_tokens_mask"][1], torch.Tensor([1] + [0] * (len(expected_tokens[1]) - 2) + [1])
            )
        )

        # Test 4:
        # add_special_tokens=False
        expected_tokens = [
            self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(t, bos=False, eos=False) for t in text
        ]
        tokens = self.tokenizer(text, add_special_tokens=False, return_special_tokens_mask=True)
        self.assertIsInstance(tokens, BatchEncoding)
        self.assertEqual(tokens["input_ids"], expected_tokens)
        self.assertEqual(tokens["attention_mask"], [[1] * len(t) for t in expected_tokens])
        self.assertEqual(tokens["special_tokens_mask"], [[0] * len(t) for t in expected_tokens])

    def test_batch_call_with_truncation(self):
        # Test 1:
        # truncation=True
        text = ["Hello world!", "Hello world! Longer" * 10]
        expected_tokens = [self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(t, bos=True, eos=True) for t in text]

        for truncation in [True, "longest_first", TruncationStrategy.LONGEST_FIRST]:
            tokens = self.tokenizer(text, truncation=True, max_length=10, return_special_tokens_mask=True)
            self.assertIsInstance(tokens, BatchEncoding)
            self.assertEqual(tokens["input_ids"], [expected_tokens[0][:10], expected_tokens[1][:10]])
            self.assertEqual(tokens["attention_mask"], [[1] * min(len(t), 10) for t in expected_tokens])
            self.assertEqual(
                tokens["special_tokens_mask"],
                [[1 if id in self.ref_special_ids else 0 for id in ids[:10]] for ids in expected_tokens],
            )

        # Test 2:
        # truncation=False
        for truncation in [False, "do_not_truncate", TruncationStrategy.DO_NOT_TRUNCATE]:
            tokens = self.tokenizer(text, truncation=truncation, return_special_tokens_mask=True)
            self.assertIsInstance(tokens, BatchEncoding)
            self.assertEqual(tokens["input_ids"], expected_tokens)
            self.assertEqual(tokens["attention_mask"], [[1] * len(t) for t in expected_tokens])
            self.assertEqual(
                tokens["special_tokens_mask"],
                [[1] + [0] * (len(t) - 2) + [1] for t in expected_tokens],
            )

        # Test 3:
        # truncation=True or "longest_first" or TruncationStrategy.LONGEST_FIRST with return_overflowing_tokens=True and stride

        for truncation in [True, "longest_first", TruncationStrategy.LONGEST_FIRST]:
            for stride in [0, 2]:
                tokens = self.tokenizer(
                    text,
                    truncation=truncation,
                    max_length=10,
                    return_overflowing_tokens=True,
                    return_special_tokens_mask=True,
                    stride=stride,
                )
                self.assertIsInstance(tokens, BatchEncoding)
                self.assertEqual(tokens["input_ids"], [expected_tokens[0][:10], expected_tokens[1][:10]])
                self.assertEqual(tokens["attention_mask"], [[1] * min(len(t), 10) for t in expected_tokens])
                self.assertEqual(
                    tokens["overflowing_tokens"],
                    [expected_tokens[0][10 - stride :], expected_tokens[1][10 - stride :]],
                )
                self.assertEqual(
                    tokens["num_truncated_tokens"], [len(expected_tokens[0]) - 10, len(expected_tokens[1]) - 10]
                )
                self.assertEqual(
                    tokens["special_tokens_mask"],
                    [[1 if id in self.ref_special_ids else 0 for id in ids[:10]] for ids in expected_tokens],
                )

    def test_batch_call_with_padding(self):
        # Test 1:
        # padding=False or padding=True or "do_not_pad" or PaddingStrategy.DO_NOT_PAD or padding="longest" or PaddingStrategy.LONGEST
        text = ["Hello world!", "Hello world! Longer"]
        expected_tokens = [self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(t, bos=True, eos=True) for t in text]
        for padding in [False, "do_not_pad", PaddingStrategy.DO_NOT_PAD]:
            tokens = self.tokenizer(text, padding=padding, return_special_tokens_mask=True)
            self.assertIsInstance(tokens, BatchEncoding)
            self.assertEqual(tokens["input_ids"], expected_tokens)
            self.assertEqual(tokens["attention_mask"], [[1] * len(t) for t in expected_tokens])
            self.assertEqual(
                tokens["special_tokens_mask"],
                [[1] + [0] * (len(t) - 2) + [1] for t in expected_tokens],
            )

        # Test 2:
        # padding="max_length" or PaddingStrategy.MAX_LENGTH
        for padding in ["max_length", PaddingStrategy.MAX_LENGTH]:
            tokens = self.tokenizer(text, padding=padding, max_length=20, return_special_tokens_mask=True)
            self.assertIsInstance(tokens, BatchEncoding)
            num_padding = [20 - len(t) for t in expected_tokens]
            self.assertEqual(
                tokens["input_ids"],
                [
                    num_padding[0] * [self.tokenizer.pad_token_id] + expected_tokens[0],
                    num_padding[1] * [self.tokenizer.pad_token_id] + expected_tokens[1],
                ],
            )
            self.assertEqual(
                tokens["attention_mask"],
                [
                    num_padding[0] * [0] + [1] * len(expected_tokens[0]),
                    num_padding[1] * [0] + [1] * len(expected_tokens[1]),
                ],
            )
            self.assertEqual(
                tokens["special_tokens_mask"],
                [
                    num_padding[0] * [1] + [1] + [0] * (len(expected_tokens[0]) - 2) + [1],
                    num_padding[1] * [1] + [1] + [0] * (len(expected_tokens[1]) - 2) + [1],
                ],
            )

        # Test 3:
        # padding=True or "longest" or PaddingStrategy.LONGEST
        for padding in [True, "longest", PaddingStrategy.LONGEST]:
            tokens = self.tokenizer(text, padding=padding, return_special_tokens_mask=True)
            self.assertIsInstance(tokens, BatchEncoding)
            num_padding = [len(expected_tokens[1]) - len(t) for t in expected_tokens]
            self.assertEqual(
                tokens["input_ids"],
                [
                    num_padding[0] * [self.tokenizer.pad_token_id] + expected_tokens[0],
                    num_padding[1] * [self.tokenizer.pad_token_id] + expected_tokens[1],
                ],
            )
            self.assertEqual(
                tokens["attention_mask"],
                [
                    num_padding[0] * [0] + [1] * len(expected_tokens[0]),
                    num_padding[1] * [0] + [1] * len(expected_tokens[1]),
                ],
            )
            self.assertEqual(
                tokens["special_tokens_mask"],
                [
                    num_padding[0] * [1] + [1] + [0] * (len(expected_tokens[0]) - 2) + [1],
                    num_padding[1] * [1] + [1] + [0] * (len(expected_tokens[1]) - 2) + [1],
                ],
            )

        # Test 4:
        # pad_to_multiple_of
        tokens = self.tokenizer(
            text, padding=True, max_length=32, pad_to_multiple_of=16, return_special_tokens_mask=True
        )
        self.assertIsInstance(tokens, BatchEncoding)
        num_padding = [16 - len(t) for t in expected_tokens]
        self.assertEqual(
            tokens["input_ids"],
            [
                num_padding[0] * [self.tokenizer.pad_token_id] + expected_tokens[0],
                num_padding[1] * [self.tokenizer.pad_token_id] + expected_tokens[1],
            ],
        )
        self.assertEqual(
            tokens["attention_mask"],
            [
                num_padding[0] * [0] + [1] * len(expected_tokens[0]),
                num_padding[1] * [0] + [1] * len(expected_tokens[1]),
            ],
        )
        self.assertEqual(
            tokens["special_tokens_mask"],
            [
                num_padding[0] * [1] + [1] + [0] * (len(expected_tokens[0]) - 2) + [1],
                num_padding[1] * [1] + [1] + [0] * (len(expected_tokens[1]) - 2) + [1],
            ],
        )

        # Test 5:
        # padding="max_length" or PaddingStrategy.MAX_LENGTH and padding_side="right"
        for padding in ["max_length", PaddingStrategy.MAX_LENGTH]:
            tokens = self.tokenizer(
                text, padding=padding, max_length=20, padding_side="right", return_special_tokens_mask=True
            )
            self.assertIsInstance(tokens, BatchEncoding)
            num_padding = [20 - len(t) for t in expected_tokens]
            self.assertEqual(
                tokens["input_ids"],
                [
                    expected_tokens[0] + num_padding[0] * [self.tokenizer.pad_token_id],
                    expected_tokens[1] + num_padding[1] * [self.tokenizer.pad_token_id],
                ],
            )
            self.assertEqual(
                tokens["attention_mask"],
                [
                    [1] * len(expected_tokens[0]) + num_padding[0] * [0],
                    [1] * len(expected_tokens[1]) + num_padding[1] * [0],
                ],
            )
            self.assertEqual(
                tokens["special_tokens_mask"],
                [
                    [1] + [0] * (len(expected_tokens[0]) - 2) + [1] + num_padding[0] * [1],
                    [1] + [0] * (len(expected_tokens[1]) - 2) + [1] + num_padding[1] * [1],
                ],
            )

    def test_batch_call_with_padding_and_truncation(self):
        # Test 1:
        # padding=True or "longest" or PaddingStrategy.LONGEST or "max_length" or PaddingStragy.MAX_LENGTH
        # and truncation=True or "longest_first" or TruncationStrategy.LONGEST_FIRST
        # and max_length
        text = ["Hello world!", "Hello world! Longer" * 10]
        expected_tokens = [self.ref_tokenizer.instruct_tokenizer.tokenizer.encode(t, bos=True, eos=True) for t in text]
        for padding in [True, "longest", PaddingStrategy.LONGEST, "max_length", PaddingStrategy.MAX_LENGTH]:
            for truncation in [True, "longest_first", TruncationStrategy.LONGEST_FIRST]:
                tokens = self.tokenizer(
                    text, padding=padding, truncation=truncation, max_length=10, return_special_tokens_mask=True
                )
                num_padding = [max(0, 10 - len(t)) for t in expected_tokens]
                self.assertIsInstance(tokens, BatchEncoding)
                self.assertEqual(
                    tokens["input_ids"],
                    [num_padding[i] * [self.tokenizer.pad_token_id] + t[:10] for i, t in enumerate(expected_tokens)],
                )
                self.assertEqual(
                    tokens["attention_mask"],
                    [num_padding[i] * [0] + [1] * min(len(t), 10) for i, t in enumerate(expected_tokens)],
                )
                self.assertEqual(
                    tokens["special_tokens_mask"],
                    [
                        num_padding[i] * [1] + [1 if id in self.ref_special_ids else 0 for id in ids[:10]]
                        for i, ids in enumerate(expected_tokens)
                    ],
                )

        # Test 2:
        # padding=True or "longest" or PaddingStrategy.LONGEST and truncation=True or "longest_first" or TruncationStrategy.LONGEST_FIRST
        # and no max_length
        for padding in ["longest", PaddingStrategy.LONGEST]:
            for truncation in [True, "longest_first", TruncationStrategy.LONGEST_FIRST]:
                tokens = self.tokenizer(text, padding=padding, truncation=truncation, return_special_tokens_mask=True)
                self.assertIsInstance(tokens, BatchEncoding)
                num_padding = [max(len(t) for t in expected_tokens) - len(t) for t in expected_tokens]
                self.assertEqual(
                    tokens["input_ids"],
                    [num_padding[i] * [self.tokenizer.pad_token_id] + t for i, t in enumerate(expected_tokens)],
                )
                self.assertEqual(
                    tokens["attention_mask"],
                    [num_padding[i] * [0] + [1] * len(t) for i, t in enumerate(expected_tokens)],
                )
                self.assertEqual(
                    tokens["special_tokens_mask"],
                    [
                        num_padding[i] * [1] + [1 if id in self.ref_special_ids else 0 for id in ids]
                        for i, ids in enumerate(expected_tokens)
                    ],
                )
