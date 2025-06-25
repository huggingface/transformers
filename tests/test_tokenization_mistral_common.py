import tempfile
import unittest

import torch
from mistral_common.exceptions import InvalidMessageStructureException
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_mistral_common import MistralCommonTokenizer
from transformers.tokenization_utils_base import BatchEncoding, TruncationStrategy
from transformers.utils import PaddingStrategy


class TestMistralCommonTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tokenizer: MistralCommonTokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503", tokenizer_type="mistral"
        )
        cls.ref_tokenizer: MistralTokenizer = MistralTokenizer.from_hf_hub(
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
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
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503", padding_side="right"
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
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503", padding_side="right"
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
        left_tokenizer = MistralCommonTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", truncation_side="left")
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
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://picsum.photos/id/237/200/300"},
                    },
                ],
            },
        ]

        expected_tokenized = self.ref_tokenizer.encode_chat_completion(ChatCompletionRequest.from_openai(conversation))

        self.assertEqual(self.tokenizer.apply_chat_template(conversation, tokenize=True), expected_tokenized.tokens)

    def test_apply_chat_template_with_truncation(self):
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
                            "image_url": {"url": "https://picsum.photos/id/237/200/300"},
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

        for text, token, expected in zip(text_outputs, token_outputs, expected_tokenized, strict=True):
            self.assertEqual(text, expected.text)
            self.assertEqual(token, expected.tokens)

        with self.assertRaises(
            ValueError,
            msg="Kwargs [unk_args] are not supported by `MistralCommonTokenizer.batch_apply_chat_template`.",
        ):
            self.tokenizer.apply_chat_template(conversations, tools=tools, tokenize=True, unk_args="")

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

        for output, expected in zip(token_outputs, expected_tokenized, strict=True):
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

        for output, expected in zip(token_outputs, self.tokenized_fixture_conversations, strict=True):
            self.assertEqual(output, expected.tokens[:20])

        # Test 2:
        # without truncation
        token_outputs = self.tokenizer.apply_chat_template(
            self.fixture_conversations, tokenize=True, truncation=False, max_length=20
        )
        for output, expected in zip(token_outputs, self.tokenized_fixture_conversations, strict=True):
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
                for output, expected in zip(token_outputs, self.tokenized_fixture_conversations, strict=True):
                    self.assertEqual(output, expected.tokens)

            max_length = 20 if padding == PaddingStrategy.MAX_LENGTH else None

            token_outputs = self.tokenizer.apply_chat_template(
                self.fixture_conversations, tokenize=True, padding=padding, max_length=max_length
            )

            if padding != PaddingStrategy.MAX_LENGTH:
                longest = max(len(tokenized.tokens) for tokenized in self.tokenized_fixture_conversations)
                for output, expected in zip(token_outputs, self.tokenized_fixture_conversations, strict=True):
                    self.assertEqual(
                        output,
                        [self.tokenizer.pad_token_id] * (longest - len(expected.tokens)) + expected.tokens,
                    )
            else:
                for output, expected in zip(token_outputs, self.tokenized_fixture_conversations, strict=True):
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
            for output, expected in zip(token_outputs, self.tokenized_fixture_conversations, strict=True):
                self.assertEqual(output, expected.tokens)

    def test_batch_apply_chat_template_with_padding_and_truncation(
        self,
    ):
        max_length = 20
        for padding in [True, "max_length", PaddingStrategy.LONGEST, PaddingStrategy.MAX_LENGTH]:
            token_outputs = self.tokenizer.apply_chat_template(
                self.fixture_conversations, tokenize=True, truncation=True, padding=padding, max_length=max_length
            )
            for output, expected in zip(token_outputs, self.tokenized_fixture_conversations, strict=True):
                self.assertEqual(
                    output, [self.tokenizer.pad_token_id] * (20 - len(expected.tokens)) + expected.tokens[:20]
                )
        for padding in [False, "do_not_pad", PaddingStrategy.DO_NOT_PAD]:
            token_outputs = self.tokenizer.apply_chat_template(
                self.fixture_conversations, tokenize=True, truncation=True, padding=padding, max_length=max_length
            )
            for output, expected in zip(token_outputs, self.tokenized_fixture_conversations, strict=True):
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
