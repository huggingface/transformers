# Optionally test tokenizers-backend API in transformers

import inspect
import shutil
import tempfile
import unittest
from typing import TYPE_CHECKING

from parameterized import parameterized

from transformers import PreTrainedTokenizerFast, SpecialTokensMixin
from transformers.testing_utils import require_tokenizers
from transformers.tokenization_utils import AddedToken

SMALL_TRAINING_CORPUS = [
    ["This is the first sentence.", "This is the second one."],
    ["This sentence (contains #) over symbols and numbers 12 3.", "But not this one."],
]

if TYPE_CHECKING:
    pass


class TokenizersBackendTesterMixin:
    """
    Tests that specifically test the tokenizers-backend.
    These tests don't need to be run for every model, just once to verify the backend works correctly.
    """

    tokenizer_class = None
    rust_tokenizer_class = None
    from_pretrained_id = None
    from_pretrained_kwargs = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.from_pretrained_id = (
            [cls.from_pretrained_id] if isinstance(cls.from_pretrained_id, str) else cls.from_pretrained_id
        )
        cls.tokenizers_list = [
            (
                cls.rust_tokenizer_class,
                pretrained_id,
                cls.from_pretrained_kwargs if cls.from_pretrained_kwargs is not None else {},
            )
            for pretrained_id in cls.from_pretrained_id
        ]
        cls.tmpdirname = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @classmethod
    def get_rust_tokenizer(cls, pretrained_name=None, **kwargs) -> PreTrainedTokenizerFast:
        pretrained_name = pretrained_name or cls.tmpdirname
        return cls.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    def test_alignment_methods(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.get_rust_tokenizer(pretrained_name, **kwargs)

                words = ["Wonderful", "no", "inspiration", "example", "with", "subtoken"]
                text = " ".join(words)
                batch_size = 3

                encoding = tokenizer_r.encode_plus(text, add_special_tokens=False)

                batch_encoding = tokenizer_r([text] * batch_size, add_special_tokens=False)
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

                pair_batch_encoding = tokenizer_r(
                    [text] * batch_size, [pair_text] * batch_size, add_special_tokens=False
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

                pair_batch_encoding = tokenizer_r(
                    [text] * batch_size, [pair_text] * batch_size, add_special_tokens=True
                )
                pair_batch_sequence_ids = [
                    pair_batch_encoding.token_to_sequence(1, i)
                    for i in range(len(pair_batch_encoding["input_ids"][0]))
                ]
                self.assertIn(0, pair_batch_sequence_ids)
                self.assertIn(1, pair_batch_sequence_ids)
                if tokenizer_r.num_special_tokens_to_add(pair=True):
                    self.assertIn(None, pair_batch_sequence_ids)

    def test_offsets_mapping(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.get_rust_tokenizer(pretrained_name, **kwargs)

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

    def test_training_new_tokenizer(self):
        # This feature only exists for fast tokenizers
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
            if getattr(tokenizer, token) is not None:
                special_token = getattr(tokenizer, token)
                special_tokens_map[special_token] = f"{special_token}a"

        # Train new tokenizer
        new_tokenizer = tokenizer.train_new_from_iterator(
            SMALL_TRAINING_CORPUS, 100, special_tokens_map=special_tokens_map
        )

        # Check the changes
        for token in special_tokens_list:
            # Get the private one to avoid unnecessary warnings.
            if getattr(tokenizer, token) is None:
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
                special_token.content = new_special_token_str
                self.assertTrue(
                    find,
                    f"'{special_token.__repr__()}' should appear as an `AddedToken` in the all_special_tokens_extended = "
                    f"{[k for k in new_tokenizer.all_special_tokens_extended if str(k) == new_special_token_str]} but it is missing"
                    ", this means that the new tokenizers did not keep the `rstrip`, `lstrip`, `normalized` etc attributes.",
                )
            elif special_token not in special_tokens_map:
                # The special token must appear identically in the list of the new tokenizer.
                self.assertTrue(
                    special_token in new_tokenizer.all_special_tokens_extended,
                    f"'{special_token.__repr__()}' should be in {new_tokenizer.all_special_tokens_extended}",
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

    @parameterized.expand([(True,), (False,)])
    def test_rust_tokenizer_add_prefix_space(self, add_prefix_space):
        for tokenizer, pretrained_name, _ in self.tokenizers_list:
            fast_tokenizer = tokenizer.from_pretrained(pretrained_name, add_prefix_space=add_prefix_space)
            self.assertEqual(fast_tokenizer.add_prefix_space, add_prefix_space)
            # Only the ByteLevel pre-tokenizer has the `add_prefix_space` attribute, we have to ensure that it's set correctly
            if hasattr(fast_tokenizer.backend_tokenizer.pre_tokenizer, "add_prefix_space"):
                self.assertEqual(fast_tokenizer.backend_tokenizer.pre_tokenizer.add_prefix_space, add_prefix_space)
