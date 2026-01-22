# Sentencepiece backend layer tests

import shutil
import tempfile
from typing import TYPE_CHECKING

from transformers import AutoTokenizer, PythonBackend, TokenizersBackend
from transformers.tokenization_python import AddedToken


if TYPE_CHECKING:
    pass


class SentencePieceBackendTesterMixin:
    """
    Tests that specifically test the SentencePiece backend.
    """

    tokenizer_class = None
    rust_tokenizer_class = None
    test_sentencepiece = True
    test_sentencepiece_ignore_case = False
    test_slow_tokenizer = True
    test_rust_tokenizer = False
    from_pretrained_id = "huggyllama/llama-7b"
    from_pretrained_kwargs = {"use_fast": False}

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdirname = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    @classmethod
    def get_tokenizer(cls, **kwargs) -> PythonBackend:
        merged_kwargs = {}
        if cls.from_pretrained_kwargs is not None:
            merged_kwargs.update(cls.from_pretrained_kwargs)
        merged_kwargs.update(kwargs)
        return AutoTokenizer.from_pretrained(cls.from_pretrained_id, **merged_kwargs)

    @classmethod
    def get_rust_tokenizer(cls, **kwargs) -> TokenizersBackend:
        return cls.rust_tokenizer_class.from_pretrained(cls.from_pretrained_id, **kwargs)

    def get_tokenizers(self, fast=True, **kwargs):
        if fast and self.test_rust_tokenizer and self.test_slow_tokenizer:
            return [self.get_tokenizer(**kwargs), self.get_rust_tokenizer(**kwargs)]
        elif fast and self.test_rust_tokenizer:
            return [self.get_rust_tokenizer(**kwargs)]
        elif self.test_slow_tokenizer:
            return [self.get_tokenizer(**kwargs)]
        else:
            raise ValueError("This tokenizer class has no tokenizer to be tested.")

    def test_sentencepiece_tokenize_and_convert_tokens_to_string(self):
        """Test ``_tokenize`` and ``convert_tokens_to_string``."""
        if not self.test_sentencepiece:
            self.skipTest(reason="test_sentencepiece is set to False")

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

        special_tokens = tokenizer.all_special_tokens
        special_tokens_string = tokenizer.convert_tokens_to_string(special_tokens)
        for special_token in special_tokens:
            self.assertIn(special_token, special_tokens_string)

        if self.test_rust_tokenizer:
            rust_tokenizer = self.get_rust_tokenizer()
            special_tokens_string_rust = rust_tokenizer.convert_tokens_to_string(special_tokens)
            self.assertEqual(special_tokens_string, special_tokens_string_rust)

    def test_sentencepiece_tokenize_and_decode(self):
        if not self.test_sentencepiece:
            self.skipTest(reason="test_sentencepiece is set to False")

        text = "This is text to test the tokenizer."
        if self.test_rust_tokenizer:
            tokenizer = self.get_tokenizer()
            rust_tokenizer = self.get_rust_tokenizer()

            slow_ids = tokenizer(text).input_ids
            fast_ids = rust_tokenizer(text).input_ids
            self.assertEqual(slow_ids, fast_ids)

            slow_decoded = tokenizer.decode(slow_ids)
            fast_decoded = rust_tokenizer.decode(slow_ids)
            self.assertEqual(slow_decoded, fast_decoded)

    def test_save_sentencepiece_tokenizer(self) -> None:
        text = "This is text to test the tokenizer."

        tokenizer_slow_1 = self.get_tokenizer()
        encoding_tokenizer_slow_1 = tokenizer_slow_1(text)

        tmpdirname_1 = tempfile.mkdtemp()
        tmpdirname_2 = tempfile.mkdtemp()

        tokenizer_slow_1.save_pretrained(tmpdirname_1)
        tokenizer_slow_2 = self.tokenizer_class.from_pretrained(tmpdirname_1)
        encoding_tokenizer_slow_2 = tokenizer_slow_2(text)

        shutil.rmtree(tmpdirname_1)
        tokenizer_slow_2.save_pretrained(tmpdirname_2)

        tokenizer_slow_3 = self.tokenizer_class.from_pretrained(tmpdirname_2)
        encoding_tokenizer_slow_3 = tokenizer_slow_3(text)
        shutil.rmtree(tmpdirname_2)

        self.assertEqual(encoding_tokenizer_slow_1, encoding_tokenizer_slow_2)
        self.assertEqual(encoding_tokenizer_slow_1, encoding_tokenizer_slow_3)

    def test_added_token_are_matched_longest_first(self):
        tokenizers = self.get_tokenizers(fast=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                try:
                    tokenizer.add_tokens([AddedToken("extra_id_1")])
                    tokenizer.add_tokens([AddedToken("extra_id_100")])
                except Exception:
                    # Canine cannot add tokens which are not codepoints
                    self.skipTest(reason="Cannot add those Added tokens")

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

    def test_added_tokens_do_lower_case(self):
        tokenizer = self.get_tokenizer(do_lower_case=True)
        if not hasattr(tokenizer, "do_lower_case") or not tokenizer.do_lower_case:
            self.skipTest(reason="Tokenizer does not support do_lower_case")

        special_token = tokenizer.all_special_tokens[0]

        text = special_token + " aaaaa bbbbbb low cccccccccdddddddd l " + special_token
        text2 = special_token + " AAAAA BBBBBB low CCCCCCCCCDDDDDDDD l " + special_token

        toks_before_adding = tokenizer.tokenize(text)  # toks before adding new_toks

        new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd", "AAAAA BBBBBB", "CCCCCCCCCDDDDDDDD"]
        added = tokenizer.add_tokens([AddedToken(tok, lstrip=True, rstrip=True) for tok in new_toks])

        toks_after_adding = tokenizer.tokenize(text)
        toks_after_adding2 = tokenizer.tokenize(text2)

        # Rust tokenizers don't lowercase added tokens at the time calling `tokenizer.add_tokens`,
        # while python tokenizers do, so new_toks 0 and 2 would be treated as the same, so do new_toks 1 and 3.
        self.assertIn(added, [2, 4])

        self.assertListEqual(toks_after_adding, toks_after_adding2)
        self.assertTrue(
            len(toks_before_adding) > len(toks_after_adding),  # toks_before_adding should be longer
        )

        # Check that none of the special tokens are lowercased
        sequence_with_special_tokens = "A " + " yEs ".join(tokenizer.all_special_tokens) + " B"
        # Convert the tokenized list to str as some special tokens are tokenized like normal tokens
        # which have a prefix spacee e.g. the mask token of Albert, and cannot match the original
        # special tokens exactly.
        tokenized_sequence = "".join(tokenizer.tokenize(sequence_with_special_tokens))

        for special_token in tokenizer.all_special_tokens:
            self.assertTrue(special_token in tokenized_sequence or special_token.lower() in tokenized_sequence)

    def test_add_tokens_tokenizer(self):
        tokenizer = self.get_tokenizer(do_lower_case=False)
        vocab_size = tokenizer.vocab_size
        all_size = len(tokenizer)

        self.assertNotEqual(vocab_size, 0)

        new_toks = [
            AddedToken("aaaaa bbbbbb", rstrip=True, lstrip=True),
            AddedToken("cccccccccdddddddd", rstrip=True, lstrip=True),
        ]
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

        new_toks_2 = {
            "eos_token": AddedToken(">>>>|||<||<<|<<", rstrip=True, lstrip=True),
            "pad_token": AddedToken("<<<<<|||>|>>>>|>", rstrip=True, lstrip=True),
        }
        added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
        vocab_size_3 = tokenizer.vocab_size
        all_size_3 = len(tokenizer)

        self.assertNotEqual(vocab_size_3, 0)
        self.assertEqual(vocab_size, vocab_size_3)
        self.assertEqual(added_toks_2, len(new_toks_2))
        self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

        tokens = tokenizer.encode(
            ">>>>|||<||<<|<< aaaaa bbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l", add_special_tokens=False
        )

        self.assertGreaterEqual(len(tokens), 6)
        self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[0], tokens[1])

        self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
        self.assertGreater(tokens[-2], tokens[-3])
        self.assertEqual(tokens[0], tokenizer.eos_token_id)
        self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    def test_add_special_tokens(self):
        self.skipTest(reason="Redundant with test_add_tokens_tokenizer")

    def test_add_tokens(self):
        if not self.test_rust_tokenizer:
            self.skipTest(reason="test_rust_tokenizer is set to False")

        tokenizer_r = self.get_rust_tokenizer()

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

    def test_compare_add_special_tokens(self):
        if not self.test_rust_tokenizer:
            self.skipTest(reason="test_rust_tokenizer is set to False")

        tokenizer_r = self.get_rust_tokenizer()

        simple_num_special_tokens_to_add = tokenizer_r.num_special_tokens_to_add(pair=False)

        for text in ["", " "]:
            # tokenize()
            no_special_tokens = tokenizer_r.tokenize(text, add_special_tokens=False)
            with_special_tokens = tokenizer_r.tokenize(text, add_special_tokens=True)
            self.assertEqual(len(no_special_tokens), len(with_special_tokens) - simple_num_special_tokens_to_add)

            # Single input
            no_special_tokens = tokenizer_r(text, add_special_tokens=False)
            with_special_tokens = tokenizer_r(text, add_special_tokens=True)
            for key in no_special_tokens:
                self.assertEqual(
                    len(no_special_tokens[key]),
                    len(with_special_tokens[key]) - simple_num_special_tokens_to_add,
                )

            # Batched input
            no_special_tokens = tokenizer_r([text, text], add_special_tokens=False)
            with_special_tokens = tokenizer_r([text, text], add_special_tokens=True)
            for key in no_special_tokens:
                for i_no, i_with in zip(no_special_tokens[key], with_special_tokens[key]):
                    self.assertEqual(len(i_no), len(i_with) - simple_num_special_tokens_to_add)

    def test_special_tokens_initialization(self):
        if not self.test_rust_tokenizer:
            self.skipTest(reason="test_rust_tokenizer is set to False")

        added_tokens = [AddedToken("<special>", lstrip=True)]
        tokenizer_r = self.get_rust_tokenizer(additional_special_tokens=added_tokens)
        r_output = tokenizer_r.encode("Hey this is a <special> token")

        special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

        self.assertTrue(special_token_id in r_output)

    def test_special_token_addition(self):
        tokenizer = self.get_tokenizer()
        # Create tokenizer and add an extra special token
        tokenizer.add_special_tokens({"extra_special_tokens": ["<tok>"]})
        self.assertEqual(tokenizer.extra_special_tokens, ["<tok>"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            # Load the above tokenizer and add the same special token a second time
            tokenizer_2 = self.tokenizer_class.from_pretrained(tmp_dir)
            tokenizer_2.add_special_tokens({"extra_special_tokens": ["<tok>"]})
            self.assertEqual(tokenizer_2.extra_special_tokens, ["<tok>"])

            tokenizer_2.add_special_tokens({"extra_special_tokens": ["<tok>", "<other>"]})
            self.assertEqual(tokenizer_2.extra_special_tokens, ["<tok>", "<other>"])

            tokenizer_2.add_special_tokens({"extra_special_tokens": ["<other>", "<another>"]})
            self.assertEqual(tokenizer_2.extra_special_tokens, ["<other>", "<another>"])

            tokenizer_2.add_special_tokens(
                {"extra_special_tokens": ["<tok>"]},
                replace_extra_special_tokens=False,
            )
            self.assertEqual(tokenizer_2.extra_special_tokens, ["<other>", "<another>", "<tok>"])

    def test_alignment_methods(self):
        tokenizer_r = self.get_tokenizer()
        words = ["Wonderful", "no", "inspiration", "example", "with", "subtoken"]
        text = " ".join(words)
        batch_size = 3

        encoding = tokenizer_r(text, add_special_tokens=False)

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
        self.assertEqual(batch_encoding.word_to_tokens(last_batch_index, last_word_index).end, last_token_index + 1)

        # Assert token_to_chars
        self.assertEqual(encoding.token_to_chars(0).start, 0)
        self.assertEqual(encoding.token_to_chars(0, 0).start, 0)
        self.assertEqual(encoding.token_to_chars(last_token_index).end, last_char_index + 1)
        self.assertEqual(encoding.token_to_chars(0, last_token_index).end, last_char_index + 1)
        self.assertEqual(batch_encoding.token_to_chars(1, 0).start, 0)
        self.assertEqual(batch_encoding.token_to_chars(0, last_token_index).end, last_char_index + 1)
        self.assertEqual(batch_encoding.token_to_chars(last_batch_index, last_token_index).end, last_char_index + 1)
