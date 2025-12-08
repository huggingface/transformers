# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import unittest

from transformers import (
    AddedToken,
    BatchEncoding,
    SeamlessM4TTokenizer,
    is_torch_available,
)
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
)

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


if is_torch_available():
    from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

EN_CODE = 256047
RO_CODE = 256145

SMALL_TRAINING_CORPUS = [
    ["This is the first sentence.", "This is the second one."],
    ["This sentence (contains #) over symbols and numbers 12 3.", "But not this one."],
]


@require_sentencepiece
@require_tokenizers
class SeamlessM4TTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "facebook/hf-seamless-m4t-medium"
    tokenizer_class = SeamlessM4TTokenizer
    test_rust_tokenizer = True

    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁生活', '的', '真', '<unk>', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', 'th', 'ere', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁enc', 'od', 'ed', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [9680, 248, 9, 7356, 248059, 253515, 117, 1398, 79519, 108, 855, 45299, 248079, 540, 3423, 248, 52428, 248132, 248075, 182892, 248506, 249573, 1, 249221, 2867, 94124, 2867, 94124, 94124, 2, 435, 2, 419, 275, 1617, 45893, 191422, 12516, 280, 242514, 12025, 129, 76, 248144, 94124, 248075, 9062, 528, 248072, 540, 99681, 528, 248072, 34744, 27426, 11657, 2442, 1259, 34512]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fals', 'é', '.', '▁生活', '的', '真', '<unk>', '是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '<s>', '▁hi', '<s>', 'th', 'ere', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁enc', 'od', 'ed', ':', '▁Hello', '.', '▁But', '▁ir', 'd', '▁and', '▁ปี', '▁ir', 'd', '▁ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊 I was born in 92000, and this is falsé. 生活的真<unk>是 Hi Hello Hi Hello Hello<s> hi<s>there The following string should be properly encoded: Hello. But ird and ปี ird ด Hey how are you doing"

    def test_batch_encode_plus_batch_sequence_length(self):
        # Override the parent test because SeamlessM4T uses padding=True by default
        # Tests that all encoded values have the correct size
        tokenizer = self.get_tokenizer(do_lower_case=False)
        sequences = [
            "Testing batch encode plus",
            "Testing batch encode plus with different sequence lengths",
            "Testing batch encode plus with different sequence lengths correctly pads",
        ]

        # For SeamlessM4T, encode with explicit padding=False for individual sequences too
        encoded_sequences = [tokenizer(sequence, padding=False) for sequence in sequences]
        encoded_sequences_batch = tokenizer(sequences, padding=False)
        self.assertListEqual(encoded_sequences, self.convert_batch_to_list_format(encoded_sequences_batch))

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

                    # default to padding=True so need to precise which padding is called
                    normal_tokens = tokenizer("This", pad_to_multiple_of=8, padding=False)
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
                        src_lang="eng",
                        tgt_lang="ron",
                        pad_to_multiple_of=None,
                    )
                except NotImplementedError:
                    self.skipTest(reason="Encountered NotImplementedError when calling prepare_seq2seq_batch")
                self.assertEqual(batch.input_ids.shape[1], 3)
                self.assertEqual(batch.labels.shape[1], 10)

                # TODO: not working for tgt_text
                # max_target_length will default to max_length if not specified
                batch = tokenizer.prepare_seq2seq_batch(
                    src_texts=src_text,
                    tgt_texts=tgt_text,
                    max_length=4,
                    return_tensors="pt",
                    pad_to_multiple_of=None,
                )
                self.assertEqual(batch.input_ids.shape[1], 4)
                self.assertEqual(batch.labels.shape[1], 4)

                batch_encoder_only = tokenizer.prepare_seq2seq_batch(
                    src_texts=src_text,
                    max_length=4,
                    max_target_length=10,
                    return_tensors="pt",
                    pad_to_multiple_of=None,
                )
                self.assertEqual(batch_encoder_only.input_ids.shape[1], 4)
                self.assertEqual(batch_encoder_only.attention_mask.shape[1], 4)
                self.assertNotIn("decoder_input_ids", batch_encoder_only)

    # Copied from tests.models.nllb.test_tokenization_nllb.NllbTokenizationTest.test_special_tokens_initialization
    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.get_tokenizer(pretrained_name, additional_special_tokens=added_tokens, **kwargs)
                r_output = tokenizer_r.encode("Hey this is a <special> token")

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

                self.assertTrue(special_token_id in r_output)

    def test_training_new_tokenizer(self):
        # This feature only exists for fast tokenizers
        if not self.test_rust_tokenizer:
            self.skipTest(reason="test_rust_tokenizer is set to False")

        tokenizer = self.get_tokenizer()
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
        # make sure it has the same prefix tokens first
        new_tokenizer.tgt_lang = tokenizer.tgt_lang
        tokenizer.tgt_lang = tokenizer.tgt_lang
        self.assertEqual(tokenizer.num_special_tokens_to_add(False), new_tokenizer.num_special_tokens_to_add(False))
        self.assertEqual(tokenizer.num_special_tokens_to_add(True), new_tokenizer.num_special_tokens_to_add(True))

        # Check we have the correct max_length for both pair and non-pair inputs.
        self.assertEqual(tokenizer.max_len_single_sentence, new_tokenizer.max_len_single_sentence)
        self.assertEqual(tokenizer.max_len_sentences_pair, new_tokenizer.max_len_sentences_pair)

        # Assert the set of special tokens match as we didn't ask to change them
        self.assertSequenceEqual(
            tokenizer.all_special_tokens,
            new_tokenizer.all_special_tokens,
        )

        self.assertDictEqual(tokenizer.special_tokens_map, new_tokenizer.special_tokens_map)


@require_torch
@require_sentencepiece
@require_tokenizers
class SeamlessM4TDistilledIntegrationTest(unittest.TestCase):
    checkpoint_name = "facebook/hf-seamless-m4t-medium"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        """ Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for Syria is that "there is no military solution" to the nearly five-year conflict and more weapons will only worsen the violence and misery for millions of people.""",
    ]
    tgt_text = [
        "Şeful ONU declară că nu există o soluţie militară în Siria",
        "Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar al Rusiei"
        ' pentru Siria este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi că noi arme nu vor'
        " face decât să înrăutăţească violenţele şi mizeria pentru milioane de oameni.",
    ]

    expected_src_tokens = [256047, 16297, 134408, 8165, 248066, 14734, 950, 1135, 105721, 3573, 83, 27352, 108, 49486, 3]  # fmt: skip

    @classmethod
    def setUpClass(cls):
        cls.tokenizer: SeamlessM4TTokenizer = SeamlessM4TTokenizer.from_pretrained(
            cls.checkpoint_name, src_lang="eng", tgt_lang="ron"
        )
        # cls.pad_token_id = 1
        return cls

    def setUp(self):
        # Some tests may change to source mode and not reset
        self.tokenizer.set_tgt_lang_special_tokens(self.tokenizer.tgt_lang)

    def test_int_remove_extra_whitespaces(self):
        # make sure the extra spaces are eaten. Since the sample vocab does not have
        # `______`. sentencepiece.NormalizerSpec.remove_extra_whitespaces attribute is set to False

        input_ids = self.tokenizer.encode("       . Hello")
        self.assertEqual(input_ids, [3, 256145, 81, 94124, 3])
        tokens = self.tokenizer.tokenize(" . Hello")
        self.assertEqual(tokens, ["▁.", "▁Hello"])

        # `'▁'` is also a whitespace
        input_ids = self.tokenizer.encode("▁He is not")
        self.assertEqual(input_ids, [3, 256145, 1808, 248, 2294, 3])
        tokens = self.tokenizer.tokenize("▁He is not")

        self.assertEqual(tokens, ["▁He", "▁is", "▁not"])  # no extra space added

        input_ids = self.tokenizer.encode("▁He is not<s>             ▁He")
        self.assertEqual(input_ids, [3, 256145, 1808, 248, 2294, 2, 1808, 3])
        tokens = self.tokenizer.tokenize("▁He is not<s>              ▁He")
        self.assertEqual(tokens, ["▁He", "▁is", "▁not", "<s>", "▁He"])  # spaces are eaten by spm + our strip
        # make sure that the output after the extra id is the same as if
        # extra_id was not there
        input_ids = self.tokenizer.encode("▁He is not             ▁He")
        self.assertEqual(input_ids, [3, 256145, 1808, 248, 2294, 1808, 3])
        tokens = self.tokenizer.tokenize("▁He is not              ▁He")
        self.assertEqual(tokens, ["▁He", "▁is", "▁not", "▁He"])  # spaces are eaten by spm even if not start

    def test_language_codes(self):
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("__ace_Latn__"), 256002)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("__shn__"), 256152)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("__eng__"), 256047)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("__fra__"), 256057)
        self.assertEqual(self.tokenizer.convert_tokens_to_ids("__quy__"), 256144)

    def test_tokenizer_tgt_lang(self):
        ids = self.tokenizer(self.src_text, src_lang="fra").input_ids[0]
        self.assertListEqual(self.expected_src_tokens[1:], ids[1 : len(self.expected_src_tokens)])
        self.assertEqual(256057, ids[0])

        rest_ids = ids[len(self.expected_src_tokens) :]
        self.assertListEqual([0] * len(rest_ids), rest_ids)

        ids = self.tokenizer(self.src_text, src_lang="__shn__").input_ids[0]
        self.assertListEqual(self.expected_src_tokens[1:], ids[1 : len(self.expected_src_tokens)])
        self.assertEqual(256152, ids[0])

    # Copied from tests.models.nllb.test_tokenization_nllb.NllbDistilledIntegrationTest.test_enro_tokenizer_decode_ignores_language_codes
    def test_enro_tokenizer_decode_ignores_language_codes(self):
        self.assertIn(RO_CODE, self.tokenizer.all_special_ids)
        generated_ids = [RO_CODE, 4254, 98068, 112923, 39072, 3909, 713, 102767, 26, 17314, 35642, 14683, 33118, 2022, 66987, 2, 256047]  # fmt: skip

        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        expected_romanian = self.tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        self.assertEqual(result, expected_romanian)
        self.assertNotIn(self.tokenizer.eos_token, result)

    def test_enro_tokenizer_truncation(self):
        src_text = ["this is gunna be a long sentence " * 20]
        assert isinstance(src_text[0], str)
        desired_max_length = 10
        ids = self.tokenizer(src_text, max_length=desired_max_length, truncation=True).input_ids[0]
        self.assertEqual(ids[-1], 3)
        self.assertEqual(ids[0], EN_CODE)
        self.assertEqual(len(ids), desired_max_length)

    @require_torch
    def test_enro_tokenizer_prepare_batch(self):
        batch = self.tokenizer(
            self.src_text,
            text_target=self.tgt_text,
            padding=True,
            truncation=True,
            max_length=len(self.expected_src_tokens),
            pad_to_multiple_of=None,
            return_tensors="pt",
        )
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.tokenizer.convert_tokens_to_ids("__ron__")
        )

        self.assertIsInstance(batch, BatchEncoding)

        self.assertEqual((2, 15), batch.input_ids.shape)
        self.assertEqual((2, 15), batch.attention_mask.shape)
        result = batch.input_ids.tolist()[0]
        self.assertListEqual(self.expected_src_tokens, result)
        self.assertEqual(RO_CODE, batch.decoder_input_ids[0, 0])  # EOS
        # Test that special tokens are reset
        self.assertEqual(self.tokenizer.prefix_tokens, [EN_CODE])
        self.assertEqual(self.tokenizer.suffix_tokens, [self.tokenizer.eos_token_id])

    def test_seq2seq_max_length(self):
        batch = self.tokenizer(
            self.src_text, padding=True, truncation=True, max_length=3, return_tensors="pt", pad_to_multiple_of=None
        )
        targets = self.tokenizer(
            text_target=self.tgt_text, padding=True, truncation=True, max_length=10, return_tensors="pt"
        )
        labels = targets["input_ids"]
        batch["decoder_input_ids"] = shift_tokens_right(
            labels,
            self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tgt_lang),
        )

        self.assertEqual(batch.input_ids.shape[1], 3)
        self.assertEqual(batch.decoder_input_ids.shape[1], 10)

    @require_torch
    def test_tokenizer_translation(self):
        inputs = self.tokenizer._build_translation_inputs(
            "A test", return_tensors="pt", src_lang="eng", tgt_lang="fra"
        )

        self.assertEqual(
            nested_simplify(inputs),
            {
                # A, test, EOS, en_XX
                "input_ids": [[256047, 70, 7356, 3]],
                "attention_mask": [[1, 1, 1, 1]],
                # ar_AR
                "forced_bos_token_id": 256057,
            },
        )


@require_sentencepiece
@require_tokenizers
class CommonSpmIntegrationTests(unittest.TestCase):
    """
    A class that regroups important test to make sure that we properly handle the special tokens.
    """

    @classmethod
    def setUpClass(cls):
        tokenizer = SeamlessM4TTokenizer.from_pretrained(SAMPLE_VOCAB)
        tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("<s>", rstrip=False, lstrip=False)]})
        cls.tokenizer = tokenizer
        return cls

    def setUp(self):
        self.tokenizer.set_tgt_lang_special_tokens(self.tokenizer.tgt_lang)

    def test_add_dummy_prefix(self):
        # make sure `'▁'` is prepended properly
        input_ids = self.tokenizer.encode(". Hello")
        self.assertEqual(input_ids, [3, 1, 8, 5, 157, 87, 21, 3])

        tokens = self.tokenizer.tokenize(". Hello")
        self.assertEqual(tokens, ["▁", ".", "▁He", "ll", "o"])

        tokens = self.tokenizer.tokenize("")
        self.assertEqual(tokens, [])

        tokens = self.tokenizer.tokenize(" ")
        self.assertEqual(tokens, [])

        tokens = self.tokenizer.tokenize("▁")
        self.assertEqual(tokens, [])

    def test_character_after_special_token(self):
        # Make sure that `tokenizer.tokenize` is similar to
        # adding the equivalent special token to the vocab
        input_ids = self.tokenizer.encode("Hey <s>I")
        self.assertEqual(input_ids, [3, 1, 157, 31, 2, 101, 3])

        tokens = self.tokenizer.tokenize("<s>I")
        self.assertEqual(tokens, ["<s>", "I"])

        input_ids = self.tokenizer.encode("Hello, <s>,")
        self.assertEqual(input_ids, [3, 1, 157, 87, 21, 4, 2, 4, 3])
        tokens = self.tokenizer.tokenize("Hello, <s>,")
        self.assertEqual(tokens, ["▁He", "ll", "o", ",", "<s>", ","])

    def test_special_tokens_strip(self):
        input_ids = self.tokenizer.encode(" <s> ,")
        self.assertEqual(input_ids, [3, 1, 2, 8, 4, 3])
        tokens = self.tokenizer.tokenize(" <s> ,")
        # spaces are eaten by rstrip / lstrip + normalizer
        self.assertEqual(tokens, ["<s>", "▁", ","])

        input_ids = self.tokenizer.encode("No <s> He")
        self.assertEqual(input_ids, [3, 1, 285, 2, 157, 3])
        tokens = self.tokenizer.tokenize("No <s> ▁He")
        self.assertEqual(tokens, ["▁No", "<s>", "▁He"])  # spaces are eaten by rstrip / lstrip
