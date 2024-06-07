# coding=utf-8


import json
import os
import unittest

from transformers import AddedToken, YuanTokenizer, YuanTokenizerFast
from transformers.models.yuan2_m32.tokenization_yuan import VOCAB_FILES_NAMES, bytes_to_unicode
from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class YuanTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "Yuan/yuan-tokenizer"
    tokenizer_class = YuanTokenizer
    rust_tokenizer_class = YuanTokenizerFast
    test_slow_tokenizer = True
    test_rust_tokenizer = True
    space_between_special_tokens = False
    from_pretrained_kwargs = None
    test_seq2seq = False

    def setUp(self):
        super().setUp()

        # this make sure the vocabuary is complete at the byte level.
        vocab = list(bytes_to_unicode().values())
        # the vocabulary, note:
        # - `"\u0120n"`, `"\u0120lowest"`, `"\u0120newer"`, and `"\u0120wider"` are ineffective, because there are
        #   not in the merges.
        # - `"01"` is ineffective, because the merge is ineffective due to pretokenization.
        vocab.extend(
            [
                "\u0120l",
                "\u0120n",
                "\u0120lo",
                "\u0120low",
                "er",
                "\u0120lowest",
                "\u0120newer",
                "\u0120wider",
                "01",
                ";}",
                ";}\u010a",
                "\u00cf\u0135",
                "\u0120#",
                "##",
            ]
        )

        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        # note: `"0 1"` is in the merges, but the pretokenization rules render it ineffective
        merges = [
            "#version: 0.2",
            "\u0120 l",
            "\u0120l o",
            "\u0120lo w",
            "e r",
            "0 1",
            "; }",
            ";} \u010a",
            "\u00cf \u0135",
            "\u0120 #",
            "# #",
        ]

        self.special_tokens_map = {"eos_token": "<|endoftext|>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return YuanTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return YuanTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        # this case should cover
        # - NFC normalization (code point U+03D3 has different normalization forms under NFC, NFD, NFKC, and NFKD)
        # - the pretokenization rules (spliting digits and merging symbols with \n\r)
        input_text = "lower lower newer 010;}\n<|endoftext|>\u03d2\u0301"
        output_text = "lower lower newer 010;}\n<|endoftext|>\u03d3"
        return input_text, output_text

    def test_python_full_tokenizer(self):
        tokenizer = self.get_tokenizer()
        sequence, _ = self.get_input_output_texts(tokenizer)
        bpe_tokens = [
            "l",
            "o",
            "w",
            "er",
            "\u0120low",
            "er",
            "\u0120",
            "n",
            "e",
            "w",
            "er",
            "\u0120",
            "0",
            "1",
            "0",
            ";}\u010a",
            "<|endoftext|>",
            "\u00cf\u0135",
        ]
        tokens = tokenizer.tokenize(sequence)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens
        input_bpe_tokens = [75, 78, 86, 260, 259, 260, 220, 77, 68, 86, 260, 220, 15, 16, 15, 266, 270, 267]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    @unittest.skip("We disable the test of pretokenization as it is not reversible.")
    def test_pretokenized_inputs(self):
        # the test case in parent class uses str.split to "pretokenize",
        # which eats the whitespaces, which, in turn, is not reversible.
        # the results, by nature, should be different.
        pass

    @unittest.skip("We disable the test of clean up tokenization spaces as it is not applicable.")
    def test_clean_up_tokenization_spaces(self):
        # it only tests bert-base-uncased and clean_up_tokenization_spaces is not applicable to this tokenizer
        pass

    def test_nfc_normalization(self):
        # per https://unicode.org/faq/normalization.html, there are three characters whose normalization forms
        # under NFC, NFD, NFKC, and NFKD are all different
        # using these, we can make sure only NFC is applied
        input_string = "\u03d2\u0301\u03d2\u0308\u017f\u0307"  # the NFD form
        output_string = "\u03d3\u03d4\u1e9b"  # the NFC form

        if self.test_slow_tokenizer:
            tokenizer = self.get_tokenizer()
            tokenizer_output_string, _ = tokenizer.prepare_for_tokenization(input_string)
            self.assertEqual(tokenizer_output_string, output_string)

        if self.test_rust_tokenizer:
            tokenizer = self.get_rust_tokenizer()
            # we can check the class of the normalizer, but it would be okay if Sequence([NFD, NFC]) is used
            # let's check the output instead
            tokenizer_output_string = tokenizer.backend_tokenizer.normalizer.normalize_str(input_string)
            self.assertEqual(tokenizer_output_string, output_string)

    def test_slow_tokenizer_token_with_number_sign(self):
        if not self.test_slow_tokenizer:
            return

        sequence = " ###"
        token_ids = [268, 269]

        tokenizer = self.get_tokenizer()
        self.assertListEqual(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sequence)), token_ids)

    def test_slow_tokenizer_decode_spaces_between_special_tokens_default(self):
        # YuanTokenizer changes the default `spaces_between_special_tokens` in `decode` to False
        if not self.test_slow_tokenizer:
            return

        # tokenizer has a special token: `"<|endfotext|>"` as eos, but it is not `legacy_added_tokens`
        # special tokens in `spaces_between_special_tokens` means spaces between `legacy_added_tokens`
        # that would be `"<|im_start|>"` and `"<|im_end|>"` in Yuan2M32 Models
        token_ids = [259, 260, 270, 271, 26]
        sequence = " lower<|endoftext|><|im_start|>;"
        sequence_with_space = " lower<|endoftext|> <|im_start|> ;"

        tokenizer = self.get_tokenizer()
        # let's add a legacy_added_tokens
        im_start = AddedToken(
            "<|im_start|>", single_word=False, lstrip=False, rstrip=False, special=True, normalized=False
        )
        tokenizer.add_tokens([im_start])

        # `spaces_between_special_tokens` defaults to False
        self.assertEqual(tokenizer.decode(token_ids), sequence)

        # but it can be set to True
        self.assertEqual(tokenizer.decode(token_ids, spaces_between_special_tokens=True), sequence_with_space)

    @slow
    def test_tokenizer_integration(self):
        sequences = [
            "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides ",
            "general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet...) for Natural ",
            "Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained "
        ]

        expected_encoding = {'input_ids': tensor([[ 81612,    414,    313,    689, 123514,  29891,   2998,    408,    282,
           3637,  25350,  29899, 133624,    322,    282,   3637,  25350,  29899,
          93289,  29899,   2151,  29897,   8128,  29871,  77188,  77188,  77188,
          77188,  77188,  77188,  77188],
        [  2498,  29899,  83264,  78342,    313,  86091,  29892,    402,   7982,
          29899,  29906,  29892,   1528,  13635,  86799,  29892,  29871, 106320,
          29924,  29892,   6652,    309,  82188,  29892,  29871, 106320,   6779,
          11410,    363,  18385,  29871],
        [ 17088,  81882,    313,  29940,  82793,  29897,    322,  18385,  17088,
          28203,    313,  29940,  82992,  29897,    411,    975,  29871,  29941,
          29906,  29974,  90911,  29871,  77188,  77188,  77188,  77188,  77188,
          77188,  77188,  77188,  77188]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0]])}

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="Yuan/yuan-tokenizer",
            revision="5909c8222473b2c73b0b73fb054552cd4ef6a8eb",
            sequences=sequences,
        )
