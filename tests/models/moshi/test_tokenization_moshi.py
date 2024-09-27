# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import pickle
import shutil
import tempfile
import unittest

from huggingface_hub import hf_hub_download

from transformers import (
    SPIECE_UNDERLINE,
    AddedToken,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
)

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class MoshiTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = ["kmhf/hf-moshiko"]
    rust_tokenizer_class = PreTrainedTokenizerFast

    test_slow_tokenizer = False
    test_rust_tokenizer = True
    test_sentencepiece = True
    from_pretrained_kwargs = {}

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = PreTrainedTokenizerFast(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizers(self, **kwargs):
        kwargs.update({"pad_token": "<PAD>"})
        return super().get_tokenizers(**kwargs)

    def test_full_tokenizer(self):
        tokenizer = PreTrainedTokenizerFast(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [285, 46, 10, 170, 382],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [8, 21, 84, 55, 24, 19, 7, 0, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, 0, 4],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "<unk>",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "<unk>",
                ".",
            ],
        )

    @require_torch
    def test_batch_tokenization(self):
        if not self.test_seq2seq:
            self.skipTest(reason="test_seq2seq is set to False")

        tokenizers = self.get_tokenizers()
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                # Longer text that will definitely require truncation.
                text = [
                    " UN Chief Says There Is No Military Solution in Syria",
                    " Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for"
                    " Syria is that 'there is no military solution' to the nearly five-year conflict and more weapons"
                    " will only worsen the violence and misery for millions of people.",
                ]
                try:
                    batch = tokenizer(
                        text=text,
                        max_length=3,
                        max_target_length=10,
                        return_tensors="pt",
                    )
                except NotImplementedError:
                    self.skipTest(reason="Encountered NotImplementedError when calling tokenizer")
                self.assertEqual(batch.input_ids.shape[1], 3)
                # max_target_length will default to max_length if not specified
                batch = tokenizer(text, max_length=3, return_tensors="pt")
                self.assertEqual(batch.input_ids.shape[1], 3)

                batch_encoder_only = tokenizer(text=text, max_length=3, max_target_length=10, return_tensors="pt")
                self.assertEqual(batch_encoder_only.input_ids.shape[1], 3)
                self.assertEqual(batch_encoder_only.attention_mask.shape[1], 3)
                self.assertNotIn("decoder_input_ids", batch_encoder_only)

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

    @slow
    def test_tokenizer_integration(self):
        expected_encoding = {'input_ids': [[1, 4103, 689, 414, 313, 24784, 368, 2998, 408, 282, 3637, 25350, 29899, 9067, 414, 322, 282, 3637, 25350, 29899, 1457, 3018, 1312, 29899, 2151, 29897, 8128, 2498, 29899, 15503, 4220, 6956, 1973, 313, 13635, 29911, 29892, 402, 7982, 29899, 29906, 29892, 1528, 13635, 29911, 29874, 29892, 1060, 26369, 29892, 6652, 309, 29933, 814, 29892, 1060, 29931, 6779, 11410, 363, 18385, 17088, 7634, 11235, 313, 25103, 29965, 29897, 322, 18385, 17088, 28203, 313, 25103, 29954, 29897, 411, 975, 29871, 29941, 29906, 29974, 758, 3018, 1312, 4733, 297, 29871, 29896, 29900, 29900, 29974, 10276, 322, 6483, 1006, 3372, 3097, 1546, 435, 1165, 29892, 10772, 29911, 25350, 322, 323, 6073, 17907, 29889], [1, 350, 20161, 338, 8688, 304, 758, 29899, 14968, 6483, 21000, 8684, 284, 22540, 515, 443, 29880, 24025, 1426, 491, 14002, 368, 4195, 292, 373, 1716, 2175, 322, 1492, 3030, 297, 599, 15359, 29889], [1, 450, 4996, 17354, 1701, 29916, 432, 17204, 975, 278, 17366, 11203, 29889]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # fmt: skip

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="hf-internal-testing/llama-tokenizer",
            revision="0984d03108b1a041ed679bd253b6519b7e1a4778",
            padding=False,
        )

    def test_picklable(self):
        with tempfile.NamedTemporaryFile() as f:
            shutil.copyfile(SAMPLE_VOCAB, f.name)
            tokenizer = PreTrainedTokenizerFast(f.name, keep_accents=True)
            pickled_tokenizer = pickle.dumps(tokenizer)
        pickle.loads(pickled_tokenizer)

    def test_load_tokenizer_with_model_file_only(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            hf_hub_download(repo_id="huggyllama/llama-7b", filename="tokenizer.model", local_dir=tmp_dir)
            tokenizer_fast = self.rust_tokenizer_class.from_pretrained(tmp_dir)
            self.assertEqual(tokenizer_fast.encode("This is a test"), [1, 910, 338, 263, 1243])


@require_torch
@require_sentencepiece
@require_tokenizers
class MoshiIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_name = "hf-internal-testing/llama-tokenizer-non-normalized"
        cls.rust_tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        return cls

    @require_torch
    def integration_tests(self):
        inputs = self.tokenizer(
            ["The following string should be properly encoded: Hello.", "But ird and ปี   ird   ด"],
            return_tensors="pt",
        )

        self.assertEqual(
            nested_simplify(inputs),
            {
                "input_ids": [
                    [1, 450, 1494, 1347, 881, 367, 6284, 18511, 29901, 15043, 29889],
                    [1, 1205, 29871, 1823, 322, 29871, 31010, 30691, 1678, 1823, 1678, 30718],
                ],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            },
        )

    def test_fast_special_tokens(self):
        fast_tokenizer = self.rust_tokenizer

        fast_tokenizer.add_eos_token = False
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [1, 319, 4559, 1243]

        fast_tokenizer.add_eos_token = True
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [1, 319, 4559, 1243, 2]

        fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer", add_eos_token=True, add_bos_token=False
        )
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [319, 4559, 1243, 2]

        self.rust_tokenizer.add_eos_token = False

    def test_simple_encode_decode(self):
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(rust_tokenizer.encode("This is a test"), [1, 910, 338, 263, 1243])
        self.assertEqual(rust_tokenizer.decode([1, 910, 338, 263, 1243], skip_special_tokens=True), "This is a test")

        # bytefallback showcase
        self.assertEqual(rust_tokenizer.encode("生活的真谛是"), [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392])  # fmt: skip
        self.assertEqual(
            rust_tokenizer.decode(
                [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392], skip_special_tokens=True
            ),
            "生活的真谛是",
        )

        # Inner spaces showcase
        self.assertEqual(rust_tokenizer.encode("Hi  Hello"), [1, 6324, 29871, 15043])
        self.assertEqual(rust_tokenizer.decode([1, 6324, 29871, 15043], skip_special_tokens=True), "Hi  Hello")

        self.assertEqual(rust_tokenizer.encode("Hi   Hello"), [1, 6324, 259, 15043])
        self.assertEqual(rust_tokenizer.decode([1, 6324, 259, 15043], skip_special_tokens=True), "Hi   Hello")

        self.assertEqual(rust_tokenizer.encode(""), [1])

        self.assertEqual(rust_tokenizer.encode(" "), [1, 259])

        self.assertEqual(rust_tokenizer.encode("  "), [1, 1678])

        self.assertEqual(rust_tokenizer.encode(" Hello"), [1, 29871, 15043])

    def test_no_differences_showcase(self):
        rust_tokenizer = self.rust_tokenizer
        self.assertEqual(rust_tokenizer.encode(""), [1])

        self.assertEqual(rust_tokenizer.encode(" "), [1, 259])

        self.assertEqual(rust_tokenizer.encode("  "), [1, 1678])

        self.assertEqual(rust_tokenizer.encode(" Hello"), [1, 29871, 15043])

        self.assertEqual(rust_tokenizer.encode("<s>"), [1, 1])

    def test_no_differences_decode(self):
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(rust_tokenizer.decode([869]), ".")

        self.assertEqual(rust_tokenizer.decode([30112, 869]), "ا .")

    def test_no_differences_special_tokens(self):
        rust_tokenizer = self.rust_tokenizer
        self.assertEqual(rust_tokenizer.encode(""), [1])

        self.assertEqual(rust_tokenizer.encode("<s>"), [1, 1])

    def test_special_token_special_word(self):
        # the word inform should be split as ['in', 'form']
        tokenizer = PreTrainedTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=False, from_slow=True)
        tokenizer.add_tokens([AddedToken("<REPR_END>", rstrip=True, lstrip=True)], special_tokens=False)

        example_inputs = tokenizer.tokenize("<REPR_END>inform<s>. Hey.       .")
        self.assertEqual(example_inputs, ["<REPR_END>", "in", "form", "<s>", ".", "▁Hey", ".", "▁▁▁▁▁▁", "▁."])

        # Make sure dummy space is added if it is indeed the first word
        example_inputs = tokenizer.tokenize("inform<s>. Hey.       .")
        self.assertEqual(example_inputs, ["▁inform", "<s>", ".", "▁Hey", ".", "▁▁▁▁▁▁", "▁."])
        out1 = tokenizer.decode(
            tokenizer.encode("<REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=False
        )
        self.assertEqual(out1, "<REPR_END>inform")
        out2 = tokenizer.decode(
            tokenizer.encode("<REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=True
        )
        # decoding strips the added prefix space.
        self.assertEqual(out2, "<REPR_END>inform")
        input_ids = tokenizer.encode("<REPR_END>inform", add_special_tokens=False)
        self.assertEqual(input_ids, [32000, 262, 689])  # 29871 is the spiece underline, '▁' added as it should

        out2 = tokenizer.decode(
            tokenizer.encode(" <REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=False
        )
        # TODO @ArthurZ currently we strip left and right, so this will not keep the spaces
        self.assertEqual(out2, "<REPR_END>inform")

        ### Let's make sure decoding does not add extra spaces here and there
        # TODO @ArthurZ this should be affected by the lstrip/rstrip/single word /normalize refactoring
        # Since currently we always strip left and right of the token, results are as such
        input_ids = tokenizer.encode("<s> Hello<s>how", add_special_tokens=False)
        self.assertEqual(input_ids, [1, 15043, 1, 3525])
        tokens = tokenizer.tokenize("<s> Hello<s>how", add_special_tokens=False)
        self.assertEqual(tokens, ["<s>", "▁Hello", "<s>", "how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "<s> Hello<s>how")

        # Let's make sure that if there are any spaces, we don't remove them!
        input_ids = tokenizer.encode(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(input_ids, [29871, 1, 15043, 1, 920])
        tokens = tokenizer.tokenize(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(tokens, ["▁", "<s>", "▁Hello", "<s>", "▁how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "<s> Hello<s> how")

        # Let's make sure the space is preserved
        input_ids = tokenizer.encode("hello", add_special_tokens=True)
        self.assertEqual(input_ids, [1, 22172])
        tokens = tokenizer.tokenize("hello")
        self.assertEqual(tokens, ["▁hello"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "<s> hello")

        input_ids = tokenizer.encode("hello", add_special_tokens=False)
        self.assertEqual(input_ids, [22172])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "hello")

    def test_no_prefix_space(self):
        tokenizer_no_prefix_space = AutoTokenizer.from_pretrained("huggyllama/llama-7b", add_prefix_space=False)
        no_prefix_space_tokens = tokenizer_no_prefix_space.tokenize("Hey")
        self.assertEqual(no_prefix_space_tokens, ["H", "ey"])

        tokenizer = AutoTokenizer.from_pretrained(
            "huggyllama/llama-7b", legacy=False, from_slow=True, add_prefix_space=False
        )
        tokenizer.add_tokens([AddedToken("<REPR_END>", rstrip=True, lstrip=True)], special_tokens=False)

        example_inputs = tokenizer.tokenize("<REPR_END>inform<s>. Hey.       .")
        self.assertEqual(example_inputs, ["<REPR_END>", "in", "form", "<s>", ".", "▁Hey", ".", "▁▁▁▁▁▁", "▁."])

        # Make sure dummy space is added if it is indeed the first word
        example_inputs = tokenizer.tokenize("inform<s>. Hey.       .")
        self.assertEqual(example_inputs, ["in", "form", "<s>", ".", "▁Hey", ".", "▁▁▁▁▁▁", "▁."])
        out1 = tokenizer.decode(
            tokenizer.encode("<REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=False
        )
        self.assertEqual(out1, "<REPR_END>inform")
        out2 = tokenizer.decode(
            tokenizer.encode("<REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=True
        )
        # decoding strips the added prefix space.
        self.assertEqual(out2, "<REPR_END>inform")
        input_ids = tokenizer.encode("<REPR_END>inform", add_special_tokens=False)
        self.assertEqual(input_ids, [32000, 262, 689])  # 29871 is the spiece underline, '▁' added as it should

        out2 = tokenizer.decode(
            tokenizer.encode(" <REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=False
        )
        self.assertEqual(out2, "<REPR_END>inform")

        input_ids = tokenizer.encode("<s> Hello<s>how", add_special_tokens=False)
        self.assertEqual(input_ids, [1, 15043, 1, 3525])
        tokens = tokenizer.tokenize("<s> Hello<s>how", add_special_tokens=False)
        self.assertEqual(tokens, ["<s>", "▁Hello", "<s>", "how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "<s> Hello<s>how")

        # Let's make sure that if there are any spaces, we don't remove them!
        input_ids = tokenizer.encode(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(input_ids, [29871, 1, 15043, 1, 920])
        tokens = tokenizer.tokenize(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(tokens, ["▁", "<s>", "▁Hello", "<s>", "▁how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, " <s> Hello<s> how")

        # Let's make sure the space is preserved
        input_ids = tokenizer.encode("hello", add_special_tokens=True)
        self.assertEqual(input_ids, [1, 12199])
        tokens = tokenizer.tokenize("hello")
        self.assertEqual(tokens, ["hello"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "<s>hello")

        input_ids = tokenizer.encode("hello", add_special_tokens=False)
        self.assertEqual(input_ids, [12199])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "hello")

    def test_some_edge_cases(self):
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", legacy=False)

        sp_tokens = tokenizer.sp_model.encode("<s>>", out_type=str)
        self.assertEqual(sp_tokens, ["<", "s", ">>"])
        tokens = tokenizer.tokenize("<s>>")
        self.assertNotEqual(sp_tokens, tokens)
        self.assertEqual(tokens, ["<s>", ">"])

        tokens = tokenizer.tokenize("")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, tokenizer.sp_model.encode("", out_type=str))

        tokens = tokenizer.tokenize(" ")
        self.assertEqual(tokens, ["▁▁"])
        # a dummy prefix space is not added by the sp_model as it was de-activated
        self.assertEqual(tokens, tokenizer.sp_model.encode("  ", out_type=str))

        tokens = tokenizer.tokenize("▁")
        self.assertEqual(tokens, ["▁▁"])
        # a dummy prefix space is not added by the sp_model as it was de-activated
        self.assertEqual(tokens, tokenizer.sp_model.encode("▁▁", out_type=str))

        tokens = tokenizer.tokenize(" ▁")
        self.assertEqual(tokens, ["▁▁▁"])
        # a dummy prefix space is not added by the sp_model as it was de-activated
        self.assertEqual(tokens, tokenizer.sp_model.encode("▁▁▁", out_type=str))

    def test_fast_post_processor(self):
        tokenizer = PreTrainedTokenizerFast(
            SAMPLE_VOCAB, eos_token=None, bos_token=None, add_bos_token=False, add_eos_token=False
        )
        tokenizer.encode(" Hey ")

        with self.assertRaises(ValueError):
            tokenizer = PreTrainedTokenizerFast(
                SAMPLE_VOCAB, bos_token=None, eos_token="<s>", add_bos_token=True, add_eos_token=False
            )
        with self.assertRaises(ValueError):
            tokenizer = PreTrainedTokenizerFast(SAMPLE_VOCAB, eos_token=None, add_bos_token=True, add_eos_token=True)


@require_sentencepiece
@require_tokenizers
class CommonSpmIntegrationTests(unittest.TestCase):
    """
    A class that regroups important test to make sure that we properly handle the special tokens.
    """

    def test_edge_case_tabulation(self):
        fast_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/dummy-gemma")
        input_text = "Hey<eos>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61"
        EXPECTED_IDS = [ 2, 6750, 1, 235265, 235248, 255969, 235248, 109, 4747, 139, 235335, 139, 216311, 241316, 139, 239880, 235341, 144, 235269, 235248, 235274, 235284, 235304, 235310, 235248, 235274, 235308, 235248, 235308, 235269, 235318, 235274]  # fmt: skip
        EXPECTED_TOKENS = [ "Hey", "<eos>", ".", "▁", "\t\t", "▁", "\n\n", "you", "▁▁", "é", "▁▁", "@#", "😈", "▁▁", "🤗", "!", "▁▁▁▁▁▁▁", ",", "▁", "1", "2", "3", "4", "▁", "1", "5", "▁", "5", ",", "6", "1"]  # fmt: skip

        tokens = fast_tokenizer.tokenize(input_text)
        with self.subTest("test fast edge case fast"):
            self.assertEqual(tokens, EXPECTED_TOKENS)

        input_ids = fast_tokenizer.encode(input_text)
        with self.subTest("test fast edge case fast"):
            self.assertEqual(input_ids, EXPECTED_IDS)

        text = fast_tokenizer.decode(EXPECTED_IDS)
        with self.subTest("test fast edge case fast"):
            self.assertEqual(text, "<bos>Hey<eos>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61")

        input_text = "\t\t\t\t \n\n61"
        EXPECTED_IDS = [2, 255971, 235248, 109, 235318, 235274]
        EXPECTED_TOKENS = ["\t\t\t\t", "▁", "\n\n", "6", "1"]

        tokens = fast_tokenizer.tokenize(input_text)
        with self.subTest("test fast edge case fast"):
            self.assertEqual(tokens, EXPECTED_TOKENS)

        input_ids = fast_tokenizer.encode(input_text)
        with self.subTest("test fast edge case fast"):
            self.assertEqual(input_ids, EXPECTED_IDS)

        text = fast_tokenizer.decode(EXPECTED_IDS)
        with self.subTest("test fast edge case fast"):
            self.assertEqual(text, "<bos>\t\t\t\t \n\n61")
