# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
    is_torch_available,
)
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
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


if is_torch_available():
    pass


@require_sentencepiece
@require_tokenizers
class OLMoTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "allenai/OLMo-1B"

    # `tokenizer_class` is normally supposed to be a slow tokenizer. It is set to the fast tokenizer
    # because there is no slow OLMo tokenizer and some fast tests still expect this to be set.
    tokenizer_class = GPTNeoXTokenizerFast
    rust_tokenizer_class = GPTNeoXTokenizerFast

    test_slow_tokenizer = False
    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        tokenizer = GPTNeoXTokenizerFast.from_pretrained("allenai/OLMo-1B")
        tokenizer.save_pretrained(self.tmpdirname)

    def test_full_tokenizer(self):
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("allenai/OLMo-1B")

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["This", "Ġis", "Ġa", "Ġtest"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [1552, 310, 247, 1071],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            ["I", "Ġwas", "Ġborn", "Ġin", "Ġ9", "2000", ",", "Ġand", "Ġthis", "Ġis", "Ġfals", "Ã©", "."],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [42, 369, 5686, 275, 898, 6914, 13, 285, 436, 310, 21649, 860, 15],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            ["I", "Ġwas", "Ġborn", "Ġin", "Ġ9", "2000", ",", "Ġand", "Ġthis", "Ġis", "Ġfals", "Ã©", "."],
        )

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
        expected_encoding = {'input_ids': [[22904, 398, 313, 42889, 1929, 347, 268, 1767, 263, 348, 14, 16702, 398, 285, 268, 1767, 263, 348, 14, 4025, 11273, 14, 6291, 10, 3400, 2087, 14, 27299, 35615, 313, 35, 6366, 13, 443, 5736, 14, 19, 13, 8741, 35, 6366, 66, 13, 1594, 22047, 13, 3656, 300, 49340, 13, 35974, 8695, 19552, 323, 14673, 18847, 31293, 313, 19214, 54, 10, 285, 14673, 18847, 28598, 313, 19214, 40, 10, 342, 689, 4567, 12, 3215, 11273, 3210, 275, 2233, 12, 11515, 285, 3676, 734, 2211, 1430, 875, 500, 991, 13, 8462, 22097, 348, 285, 41529, 20671, 15, 50279], [35, 6366, 310, 4158, 281, 638, 14, 24382, 3676, 12246, 30869, 14237, 432, 440, 22027, 2505, 407, 26277, 21839, 327, 1097, 1669, 285, 987, 3634, 275, 512, 8090, 15, 50279], [510, 3158, 8516, 30013, 27287, 689, 253, 22658, 4370, 15, 50279]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # fmt: skip

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="allenai/OLMo-1B",
            padding=False,
        )

    # This is the same as `TokenizerTesterMixin.test_encode_decode_with_spaces` except we include the fast
    # tokenizer since that's all we have for OLMo.
    def test_encode_decode_with_spaces(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                new_toks = [
                    # These are added tokens, they will be normalized....
                    AddedToken("[ABC]", normalized=True, lstrip=True, rstrip=True),
                    AddedToken("[DEF]", normalized=True, lstrip=True, rstrip=True),
                    AddedToken("GHI IHG", normalized=True, lstrip=True, rstrip=True),
                ]
                tokenizer.add_tokens(new_toks)
                tokenizer.add_tokens([AddedToken("[SAMPLE]", normalized=True)], special_tokens=True)
                inp = "[ABC][DEF][ABC]GHI IHG[DEF]"
                if self.space_between_special_tokens:
                    output = "[ABC] [DEF] [ABC] GHI IHG [DEF]"
                else:
                    output = inp
                encoded = tokenizer.encode(inp, add_special_tokens=False)
                decoded = tokenizer.decode(encoded, spaces_between_special_tokens=self.space_between_special_tokens)

                self.assertIn(decoded, [output, output.lower()])
                return


@require_torch
@require_tokenizers
class OLMoIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_name = "allenai/OLMo-1B"
        cls.rust_tokenizer = GPTNeoXTokenizerFast.from_pretrained(checkpoint_name)
        return cls

    @require_torch
    def integration_tests(self):
        inputs = self.rust_tokenizer(
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
        assert fast == [34, 3410, 1071]

        fast_tokenizer.add_eos_token = True
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [34, 3410, 1071, 50279]

    def test_simple_encode_decode(self):
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(rust_tokenizer.encode("This is a test"), [1552, 310, 247, 1071, 50279])
        self.assertEqual(
            rust_tokenizer.decode([1552, 310, 247, 1071, 50279], skip_special_tokens=True), "This is a test"
        )

        # bytefallback showcase
        self.assertEqual(rust_tokenizer.encode("生活的真谛是"), [20025, 46549, 5225, 48561, 33656, 238, 12105, 50279])  # fmt: skip
        self.assertEqual(
            rust_tokenizer.decode([20025, 46549, 5225, 48561, 33656, 238, 12105, 50279], skip_special_tokens=True),
            "生活的真谛是",
        )

        # Inner spaces showcase
        self.assertEqual(rust_tokenizer.encode("Hi  Hello"), [12764, 50276, 12092, 50279])
        self.assertEqual(rust_tokenizer.decode([12764, 50276, 12092, 50279], skip_special_tokens=True), "Hi  Hello")

        self.assertEqual(rust_tokenizer.encode("Hi   Hello"), [12764, 50275, 12092, 50279])
        self.assertEqual(rust_tokenizer.decode([12764, 50275, 12092, 50279], skip_special_tokens=True), "Hi   Hello")

        self.assertEqual(rust_tokenizer.encode(""), [50279])

        self.assertEqual(rust_tokenizer.encode(" "), [209, 50279])

        self.assertEqual(rust_tokenizer.encode("  "), [50276, 50279])

        self.assertEqual(rust_tokenizer.encode(" Hello"), [24387, 50279])
