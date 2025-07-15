# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import unittest

from datasets import load_dataset

from transformers import (
    AddedToken,
    Ernie4_5Tokenizer,
    Ernie4_5TokenizerFast,
)
from transformers.testing_utils import (
    nested_simplify,
    require_jinja,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
)


@require_torch
@require_sentencepiece
@require_tokenizers
class Ernie4_5IntegrationTest(unittest.TestCase):
    checkpoint_name = "AntonV/Ernie4_5Tokenizer"

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Ernie4_5Tokenizer.from_pretrained(cls.checkpoint_name)
        cls.rust_tokenizer = Ernie4_5TokenizerFast.from_pretrained(cls.checkpoint_name)
        return cls

    @unittest.skipIf(
        os.getenv("RUN_TOKENIZER_INTEGRATION", "0") == "0",
        "RUN_TOKENIZER_INTEGRATION=1 to run tokenizer integration tests",
    )
    def test_integration_test_xnli(self):
        import tqdm

        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        dataset = load_dataset("google/code_x_glue_ct_code_to_text", "go")
        for item in tqdm.tqdm(dataset["validation"]):
            string = item["code"]
            encoded1 = pyth_tokenizer.encode(string)
            encoded2 = rust_tokenizer.encode(string)

            self.assertEqual(encoded1, encoded2)

            decoded1 = pyth_tokenizer.decode(encoded1, skip_special_tokens=True)
            decoded2 = rust_tokenizer.decode(encoded2, skip_special_tokens=True)

            self.assertEqual(decoded1, decoded2)

        dataset = load_dataset("facebook/xnli", "all_languages")

        for item in tqdm.tqdm(dataset["train"]):
            for string in item["premise"].values():
                encoded1 = pyth_tokenizer.encode(string)
                encoded2 = rust_tokenizer.encode(string)

                self.assertEqual(encoded1, encoded2)

                decoded1 = pyth_tokenizer.decode(encoded1, skip_special_tokens=True)
                decoded2 = rust_tokenizer.decode(encoded2, skip_special_tokens=True)

                self.assertEqual(decoded1, decoded2)

    def test_small_integration(self):
        inputs = self.tokenizer(
            ["The following string should be properly encoded: Hello.", "But ird and ปี   ird   ด"],
            return_tensors="pt",
        )

        self.assertEqual(
            nested_simplify(inputs),
            {
                "input_ids": [
                    [700, 1997, 1530, 1189, 390, 10962, 20622, 93963, 30802, 93937],
                    [2398, 93919, 2080, 326, 93919, 81175, 1111, 2080, 1111, 95676],
                ],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            },
        )

    def test_add_special_tokens(self):
        slow_tokenizer = Ernie4_5Tokenizer.from_pretrained(self.checkpoint_name, add_bos_token=True)
        fast_tokenizer = Ernie4_5TokenizerFast.from_pretrained(self.checkpoint_name, add_bos_token=True)

        slow_tokenizer.add_eos_token = False
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)
        assert slow == [1, 93957, 3983, 1167]

        slow_tokenizer.add_eos_token = True
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)
        assert slow == [1, 93957, 3983, 1167, 2]

        fast_tokenizer.add_eos_token = False
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [1, 93957, 3983, 1167]

        fast_tokenizer.add_eos_token = True
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [1, 93957, 3983, 1167, 2]

        slow_tokenizer.add_bos_token = False
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)
        assert slow == [93957, 3983, 1167, 2]

        fast_tokenizer.add_bos_token = False
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [93957, 3983, 1167, 2]

    def test_encode_decode(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        # base showcase
        basic_text = "This is a test"
        self.assertEqual(pyth_tokenizer.encode(basic_text), [1720, 357, 274, 1167])
        self.assertEqual(rust_tokenizer.encode(basic_text), [1720, 357, 274, 1167])
        self.assertEqual(pyth_tokenizer.decode([1, 1720, 357, 274, 1167], skip_special_tokens=True), basic_text)
        self.assertEqual(rust_tokenizer.decode([1, 1720, 357, 274, 1167], skip_special_tokens=True), basic_text)

        # bytefallback showcase
        bytefallback_text = "Hey<eos>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61"
        # fmt: off
        self.assertEqual(
            pyth_tokenizer.encode(bytefallback_text),
            [25027, 93953, 92335, 6140, 93919, 22, 22, 93919, 23, 23, 5557, 269, 94039, 269, 94027, 93993, 253, 172, 165, 149, 269, 253, 172, 177, 164, 94018, 426, 1089, 93919, 4, 5, 6, 7, 93919, 4, 8, 93919, 8, 93938, 9, 4,],
        )
        self.assertEqual(
            rust_tokenizer.encode(bytefallback_text),
            [25027, 93953, 92335, 6140, 93919, 22, 22, 93919, 23, 23, 5557, 269, 94039, 269, 94027, 93993, 253, 172, 165, 149, 269, 253, 172, 177, 164, 94018, 426, 1089, 93919, 4, 5, 6, 7, 93919, 4, 8, 93919, 8, 93938, 9, 4,],
        )
        self.assertEqual(
            pyth_tokenizer.decode(
                [25027, 93953, 92335, 6140, 93919, 22, 22, 93919, 23, 23, 5557, 269, 94039, 269, 94027, 93993, 253, 172, 165, 149, 269, 253, 172, 177, 164, 94018, 426, 1089, 93919, 4, 5, 6, 7, 93919, 4, 8, 93919, 8, 93938, 9, 4,],
            ),
            bytefallback_text,
        )
        self.assertEqual(
            rust_tokenizer.decode(
                [25027, 93953, 92335, 6140, 93919, 22, 22, 93919, 23, 23, 5557, 269, 94039, 269, 94027, 93993, 253, 172, 165, 149, 269, 253, 172, 177, 164, 94018, 426, 1089, 93919, 4, 5, 6, 7, 93919, 4, 8, 93919, 8, 93938, 9, 4,],
            ),
            bytefallback_text,
        )
        # fmt: on

    def test_no_differences_showcase(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer
        self.assertEqual(pyth_tokenizer.encode(""), [])
        self.assertEqual(rust_tokenizer.encode(""), [])

        self.assertEqual(pyth_tokenizer.encode(" "), [93919])
        self.assertEqual(rust_tokenizer.encode(" "), [93919])

        self.assertEqual(pyth_tokenizer.encode("  "), [269])
        self.assertEqual(rust_tokenizer.encode("  "), [269])

        self.assertEqual(pyth_tokenizer.encode(" Hello"), [30802])
        self.assertEqual(rust_tokenizer.encode(" Hello"), [30802])

        self.assertEqual(pyth_tokenizer.encode("<s>"), [1])
        self.assertEqual(rust_tokenizer.encode("<s>"), [1])

    def test_no_differences_decode(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(pyth_tokenizer.decode([93937]), ".")
        self.assertEqual(rust_tokenizer.decode([93937]), ".")

        self.assertEqual(pyth_tokenizer.decode([94014, 663]), "ا .")
        self.assertEqual(rust_tokenizer.decode([94014, 663]), "ا .")

    def test_special_token_special_word(self):
        # the word "lekacy" should be split as ['lek', 'acy']
        tokenizer = Ernie4_5Tokenizer.from_pretrained(self.checkpoint_name, add_prefix_space=True)
        tokenizer.add_tokens([AddedToken("<REPR_END>", rstrip=True, lstrip=True)], special_tokens=False)

        example_inputs = tokenizer.tokenize("<REPR_END>lekacy<s>. Hey.       .")
        self.assertEqual(example_inputs, ["<REPR_END>", "lek", "acy", "<s>", ".", "▁Hey", ".", "▁▁▁▁▁▁", "▁."])

        input_ids = tokenizer.encode("<REPR_END>lekacy", add_special_tokens=False)
        self.assertEqual(input_ids, [101304, 17289, 3004])

        # Make sure dummy space is added if it is indeed the first word
        example_inputs = tokenizer.tokenize(" legacy<s>. Hey.       .")
        self.assertEqual(example_inputs, ["▁legacy", "<s>", ".", "▁Hey", ".", "▁▁▁▁▁▁", "▁."])

        out1 = tokenizer.decode(
            tokenizer.encode("<REPR_END>lekacy", add_special_tokens=False), spaces_between_special_tokens=False
        )
        self.assertEqual(out1, "<REPR_END>lekacy")
        out2 = tokenizer.decode(
            tokenizer.encode("<REPR_END>lekacy", add_special_tokens=False), spaces_between_special_tokens=True
        )
        self.assertEqual(out2, "<REPR_END>lekacy")
        # decoding strips the added prefix space.
        out2 = tokenizer.decode(
            tokenizer.encode(" <REPR_END>lekacy", add_special_tokens=False), spaces_between_special_tokens=False
        )
        self.assertEqual(out2, "<REPR_END>lekacy")

        ### Let's make sure decoding does not add extra spaces here and there
        # Since currently we always strip left and right of the token, results are as such
        input_ids = tokenizer.encode("<s> Hello<s>how", add_special_tokens=False)
        self.assertEqual(input_ids, [1, 30802, 1, 7413])
        tokens = tokenizer.tokenize("<s> Hello<s>how", add_special_tokens=False)
        self.assertEqual(tokens, ["<s>", "▁Hello", "<s>", "how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "<s> Hello<s>how")

        # Let's make sure that if there are any spaces, we don't remove them!
        input_ids = tokenizer.encode(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(input_ids, [93919, 1, 30802, 1, 1299])
        tokens = tokenizer.tokenize(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(tokens, ["▁", "<s>", "▁Hello", "<s>", "▁how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, " <s> Hello<s> how")

        # Let's make sure the space is preserved
        # NOTE: legacy behavior, so space is not preserved anymore
        input_ids = tokenizer.encode("hello", add_special_tokens=True)
        self.assertEqual(input_ids, [18830])
        tokens = tokenizer.tokenize("hello")
        self.assertEqual(tokens, ["hello"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "hello")

    def test_no_prefix_space(self):
        tokenizer_no_prefix_space = Ernie4_5TokenizerFast.from_pretrained(self.checkpoint_name, add_prefix_space=False)
        no_prefix_space_tokens = tokenizer_no_prefix_space.tokenize("lekacy")
        self.assertEqual(no_prefix_space_tokens, ["lek", "acy"])

        tokenizer = Ernie4_5TokenizerFast.from_pretrained(
            self.checkpoint_name, legacy=False, from_slow=True, add_prefix_space=False
        )
        tokenizer.add_tokens([AddedToken("<REPR_END>", rstrip=True, lstrip=True)], special_tokens=False)

        example_inputs = tokenizer.tokenize("<REPR_END>lekacy<s>. Hey.       .")
        self.assertEqual(example_inputs, ["<REPR_END>", "lek", "acy", "<s>", ".", "▁Hey", ".", "▁▁▁▁▁▁", "▁."])

        input_ids = tokenizer.encode("<REPR_END>lekacy", add_special_tokens=False)
        self.assertEqual(input_ids, [101304, 17289, 3004])

        # Make sure dummy space is added if it is indeed the first word
        example_inputs = tokenizer.tokenize("lekacy<s>. Hey.       .")
        self.assertEqual(example_inputs, ["lek", "acy", "<s>", ".", "▁Hey", ".", "▁▁▁▁▁▁", "▁."])

        out1 = tokenizer.decode(
            tokenizer.encode("<REPR_END>lekacy", add_special_tokens=False), spaces_between_special_tokens=False
        )
        self.assertEqual(out1, "<REPR_END>lekacy")
        out2 = tokenizer.decode(
            tokenizer.encode("<REPR_END>lekacy", add_special_tokens=False), spaces_between_special_tokens=True
        )
        self.assertEqual(out2, "<REPR_END>lekacy")
        # decoding strips the added prefix space.
        out2 = tokenizer.decode(
            tokenizer.encode(" <REPR_END>lekacy", add_special_tokens=False), spaces_between_special_tokens=False
        )
        self.assertEqual(out2, "<REPR_END>lekacy")

        input_ids = tokenizer.encode("<s> Hello<s>how", add_special_tokens=False)
        self.assertEqual(input_ids, [1, 30802, 1, 7413])
        tokens = tokenizer.tokenize("<s> Hello<s>how", add_special_tokens=False)
        self.assertEqual(tokens, ["<s>", "▁Hello", "<s>", "how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "<s> Hello<s>how")

        # Let's make sure that if there are any spaces, we don't remove them!
        input_ids = tokenizer.encode(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(input_ids, [93919, 1, 30802, 1, 1299])
        tokens = tokenizer.tokenize(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(tokens, ["▁", "<s>", "▁Hello", "<s>", "▁how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, " <s> Hello<s> how")

        # Let's make sure the space is preserved
        input_ids = tokenizer.encode("hello", add_special_tokens=False)
        self.assertEqual(input_ids, [18830])
        tokens = tokenizer.tokenize("hello")
        self.assertEqual(tokens, ["hello"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, "hello")

    def test_some_edge_cases(self):
        tokenizer = Ernie4_5Tokenizer.from_pretrained(self.checkpoint_name, from_slow=True)

        sp_tokens = tokenizer.sp_model.encode("<s>>", out_type=str)
        self.assertEqual(sp_tokens, ["<", "s", ">>"])
        tokens = tokenizer.tokenize("<s>>")
        self.assertNotEqual(sp_tokens, tokens)
        self.assertEqual(tokens, ["<s>", ">"])

        tokens = tokenizer.tokenize("")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, tokenizer.sp_model.encode("", out_type=str))

        tokens = tokenizer.tokenize(" ")
        self.assertEqual(tokens, ["▁"])
        self.assertEqual(tokens, tokenizer.sp_model.encode(" ", out_type=str))

        tokens = tokenizer.tokenize("▁")
        self.assertEqual(tokens, ["▁"])
        self.assertEqual(tokens, tokenizer.sp_model.encode("▁", out_type=str))

        tokens = tokenizer.tokenize(" ▁")
        self.assertEqual(tokens, ["▁▁"])
        self.assertEqual(tokens, tokenizer.sp_model.encode("▁▁", out_type=str))

    @require_jinja
    def test_tokenization_for_chat(self):
        tokenizer = Ernie4_5TokenizerFast.from_pretrained(self.checkpoint_name)

        test_chats = [
            [{"role": "system", "content": "You are a helpful chatbot."}, {"role": "user", "content": "Hello!"}],
            [
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Nice to meet you."},
            ],
            [{"role": "user", "content": "Hello!"}],
        ]
        tokenized_chats = [tokenizer.apply_chat_template(test_chat) for test_chat in test_chats]
        # fmt: off
        expected_tokens = [
            [100273, 2520, 524, 274, 20472, 18819, 7074, 93937, 23, 2969, 93963, 30802, 94018, 23],
            [100273, 2520, 524, 274, 20472, 18819, 7074, 93937, 23, 2969, 93963, 30802, 94018, 23, 92267, 93963, 47434, 318, 7472, 536, 93937, 100272],
            [100273, 2969, 93963, 30802, 94018, 23]
        ]
        # fmt: on
        for tokenized_chat, expected_tokens in zip(tokenized_chats, expected_tokens):
            self.assertListEqual(tokenized_chat, expected_tokens)
