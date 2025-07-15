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

import os
import tempfile
import unittest

from datasets import load_dataset

from transformers import (
    AddedToken,
    Ernie4_5Tokenizer,
    Ernie4_5TokenizerFast,
)
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_jinja,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
)


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


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

    def test_integration(self):
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

    def test_fast_special_tokens(self):
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

    def test_simple_encode_decode(self):
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
        self.assertEqual(
            pyth_tokenizer.encode(bytefallback_text),
            [25027, 93953, 92335, 6140, 93919, 22, 22, 93919, 23, 23, 5557, 269, 94039, 269, 94027, 93993, 253, 172, 165, 149, 269, 253, 172, 177, 164, 94018, 426, 1089, 93919, 4, 5, 6, 7, 93919, 4, 8, 93919, 8, 93938, 9, 4]  # fmt: skip
        )
        self.assertEqual(
            rust_tokenizer.encode(bytefallback_text),
            [25027, 93953, 92335, 6140, 93919, 22, 22, 93919, 23, 23, 5557, 269, 94039, 269, 94027, 93993, 253, 172, 165, 149, 269, 253, 172, 177, 164, 94018, 426, 1089, 93919, 4, 5, 6, 7, 93919, 4, 8, 93919, 8, 93938, 9, 4]  # fmt: skip
        )
        self.assertEqual(
            pyth_tokenizer.decode(
                [25027, 93953, 92335, 6140, 93919, 22, 22, 93919, 23, 23, 5557, 269, 94039, 269, 94027, 93993, 253, 172, 165, 149, 269, 253, 172, 177, 164, 94018, 426, 1089, 93919, 4, 5, 6, 7, 93919, 4, 8, 93919, 8, 93938, 9, 4],  # fmt: skip
            ),
            bytefallback_text,
        )
        self.assertEqual(
            rust_tokenizer.decode(
                [25027, 93953, 92335, 6140, 93919, 22, 22, 93919, 23, 23, 5557, 269, 94039, 269, 94027, 93993, 253, 172, 165, 149, 269, 253, 172, 177, 164, 94018, 426, 1089, 93919, 4, 5, 6, 7, 93919, 4, 8, 93919, 8, 93938, 9, 4],  # fmt: skip
            ),
            bytefallback_text,
        )

        # Inner spaces showcase
        self.assertEqual(pyth_tokenizer.encode("Hi  Hello"), [17475, 269, 16276])
        self.assertEqual(rust_tokenizer.encode("Hi  Hello"), [17475, 269, 16276])
        self.assertEqual(pyth_tokenizer.decode([17475, 269, 16276]), "Hi  Hello")
        self.assertEqual(rust_tokenizer.decode([17475, 269, 16276]), "Hi  Hello")

        self.assertEqual(pyth_tokenizer.encode("Hi   Hello"), [17475, 269, 30802])
        self.assertEqual(rust_tokenizer.encode("Hi   Hello"), [17475, 269, 30802])
        self.assertEqual(pyth_tokenizer.decode([17475, 269, 30802]), "Hi   Hello")
        self.assertEqual(rust_tokenizer.decode([17475, 269, 30802]), "Hi   Hello")

        self.assertEqual(pyth_tokenizer.encode(""), [])
        self.assertEqual(rust_tokenizer.encode(""), [])

        self.assertEqual(pyth_tokenizer.encode(" "), [93919])
        self.assertEqual(rust_tokenizer.encode(" "), [93919])

        self.assertEqual(pyth_tokenizer.encode("  "), [269])
        self.assertEqual(rust_tokenizer.encode("  "), [269])

        self.assertEqual(pyth_tokenizer.encode(" Hello"), [30802])
        self.assertEqual(rust_tokenizer.encode(" Hello"), [30802])

    # TODO: I'm here
    def test_no_differences_showcase(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer
        self.assertEqual(pyth_tokenizer.encode(""), [1])
        self.assertEqual(rust_tokenizer.encode(""), [1])

        self.assertEqual(pyth_tokenizer.encode(" "), [1, 259])
        self.assertEqual(rust_tokenizer.encode(" "), [1, 259])

        self.assertEqual(pyth_tokenizer.encode("  "), [1, 1678])
        self.assertEqual(rust_tokenizer.encode("  "), [1, 1678])

        self.assertEqual(pyth_tokenizer.encode(" Hello"), [1, 29871, 15043])
        self.assertEqual(rust_tokenizer.encode(" Hello"), [1, 29871, 15043])

        self.assertEqual(pyth_tokenizer.encode("<s>"), [1, 1])
        self.assertEqual(rust_tokenizer.encode("<s>"), [1, 1])

    def test_no_differences_decode(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(pyth_tokenizer.decode([869]), ".")
        self.assertEqual(rust_tokenizer.decode([869]), ".")

        self.assertEqual(pyth_tokenizer.decode([30112, 869]), "ا .")
        self.assertEqual(rust_tokenizer.decode([30112, 869]), "ا .")

    def test_no_differences_special_tokens(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer
        self.assertEqual(pyth_tokenizer.encode(""), [1])
        self.assertEqual(rust_tokenizer.encode(""), [1])

        self.assertEqual(pyth_tokenizer.encode("<s>"), [1, 1])
        self.assertEqual(rust_tokenizer.encode("<s>"), [1, 1])

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

    def test_special_token_special_word(self):
        # the word inform should be split as ['in', 'form']
        tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=False, from_slow=True)
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
        tokenizer_no_prefix_space = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", add_prefix_space=False)
        no_prefix_space_tokens = tokenizer_no_prefix_space.tokenize("Hey")
        self.assertEqual(no_prefix_space_tokens, ["H", "ey"])

        tokenizer = LlamaTokenizerFast.from_pretrained(
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
        tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b", legacy=False)

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
        tokenizer = LlamaTokenizerFast(
            SAMPLE_VOCAB, eos_token=None, bos_token=None, add_bos_token=False, add_eos_token=False
        )
        tokenizer.encode(" Hey ")

        with self.assertRaises(ValueError):
            tokenizer = LlamaTokenizerFast(
                SAMPLE_VOCAB, bos_token=None, eos_token="<s>", add_bos_token=True, add_eos_token=False
            )
        with self.assertRaises(ValueError):
            tokenizer = LlamaTokenizerFast(SAMPLE_VOCAB, eos_token=None, add_bos_token=True, add_eos_token=True)

    @require_jinja
    def test_tokenization_for_chat(self):
        tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b", legacy=False)

        test_chats = [
            [{"role": "system", "content": "You are a helpful chatbot."}, {"role": "user", "content": "Hello!"}],
            [
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Nice to meet you."},
            ],
            [{"role": "user", "content": "Hello!"}],
        ]
        # Matt: The third test case tests the default system message, but if this is ever changed in the
        #       class/repo code then that test will fail, and the case will need to be updated.
        tokenized_chats = [tokenizer.apply_chat_template(test_chat) for test_chat in test_chats]
        # fmt: off
        expected_tokens = [
            [1, 29961, 25580, 29962, 3532, 14816, 29903, 6778, 13, 3492, 526, 263, 8444, 13563, 7451, 29889, 13, 29966, 829, 14816, 29903, 6778, 13, 13, 10994, 29991, 518, 29914, 25580, 29962],
            [1, 29961, 25580, 29962, 3532, 14816, 29903, 6778, 13, 3492, 526, 263, 8444, 13563, 7451, 29889, 13, 29966, 829, 14816, 29903, 6778, 13, 13, 10994, 29991, 518, 29914, 25580, 29962, 20103, 304, 5870, 366, 29889, 29871, 2],
            [1, 29961, 25580, 29962, 15043, 29991, 518, 29914, 25580, 29962]
        ]
        # fmt: on
        for tokenized_chat, expected_tokens in zip(tokenized_chats, expected_tokens):
            self.assertListEqual(tokenized_chat, expected_tokens)
