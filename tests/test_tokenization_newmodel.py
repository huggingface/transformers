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

import os
import tempfile
import unittest

from datasets import load_dataset

from transformers import (
    AddedToken
)
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_jinja,
    require_read_token,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
)

from .test_tokenization_common import TokenizerTesterMixin

from transformers import PreTrainedTokenizerFast
from transformers.models.llama.tokenization_spm import SPMTokenizer
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class NewModelTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "local-gemma-7b"
    tokenizer_class = PreTrainedTokenizerFast
    rust_tokenizer_class = PreTrainedTokenizerFast

    test_rust_tokenizer = False
    test_sentencepiece = True
    from_pretrained_kwargs = {}

    tokenizer = SPMTokenizer.from_pretrained(
        SAMPLE_VOCAB,
        keep_accents=True,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        do_lower_case=False,
        add_bos_token=True,
    )

    sp_model = tokenizer.sp_model

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # We have a SentencePiece fixture for testing
        model_path = "/Users/itazaporozhets/Documents/Repos/transformers/local-gemma-7b/tokenizer.model"  # Replace with your actual model path

        tokenizer = SPMTokenizer.from_pretrained(
            SAMPLE_VOCAB,
            keep_accents=True,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            do_lower_case=False,
            add_bos_token=True,
        )

        tokenizer =  PreTrainedTokenizerFast(
            tokenizer_object=convert_slow_tokenizer(tokenizer),
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            do_lower_case=False,
            add_bos_token=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(cls.tmpdirname)

    @unittest.skip(reason="Unfortunately way too slow to build a BPE with SentencePiece.")
    def test_save_slow_from_fast_and_reload_fast(self):
        pass

    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.get_rust_tokenizer(
                    pretrained_name, additional_special_tokens=added_tokens, **kwargs
                )
                r_output = tokenizer_r.encode("Hey this is a <special> token")

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

                self.assertTrue(special_token_id in r_output)

                if self.test_slow_tokenizer:
                    tokenizer_cr = self.get_rust_tokenizer(
                        pretrained_name,
                        additional_special_tokens=added_tokens,
                        **kwargs,  # , from_slow=True <- unfortunately too slow to convert
                    )
                    tokenizer_p = self.tokenizer_class.from_pretrained(
                        pretrained_name, additional_special_tokens=added_tokens, **kwargs
                    )

                    p_output = tokenizer_p.encode("Hey this is a <special> token")

                    cr_output = tokenizer_cr.encode("Hey this is a <special> token")

                    self.assertEqual(p_output, r_output)
                    self.assertEqual(cr_output, r_output)
                    self.assertTrue(special_token_id in p_output)
                    self.assertTrue(special_token_id in cr_output)

    @slow
    @require_read_token
    def test_tokenizer_integration(self):
        expected_encoding =  {'input_ids': [[2, 158434, 591, 84193, 3836, 685, 6599, 31223, 235290, 140247, 578, 6599, 31223, 235290, 145139, 235290, 3491, 235275, 6572, 3311, 235290, 38197, 109959, 591, 25894, 235269, 162174, 235290, 235284, 235269, 1791, 6362, 12481, 235269, 1576, 18622, 235269, 2900, 1136, 86684, 235269, 29092, 4632, 16994, 604, 13146, 14944, 40371, 591, 19700, 235327, 235275, 578, 13146, 14944, 25511, 591, 235300, 12474, 235275, 675, 1163, 235248, 235304, 235284, 235340, 229903, 5377, 575, 235248, 235274, 235276, 235276, 235340, 17044, 578, 5271, 1061, 118345, 1865, 125247, 235269, 8745, 111226, 578, 176888, 235265], [2, 25894, 603, 6869, 577, 953, 235290, 8297, 5271, 209099, 41642, 774, 748, 78253, 2793, 731, 51506, 34346, 611, 2145, 2731, 578, 1833, 4807, 575, 832, 16630, 235265], [2, 651, 4320, 8426, 25341, 36271, 1163, 573, 27894, 5929, 235265]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # fmt: skip
        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="google/gemma-2b",
            padding=False,
        )

    @unittest.skip(reason="worker 'gw4' crashed on CI, passing locally.")
    def test_pickle_subword_regularization_tokenizer(self):
        pass

    @unittest.skip(reason="worker 'gw4' crashed on CI, passing locally.")
    def test_subword_regularization_tokenizer(self):
        pass

    @unittest.skip(reason="Skipping")
    def test_torch_encode_plus_sent_to_model(self):
        pass

    @unittest.skip(reason="dep in v5")
    def test_prepare_for_model(self):
        pass


@require_torch
@require_sentencepiece
@require_tokenizers
class NewModelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_name = "hf-internal-testing/dummy-gemma"
        tokenizer = SPMTokenizer.from_pretrained(
            "hf-internal-testing/dummy-gemma",
            keep_accents=True,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<s>",
            do_lower_case=False,
            add_bos_token=True,
        )
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=convert_slow_tokenizer(tokenizer),
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<s>",
            do_lower_case=False,
            add_bos_token=True,
        )
        cls.old_tokenizer = tokenizer
        cls.tokenizer = fast_tokenizer
        cls.rust_tokenizer = fast_tokenizer # add this token
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
                    [2, 450, 1494, 1347, 881, 367, 6284, 18511, 29901, 15043, 29889],
                    [2, 1205, 29871, 1823, 322, 29871, 31010, 30691, 1678, 1823, 1678, 30718],
                ],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            },
        )

    def test_user_added_tokens(self):
        # Ensure that user added tokens are not split in the fast tokenizer
        slow_tokenizer = self.tokenizer
        fast_tokenizer = self.rust_tokenizer

        user_added_token = "<mask>"

        slow_tokens = slow_tokenizer.convert_ids_to_tokens(slow_tokenizer.encode(user_added_token))
        fast_tokens = slow_tokenizer.convert_ids_to_tokens(fast_tokenizer.encode(user_added_token))

        self.assertTrue(user_added_token in fast_tokens)
        self.assertEqual(slow_tokens, fast_tokens)

    def test_fast_special_tokens(self):
        slow_tokenizer = self.tokenizer
        fast_tokenizer = self.rust_tokenizer
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)
        assert slow == [2, 235280, 6453, 2121]

        fast_tokenizer.add_eos_token = False
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [2, 235280, 6453, 2121]

        fast_tokenizer.add_eos_token = True
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [2, 235280, 6453, 2121, 204]

        slow_tokenizer.add_eos_token = True
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)
        assert slow == [2, 235280, 6453, 2121, 204]

        self.tokenizer.add_eos_token = False
        self.rust_tokenizer.add_eos_token = False

    def test_fast_merge_priority(self):
        slow_tokenizer = self.tokenizer
        fast_tokenizer = self.rust_tokenizer
        text = "                                               "
        target = [168, 153]
        slow = slow_tokenizer.encode(text, add_special_tokens=False)
        assert slow == target

        fast = fast_tokenizer.encode(text, add_special_tokens=False)
        assert fast == target

    @unittest.skip(reason="Not super important and always failing. Let's skip it")
    @slow
    def test_conversion(self):
        # This is excruciatingly slow since it has to recreate the entire merge
        # list from the original vocabulary in spm
        self.rust_tokenizer.save_pretrained("./out")
        with tempfile.TemporaryDirectory() as dirname:
            self.rust_tokenizer.save_pretrained(dirname)

            with open(os.path.join(dirname, "tokenizer.json")) as f:
                old_serialized = f.read()

        new_tokenizer = convert_slow_tokenizer(self.tokenizer)
        with tempfile.NamedTemporaryFile() as f:
            new_tokenizer.save(f.name)
            # Re-opening since `f` is in bytes.
            new_serialized = open(f.name).read()
            with open("out_tokenizer.json", "w") as g:
                g.write(new_serialized)

            self.assertEqual(old_serialized, new_serialized)

    def test_simple_encode_decode(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        self.tokenizer.add_eos_token = False
        self.rust_tokenizer.add_eos_token = False

        self.assertEqual(pyth_tokenizer.encode("This is a test"), [2, 1596, 603, 476, 2121])
        self.assertEqual(rust_tokenizer.encode("This is a test"), [2, 1596, 603, 476, 2121])
        self.assertEqual(pyth_tokenizer.decode([2, 1596, 603, 476, 2121], skip_special_tokens=True), "This is a test")
        self.assertEqual(rust_tokenizer.decode([2, 1596, 603, 476, 2121], skip_special_tokens=True), "This is a test")

        # bytefallback showcase
        self.assertEqual(pyth_tokenizer.encode("生活的真谛是"), [2, 122182, 235710, 245467, 235427] )  # fmt: skip
        self.assertEqual(rust_tokenizer.encode("生活的真谛是"), [2, 122182, 235710, 245467, 235427] )  # fmt: skip
        self.assertEqual(
            pyth_tokenizer.decode([2, 122182, 235710, 245467, 235427], skip_special_tokens=True),
            "生活的真谛是",
        )
        self.assertEqual(
            rust_tokenizer.decode([2, 122182, 235710, 245467, 235427], skip_special_tokens=True),
            "生活的真谛是",
        )

        # Inner spaces showcase
        self.assertEqual(pyth_tokenizer.encode("Hi  Hello"), [2, 2151, 139, 4521])
        self.assertEqual(rust_tokenizer.encode("Hi  Hello"), [2, 2151, 139, 4521])
        self.assertEqual(pyth_tokenizer.decode([2, 2151, 139, 4521], skip_special_tokens=True), "Hi  Hello")
        self.assertEqual(rust_tokenizer.decode([2, 2151, 139, 4521], skip_special_tokens=True), "Hi  Hello")

        self.assertEqual(pyth_tokenizer.encode("Hi   Hello"), [2, 2151, 140, 4521])
        self.assertEqual(rust_tokenizer.encode("Hi   Hello"), [2, 2151, 140, 4521])
        self.assertEqual(pyth_tokenizer.decode([2, 2151, 140, 4521], skip_special_tokens=True), "Hi   Hello")
        self.assertEqual(rust_tokenizer.decode([2, 2151, 140, 4521], skip_special_tokens=True), "Hi   Hello")

        self.assertEqual(pyth_tokenizer.encode(""), [2])
        self.assertEqual(rust_tokenizer.encode(""), [2])

        self.assertEqual(pyth_tokenizer.encode(" "), [2, 235248])
        self.assertEqual(rust_tokenizer.encode(" "), [2, 235248])

        self.assertEqual(pyth_tokenizer.encode("  "), [2, 139])
        self.assertEqual(rust_tokenizer.encode("  "), [2, 139])

        self.assertEqual(pyth_tokenizer.encode(" Hello"), [2, 25957])
        self.assertEqual(rust_tokenizer.encode(" Hello"), [2, 25957])

    def test_no_differences_decode(self):
        self.tokenizer.add_eos_token = False
        self.rust_tokenizer.add_eos_token = False
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(pyth_tokenizer.decode([869]), "og")
        self.assertEqual(rust_tokenizer.decode([869]), "og")

        self.assertEqual(pyth_tokenizer.decode([30112, 869]), " expenditureog")
        self.assertEqual(rust_tokenizer.decode([30112, 869]), " expenditureog")

    def test_no_differences_special_tokens(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer
        self.assertEqual(pyth_tokenizer.encode(""), [2])
        self.assertEqual(rust_tokenizer.encode(""), [2])

        self.assertEqual(pyth_tokenizer.encode("<s>"), [2, 204])
        self.assertEqual(rust_tokenizer.encode("<s>"), [2, 204])

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

            self.assertEqual(
                encoded1,
                encoded2,
                msg="Hint: the following tokenization diff were obtained for slow vs fast:\n "
                f"elements in slow: {set(pyth_tokenizer.tokenize(string)) - set(rust_tokenizer.tokenize(string))} \nvs\n "
                f"elements in fast: {set(rust_tokenizer.tokenize(string)) - set(pyth_tokenizer.tokenize(string))} \n\n{string}",
            )

            decoded1 = pyth_tokenizer.decode(encoded1, skip_special_tokens=True)
            decoded2 = rust_tokenizer.decode(encoded1, skip_special_tokens=True)

            self.assertEqual(decoded1, decoded2)

        dataset = load_dataset("facebook/xnli", "all_languages")

        for item in tqdm.tqdm(dataset["train"]):
            for string in item["premise"].values():
                encoded1 = pyth_tokenizer.encode(string)
                encoded2 = rust_tokenizer.encode(string)

                self.assertEqual(encoded1, encoded2, msg=f"failed on {string}")

                decoded1 = pyth_tokenizer.decode(encoded1, skip_special_tokens=True)
                decoded2 = rust_tokenizer.decode(encoded2, skip_special_tokens=True)

                self.assertEqual(decoded1, decoded2)

    def test_some_edge_cases(self):
        tokenizer = self.tokenizer

        tokens = tokenizer.tokenize("<s>>")
        self.assertEqual(tokens, ["<s>", ">"])

        tokens = tokenizer.tokenize("")
        self.assertEqual(tokens, [])
        self.assertEqual(tokens, self.old_tokenizer.sp_model.encode("", out_type=str))

        tokens = tokenizer.tokenize(" ")
        self.assertEqual(tokens, ["▁"])
        # a dummy prefix space is not added by the sp_model as it was de-activated
        self.assertEqual(tokens, self.old_tokenizer.sp_model.encode(" ", out_type=str))

        tokens = tokenizer.tokenize("▁")
        self.assertEqual(tokens, ["▁"])
        # a dummy prefix space is not added by the sp_model as it was de-activated
        self.assertEqual(tokens, self.old_tokenizer.sp_model.encode("▁", out_type=str))

        tokens = tokenizer.tokenize(" ▁")
        self.assertEqual(tokens, ["▁▁"])
        # a dummy prefix space is not added by the sp_model as it was de-activated
        self.assertEqual(tokens, self.old_tokenizer.sp_model.encode("▁▁", out_type=str))


@require_sentencepiece
@require_tokenizers
class CommonSpmIntegrationTests(unittest.TestCase):
    """
    A class that regroups important test to make sure that we properly handle the special tokens.
    """
    tokenizer = SPMTokenizer.from_pretrained(
        SAMPLE_VOCAB,
        keep_accents=True,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        do_lower_case=False,
        add_bos_token=True,
    )

    def test_edge_case_tabulation(self):
        tokenizer = SPMTokenizer.from_pretrained(
            "hf-internal-testing/dummy-gemma",
            keep_accents=True,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            do_lower_case=False,
            add_bos_token=True,
        )
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=convert_slow_tokenizer(tokenizer),
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            do_lower_case=False,
            add_bos_token=True,
        )
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
