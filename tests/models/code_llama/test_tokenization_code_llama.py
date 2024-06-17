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

import os
import pickle
import shutil
import tempfile
import unittest

from datasets import load_dataset

from transformers import (
    SPIECE_UNDERLINE,
    AddedToken,
    CodeLlamaTokenizer,
    CodeLlamaTokenizerFast,
    is_torch_available,
)
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
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
class CodeLlamaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "hf-internal-testing/llama-code-tokenizer"
    tokenizer_class = CodeLlamaTokenizer
    rust_tokenizer_class = CodeLlamaTokenizerFast
    test_rust_tokenizer = False
    test_sentencepiece = True
    from_pretrained_kwargs = {}

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = CodeLlamaTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(self.tmpdirname)

    def get_tokenizers(self, **kwargs):
        kwargs.update({"pad_token": "<PAD>"})
        return super().get_tokenizers(**kwargs)

    def test_no_infilling_init(self):
        tokenizer = CodeLlamaTokenizer(SAMPLE_VOCAB, prefix_token=None, keep_accents=True)
        with self.assertRaises(ValueError):
            tokenizer.tokenize("This is <FILL_ME> prefix")

    def test_full_tokenizer(self):
        tokenizer = CodeLlamaTokenizer(SAMPLE_VOCAB, keep_accents=True)

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

    def test_save_pretrained(self):
        self.tokenizers_list = [
            (self.rust_tokenizer_class, "hf-internal-testing/llama-code-tokenizer", {}),
            (self.tokenizer_class, "hf-internal-testing/llama-code-tokenizer", {}),
            (self.tokenizer_class, "codellama/CodeLlama-34b-Instruct-hf", {}),
            (self.rust_tokenizer_class, "codellama/CodeLlama-34b-Instruct-hf", {}),
        ]
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)
                tokenizer_p = self.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it save with the same files + the tokenizer.json file for the fast one
                self.assertTrue(any("tokenizer.json" in f for f in tokenizer_r_files))
                tokenizer_r_files = tuple(f for f in tokenizer_r_files if "tokenizer.json" not in f)
                self.assertSequenceEqual(tokenizer_r_files, tokenizer_p_files)

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

                # Save tokenizer rust, legacy_format=True
                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2, legacy_format=True)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it save with the same files
                self.assertSequenceEqual(tokenizer_r_files, tokenizer_p_files)

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

                # Save tokenizer rust, legacy_format=False
                tmpdirname2 = tempfile.mkdtemp()

                tokenizer_r_files = tokenizer_r.save_pretrained(tmpdirname2, legacy_format=False)
                tokenizer_p_files = tokenizer_p.save_pretrained(tmpdirname2)

                # Checks it saved the tokenizer.json file
                self.assertTrue(any("tokenizer.json" in f for f in tokenizer_r_files))

                # Checks everything loads correctly in the same way
                tokenizer_rp = tokenizer_r.from_pretrained(tmpdirname2)
                tokenizer_pp = tokenizer_p.from_pretrained(tmpdirname2)

                # Check special tokens are set accordingly on Rust and Python
                for key in tokenizer_pp.special_tokens_map:
                    self.assertTrue(hasattr(tokenizer_rp, key))

                shutil.rmtree(tmpdirname2)

    @require_torch
    def test_batch_tokenization(self):
        if not self.test_seq2seq:
            return

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
                    return
                self.assertEqual(batch.input_ids.shape[1], 3)
                # max_target_length will default to max_length if not specified
                batch = tokenizer(text, max_length=3, return_tensors="pt")
                self.assertEqual(batch.input_ids.shape[1], 3)

                batch_encoder_only = tokenizer(text=text, max_length=3, max_target_length=10, return_tensors="pt")
                self.assertEqual(batch_encoder_only.input_ids.shape[1], 3)
                self.assertEqual(batch_encoder_only.attention_mask.shape[1], 3)
                self.assertNotIn("decoder_input_ids", batch_encoder_only)

    @unittest.skip("Unfortunately way too slow to build a BPE with SentencePiece.")
    def test_save_slow_from_fast_and_reload_fast(self):
        pass

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

                if self.test_slow_tokenizer:
                    tokenizer_cr = self.rust_tokenizer_class.from_pretrained(
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
    def test_tokenizer_integration(self):
        expected_encoding = {'input_ids': [[1, 4103, 689, 414, 313, 24784, 368, 2998, 408, 282, 3637, 25350, 29899, 9067, 414, 322, 282, 3637, 25350, 29899, 1457, 3018, 1312, 29899, 2151, 29897, 8128, 2498, 29899, 15503, 4220, 6956, 1973, 313, 13635, 29911, 29892, 402, 7982, 29899, 29906, 29892, 1528, 13635, 29911, 29874, 29892, 1060, 26369, 29892, 6652, 309, 29933, 814, 29892, 1060, 29931, 6779, 11410, 363, 18385, 17088, 7634, 11235, 313, 25103, 29965, 29897, 322, 18385, 17088, 28203, 313, 25103, 29954, 29897, 411, 975, 29871, 29941, 29906, 29974, 758, 3018, 1312, 4733, 297, 29871, 29896, 29900, 29900, 29974, 10276, 322, 6483, 1006, 3372, 3097, 1546, 435, 1165, 29892, 10772, 29911, 25350, 322, 323, 6073, 17907, 29889], [1, 350, 20161, 338, 8688, 304, 758, 29899, 14968, 6483, 21000, 8684, 284, 22540, 515, 443, 29880, 24025, 1426, 491, 14002, 368, 4195, 292, 373, 1716, 2175, 322, 1492, 3030, 297, 599, 15359, 29889], [1, 450, 4996, 17354, 1701, 29916, 432, 17204, 975, 278, 17366, 11203, 29889]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}  # fmt: skip

        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="hf-internal-testing/llama-code-tokenizer",
            revision="6eb30c03ab6a9e2cdef4d523024909ec815ddb75",
            padding=False,
        )

    def test_picklable(self):
        with tempfile.NamedTemporaryFile() as f:
            shutil.copyfile(SAMPLE_VOCAB, f.name)
            tokenizer = CodeLlamaTokenizer(f.name, keep_accents=True)
            pickled_tokenizer = pickle.dumps(tokenizer)
        pickle.loads(pickled_tokenizer)

    @unittest.skip("worker 'gw4' crashed on CI, passing locally.")
    def test_pickle_subword_regularization_tokenizer(self):
        pass

    @unittest.skip("worker 'gw4' crashed on CI, passing locally.")
    def test_subword_regularization_tokenizer(self):
        pass


@require_torch
@require_sentencepiece
@require_tokenizers
class LlamaIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_name = "hf-internal-testing/llama-code-tokenizer"
        cls.tokenizer: CodeLlamaTokenizer = CodeLlamaTokenizer.from_pretrained(checkpoint_name)
        cls.rust_tokenizer = CodeLlamaTokenizerFast.from_pretrained(checkpoint_name)
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
        slow_tokenizer = self.tokenizer
        fast_tokenizer = self.rust_tokenizer
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)
        assert slow == [1, 319, 4559, 1243]

        fast_tokenizer.add_eos_token = False
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [1, 319, 4559, 1243]

        fast_tokenizer.add_eos_token = True
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [1, 319, 4559, 1243, 2]

        slow_tokenizer.add_eos_token = True
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)
        assert slow == [1, 319, 4559, 1243, 2]

        fast_tokenizer = CodeLlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer", add_eos_token=True, add_bos_token=False
        )
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [319, 4559, 1243, 2]

        slow_tokenizer = CodeLlamaTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer", add_eos_token=True, add_bos_token=False
        )
        slow = slow_tokenizer.encode("A sample test", add_special_tokens=True)
        assert slow == [319, 4559, 1243, 2]

        self.tokenizer.add_eos_token = False
        self.rust_tokenizer.add_eos_token = False

    @slow
    def test_conversion(self):
        # This is excruciatingly slow since it has to recreate the entire merge
        # list from the original vocabulary in spm
        self.rust_tokenizer.save_pretrained("./out")
        with tempfile.TemporaryDirectory() as dirname:
            self.rust_tokenizer.save_pretrained(dirname)

            with open(os.path.join(dirname, "tokenizer.json"), "r") as f:
                old_serialized = f.read()

        new_tokenizer = convert_slow_tokenizer(self.tokenizer)
        with tempfile.NamedTemporaryFile() as f:
            new_tokenizer.save(f.name)
            # Re-opening since `f` is in bytes.
            new_serialized = open(f.name, "r").read()
            with open("out_tokenizer.json", "w") as g:
                g.write(new_serialized)

            self.assertEqual(old_serialized, new_serialized)

    def test_simple_encode_decode(self):
        pyth_tokenizer = self.tokenizer
        rust_tokenizer = self.rust_tokenizer

        self.assertEqual(pyth_tokenizer.encode("This is a test"), [1, 910, 338, 263, 1243])
        self.assertEqual(rust_tokenizer.encode("This is a test"), [1, 910, 338, 263, 1243])
        self.assertEqual(pyth_tokenizer.decode([1, 910, 338, 263, 1243], skip_special_tokens=True), "This is a test")
        self.assertEqual(rust_tokenizer.decode([1, 910, 338, 263, 1243], skip_special_tokens=True), "This is a test")

        # bytefallback showcase
        self.assertEqual(pyth_tokenizer.encode("生活的真谛是"), [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392])  # fmt: skip
        self.assertEqual(rust_tokenizer.encode("生活的真谛是"), [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392])  # fmt: skip
        self.assertEqual(
            pyth_tokenizer.decode(
                [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392], skip_special_tokens=True
            ),
            "生活的真谛是",
        )
        self.assertEqual(
            rust_tokenizer.decode(
                [1, 29871, 30486, 31704, 30210, 30848, 235, 179, 158, 30392], skip_special_tokens=True
            ),
            "生活的真谛是",
        )

        # Inner spaces showcase
        self.assertEqual(pyth_tokenizer.encode("Hi  Hello"), [1, 6324, 29871, 15043])
        self.assertEqual(rust_tokenizer.encode("Hi  Hello"), [1, 6324, 29871, 15043])
        self.assertEqual(pyth_tokenizer.decode([1, 6324, 29871, 15043], skip_special_tokens=True), "Hi  Hello")
        self.assertEqual(rust_tokenizer.decode([1, 6324, 29871, 15043], skip_special_tokens=True), "Hi  Hello")

        self.assertEqual(pyth_tokenizer.encode("Hi   Hello"), [1, 6324, 259, 15043])
        self.assertEqual(rust_tokenizer.encode("Hi   Hello"), [1, 6324, 259, 15043])
        self.assertEqual(pyth_tokenizer.decode([1, 6324, 259, 15043], skip_special_tokens=True), "Hi   Hello")
        self.assertEqual(rust_tokenizer.decode([1, 6324, 259, 15043], skip_special_tokens=True), "Hi   Hello")

        self.assertEqual(pyth_tokenizer.encode(""), [1])
        self.assertEqual(rust_tokenizer.encode(""), [1])

        self.assertEqual(pyth_tokenizer.encode(" "), [1, 259])
        self.assertEqual(rust_tokenizer.encode(" "), [1, 259])

        self.assertEqual(pyth_tokenizer.encode("  "), [1, 1678])
        self.assertEqual(rust_tokenizer.encode("  "), [1, 1678])

        self.assertEqual(pyth_tokenizer.encode(" Hello"), [1, 29871, 15043])
        self.assertEqual(rust_tokenizer.encode(" Hello"), [1, 29871, 15043])

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

        dataset = load_dataset("code_x_glue_ct_code_to_text", "go")
        for item in tqdm.tqdm(dataset["validation"]):
            string = item["code"]
            encoded1 = pyth_tokenizer.encode(string)
            encoded2 = rust_tokenizer.encode(string)

            self.assertEqual(encoded1, encoded2)

            decoded1 = pyth_tokenizer.decode(encoded1, skip_special_tokens=True)
            decoded2 = rust_tokenizer.decode(encoded2, skip_special_tokens=True)

            self.assertEqual(decoded1, decoded2)

        dataset = load_dataset("xnli", "all_languages")

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
        tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", legacy=False)
        tokenizer.add_tokens([AddedToken("<REPR_END>", rstrip=True, lstrip=True)], special_tokens=False)
        out1 = tokenizer.decode(
            tokenizer.encode("<REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=False
        )
        self.assertEqual(out1, "<REPR_END>inform")
        out2 = tokenizer.decode(
            tokenizer.encode("<REPR_END>inform", add_special_tokens=False), spaces_between_special_tokens=True
        )
        # the added prefix token should not be decoded
        self.assertEqual(out2, "<REPR_END> inform")
        input_ids = tokenizer.encode("<REPR_END>inform", add_special_tokens=False)
        self.assertEqual(input_ids, [29871, 32016, 262, 689])  # 29871 is the spiece underline, '▁'

        out2 = tokenizer.decode(
            tokenizer.encode(" <REPR_END> inform", add_special_tokens=False), spaces_between_special_tokens=False
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
        self.assertEqual(input_ids, [259, 1, 15043, 1, 920])
        tokens = tokenizer.tokenize(" <s> Hello<s> how", add_special_tokens=False)
        self.assertEqual(tokens, ["▁▁", "<s>", "▁Hello", "<s>", "▁how"])
        decoded_tokens = tokenizer.decode(input_ids)
        self.assertEqual(decoded_tokens, " <s> Hello<s> how")

    def test_fill_token(self):
        tokenizer = CodeLlamaTokenizerFast.from_pretrained(
            "codellama/CodeLlama-7b-hf", fill_token=None, prefix_token=None, suffix_token=None, middle_token=None
        )
        tokenizer.encode_plus("Hey how are you").input_ids
        tokenizer.fill_token = "<FILL_ME>"
        with self.assertRaises(ValueError):
            tokenizer.encode("Hey how <FILL_ME> are you")
            tokenizer.encode_plus("Hey how <FILL_ME> are you", "mne too")
            tokenizer.tokenize("Hey how are you", "mne too")

        tokenizer = CodeLlamaTokenizerFast.from_pretrained(
            "codellama/CodeLlama-7b-hf", revision="3773f63b4511b9e47a9a7ffc765eed7eb0169486"
        )
        tokenizer.encode("Hey how <FILL_ME> are you")
        tokenizer.encode_plus("Hey how <FILL_ME> are you", "mne too")
        tokenizer.tokenize("Hey how are you", "mne too")

    def test_spm_edge_cases(self):
        # the word inform should be split as ['in', 'form']
        tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", legacy=False)
        tokens = tokenizer.tokenize("[INST] How are you doing?<s>[/INST]")
        self.assertEqual(
            tokens, ["▁[", "INST", "]", "▁How", "▁are", "▁you", "▁doing", "?", "<s>", "[", "/", "INST", "]"]
        )
        inputs_ids = tokenizer.encode("[INST] How are you doing?<s>[/INST]")
        self.assertEqual(
            inputs_ids, [1, 518, 25580, 29962, 1128, 526, 366, 2599, 29973, 1, 29961, 29914, 25580, 29962]
        )

    def test_infilling_tokenization(self):
        PROMPTS = [
            '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
''',
            """# Installation instructions:
    ```bash
<FILL_ME>
    ```
This downloads the LLaMA inference code and installs the repository as a local pip package.
""",
            """class InterfaceManagerFactory(AbstractManagerFactory):
    def __init__(<FILL_ME>
def main():
    factory = InterfaceManagerFactory(start=datetime.now())
    managers = []
    for i in range(10):
        managers.append(factory.build(id=i))
""",
            """/-- A quasi-prefunctoid is 1-connected iff all its etalisations are 1-connected. -/
theorem connected_iff_etalisation [C D : precategoroid] (P : quasi_prefunctoid C D) :
π₁ P = 0 ↔ <FILL_ME> = 0 :=
begin
split,
{ intros h f,
    rw pi_1_etalisation at h,
    simp [h],
    refl
},
{ intro h,
    have := @quasi_adjoint C D P,
    simp [←pi_1_etalisation, this, h],
    refl
}
end
""",
        ]
        tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
        tokenizer_fast = CodeLlamaTokenizerFast.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

        formatted_prompt = tokenizer.tokenize(PROMPTS[0])
        self.assertEqual(formatted_prompt, tokenizer_fast.tokenize(PROMPTS[0]))
        prefix, suffix = PROMPTS[0].split("<FILL_ME>")
        self.assertEqual(formatted_prompt, tokenizer.tokenize(prefix, suffix))
        self.assertEqual(formatted_prompt, tokenizer_fast.tokenize(prefix, suffix))

        input_ids = tokenizer.encode(PROMPTS[0], add_special_tokens=False)
        self.assertEqual(input_ids, tokenizer_fast.encode(PROMPTS[0], add_special_tokens=False))

        prefix, suffix = PROMPTS[0].split("<FILL_ME>")
        input_ids = tokenizer.encode(PROMPTS[0])
        self.assertEqual(input_ids, tokenizer.encode(prefix, suffix=suffix))
        self.assertEqual(tokenizer.encode(prefix, suffix=suffix), tokenizer_fast.encode(prefix, suffix=suffix))

        # Adding suffix_first check for infilling tasks
        suffix_first_formatted_prompt = tokenizer.tokenize(PROMPTS[0], suffix_first=True)
        self.assertEqual(suffix_first_formatted_prompt, tokenizer_fast.tokenize(PROMPTS[0], suffix_first=True))
        prefix, suffix = PROMPTS[0].split("<FILL_ME>")
        self.assertEqual(suffix_first_formatted_prompt, tokenizer.tokenize(prefix, suffix, suffix_first=True))
        self.assertEqual(suffix_first_formatted_prompt, tokenizer_fast.tokenize(prefix, suffix, suffix_first=True))

        prefix, suffix = PROMPTS[0].split("<FILL_ME>")
        suffix_first_input_ids = tokenizer.encode(PROMPTS[0], suffix_first=True)
        self.assertEqual(suffix_first_input_ids, tokenizer.encode(prefix, suffix=suffix, suffix_first=True))
        self.assertEqual(suffix_first_input_ids, tokenizer_fast.encode(prefix, suffix=suffix, suffix_first=True))
