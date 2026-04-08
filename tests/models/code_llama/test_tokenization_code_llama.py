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
import shutil
import tempfile
import unittest

from tokenizers import AddedToken

from transformers import CodeLlamaTokenizer
from transformers.testing_utils import (
    get_tests_dir,
    nested_simplify,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
)

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


# impoprt convert_slow_tokenizer


@require_sentencepiece
@require_tokenizers
class CodeLlamaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    # TokenizerTesterMixin configuration
    from_pretrained_id = ["hf-internal-testing/llama-code-tokenizer"]
    tokenizer_class = CodeLlamaTokenizer

    integration_expected_tokens = ['▁This', '▁is', '▁a', '▁test', '▁', '<0xF0>', '<0x9F>', '<0x98>', '<0x8A>', '<0x0A>', 'I', '▁was', '▁born', '▁in', '▁', '9', '2', '0', '0', '0', ',', '▁and', '▁this', '▁is', '▁f', 'als', 'é', '.', '<0x0A>', '生', '活', '的', '真', '<0xE8>', '<0xB0>', '<0x9B>', '是', '<0x0A>', 'Hi', '▁', '▁Hello', '<0x0A>', 'Hi', '▁▁', '▁Hello', '<0x0A>', '<0x0A>', '▁', '<0x0A>', '▁▁', '<0x0A>', '▁Hello', '<0x0A>', '<s>', '<0x0A>', 'hi', '<s>', 'there', '<0x0A>', 'The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁Hello', '.', '<0x0A>', 'But', '▁', 'ird', '▁and', '▁', 'ป', 'ี', '▁▁▁', 'ird', '▁▁▁', 'ด', '<0x0A>', 'H', 'ey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [910, 338, 263, 1243, 29871, 243, 162, 155, 141, 13, 29902, 471, 6345, 297, 29871, 29929, 29906, 29900, 29900, 29900, 29892, 322, 445, 338, 285, 1338, 29948, 29889, 13, 30486, 31704, 30210, 30848, 235, 179, 158, 30392, 13, 18567, 29871, 15043, 13, 18567, 259, 15043, 13, 13, 29871, 13, 259, 13, 15043, 13, 1, 13, 2918, 1, 12711, 13, 1576, 1494, 1347, 881, 367, 6284, 18511, 29901, 15043, 29889, 13, 6246, 29871, 1823, 322, 29871, 31010, 30691, 1678, 1823, 1678, 30718, 13, 29950, 1032, 920, 526, 366, 2599]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁a', '▁test', '▁', '<0xF0>', '<0x9F>', '<0x98>', '<0x8A>', '<0x0A>', 'I', '▁was', '▁born', '▁in', '▁', '9', '2', '0', '0', '0', ',', '▁and', '▁this', '▁is', '▁f', 'als', 'é', '.', '<0x0A>', '生', '活', '的', '真', '<0xE8>', '<0xB0>', '<0x9B>', '是', '<0x0A>', 'Hi', '▁', '▁Hello', '<0x0A>', 'Hi', '▁▁', '▁Hello', '<0x0A>', '<0x0A>', '▁', '<0x0A>', '▁▁', '<0x0A>', '▁Hello', '<0x0A>', '<s>', '<0x0A>', 'hi', '<s>', 'there', '<0x0A>', 'The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encoded', ':', '▁Hello', '.', '<0x0A>', 'But', '▁', 'ird', '▁and', '▁', 'ป', 'ี', '▁▁▁', 'ird', '▁▁▁', 'ด', '<0x0A>', 'H', 'ey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"

    def test_save_and_load_tokenizer(self):
        """Override to handle non-deterministic vocabulary order from Rust tokenizer."""
        # safety check on max_len default value so we are sure the test works
        tokenizer = self.get_tokenizer()
        self.assertNotEqual(tokenizer.model_max_length, 42)

        # Now let's start the test
        tokenizer = self.get_tokenizer()
        # Isolate this from the other tests because we save additional tokens/etc
        tmpdirname = tempfile.mkdtemp()

        sample_text = " He is very happy, UNwant\u00e9d,running"
        before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
        before_vocab = tokenizer.get_vocab()
        tokenizer.save_pretrained(tmpdirname)

        after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
        after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
        after_vocab = after_tokenizer.get_vocab()
        self.assertListEqual(before_tokens, after_tokens)

        # Compare vocabularies in an order-independent way
        # The Rust tokenizer returns vocabularies in non-deterministic order
        # Some special tokens may be added during _post_init when loading, so we check that
        # all tokens from before_vocab are in after_vocab with the same IDs
        for token, token_id in before_vocab.items():
            self.assertIn(token, after_vocab, f"Token '{token}' missing in after_vocab")
            self.assertEqual(
                after_vocab[token], token_id, f"Token '{token}' has different ID: {after_vocab[token]} != {token_id}"
            )

        shutil.rmtree(tmpdirname)

        tokenizer = self.get_tokenizer(model_max_length=42)
        # Isolate this from the other tests because we save additional tokens/etc
        tmpdirname = tempfile.mkdtemp()

        sample_text = " He is very happy, UNwant\u00e9d,running"
        tokenizer.add_tokens(["bim", "bambam"])
        extra_special_tokens = tokenizer.extra_special_tokens
        extra_special_tokens.append("new_extra_special_token")
        tokenizer.add_special_tokens(
            {"extra_special_tokens": extra_special_tokens}, replace_extra_special_tokens=False
        )
        before_tokens = tokenizer.encode(sample_text, add_special_tokens=False)
        before_vocab = tokenizer.get_vocab()
        tokenizer.save_pretrained(tmpdirname)

        after_tokenizer = tokenizer.__class__.from_pretrained(tmpdirname)
        after_tokens = after_tokenizer.encode(sample_text, add_special_tokens=False)
        after_vocab = after_tokenizer.get_vocab()
        self.assertListEqual(before_tokens, after_tokens)

        for token, token_id in before_vocab.items():
            self.assertIn(token, after_vocab, f"Token '{token}' missing in after_vocab")
            self.assertEqual(
                after_vocab[token], token_id, f"Token '{token}' has different ID: {after_vocab[token]} != {token_id}"
            )

        self.assertIn("bim", after_vocab)
        self.assertIn("bambam", after_vocab)
        self.assertIn("new_extra_special_token", after_tokenizer.extra_special_tokens)

    def test_no_infilling_init(self):
        tokenizer = CodeLlamaTokenizer.from_pretrained(SAMPLE_VOCAB, prefix_token=None, keep_accents=True)
        with self.assertRaises(ValueError):
            tokenizer.tokenize("This is <FILL_ME> prefix")

    @require_torch
    def test_batch_tokenization(self):
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
                        return_tensors="pt",
                    )
                except NotImplementedError:
                    self.skipTest(reason="Encountered NotImplementedError when calling tokenizer")
                self.assertEqual(batch.input_ids.shape[1], 3)
                # max_target_length will default to max_length if not specified
                batch = tokenizer(text, max_length=3, return_tensors="pt")
                self.assertEqual(batch.input_ids.shape[1], 3)

                batch_encoder_only = tokenizer(text=text, max_length=3, return_tensors="pt")
                self.assertEqual(batch_encoder_only.input_ids.shape[1], 3)
                self.assertEqual(batch_encoder_only.attention_mask.shape[1], 3)
                self.assertNotIn("decoder_input_ids", batch_encoder_only)

    def test_special_tokens_initialization(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                added_tokens = [AddedToken("<special>", lstrip=True)]

                tokenizer_r = self.get_tokenizer(pretrained_name, additional_special_tokens=added_tokens, **kwargs)
                r_output = tokenizer_r.encode("Hey this is a <special> token")

                special_token_id = tokenizer_r.encode("<special>", add_special_tokens=False)[0]

                self.assertTrue(special_token_id in r_output)


@require_tokenizers
class LlamaIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint_name = "hf-internal-testing/llama-code-tokenizer"
        cls.tokenizer: CodeLlamaTokenizer = CodeLlamaTokenizer.from_pretrained(checkpoint_name)
        cls.rust_tokenizer = CodeLlamaTokenizer.from_pretrained(checkpoint_name)
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

        fast_tokenizer = CodeLlamaTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer", add_eos_token=True, add_bos_token=False
        )
        fast = fast_tokenizer.encode("A sample test", add_special_tokens=True)
        assert fast == [319, 4559, 1243, 2]
        self.tokenizer.add_eos_token = False
        self.rust_tokenizer.add_eos_token = False

    @unittest.skip(
        "Skipped in v5 - CodeLlama tokenization differences related to SPM legacy flag and Metaspace handling. "
        "CodeLlama always uses legacy=False (Metaspace pre_tokenizer, no normalizer)"
    )
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

    @unittest.skip(
        "Skipped in v5 - CodeLlama tokenization differences related to SPM legacy flag and Metaspace handling. "
        "CodeLlama always uses legacy=False (Metaspace pre_tokenizer, no normalizer)"
    )
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

        self.assertEqual(pyth_tokenizer.decode([869]), ".")

        self.assertEqual(pyth_tokenizer.decode([30112, 869]), "ا .")

    def test_no_differences_special_tokens(self):
        pyth_tokenizer = self.tokenizer
        self.assertEqual(pyth_tokenizer.encode(""), [1])

        self.assertEqual(pyth_tokenizer.encode("<s>"), [1, 1])

    @unittest.skipIf(
        os.getenv("RUN_TOKENIZER_INTEGRATION", "0") == "0",
        "RUN_TOKENIZER_INTEGRATION=1 to run tokenizer integration tests",
    )
    def test_integration_test_xnli(self):
        import tqdm
        from datasets import load_dataset

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

    def test_fill_token(self):
        tokenizer = CodeLlamaTokenizer.from_pretrained(
            "codellama/CodeLlama-7b-hf", fill_token=None, prefix_token=None, suffix_token=None, middle_token=None
        )
        tokenizer.encode("Hey how are you")
        tokenizer.fill_token = "<FILL_ME>"
        with self.assertRaises(ValueError):
            tokenizer.encode("Hey how <FILL_ME> are you")
            tokenizer.encode("Hey how <FILL_ME> are you", "mne too")
            tokenizer.tokenize("Hey how are you", "mne too")

        tokenizer = CodeLlamaTokenizer.from_pretrained(
            "codellama/CodeLlama-7b-hf", revision="3773f63b4511b9e47a9a7ffc765eed7eb0169486"
        )
        tokenizer.encode("Hey how <FILL_ME> are you")
        tokenizer.encode("Hey how <FILL_ME> are you", "mne too")
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

        formatted_prompt = tokenizer.tokenize(PROMPTS[0])
        prefix, suffix = PROMPTS[0].split("<FILL_ME>")
        self.assertEqual(formatted_prompt, tokenizer.tokenize(prefix, suffix))

        input_ids = tokenizer.encode(PROMPTS[0], add_special_tokens=False)

        prefix, suffix = PROMPTS[0].split("<FILL_ME>")
        input_ids = tokenizer.encode(PROMPTS[0])
        self.assertEqual(input_ids, tokenizer.encode(prefix, suffix=suffix))

        # Adding suffix_first check for infilling tasks
        suffix_first_formatted_prompt = tokenizer.tokenize(PROMPTS[0], suffix_first=True)
        prefix, suffix = PROMPTS[0].split("<FILL_ME>")
        self.assertEqual(suffix_first_formatted_prompt, tokenizer.tokenize(prefix, suffix, suffix_first=True))

        prefix, suffix = PROMPTS[0].split("<FILL_ME>")
        suffix_first_input_ids = tokenizer.encode(PROMPTS[0], suffix_first=True)
        self.assertEqual(suffix_first_input_ids, tokenizer.encode(prefix, suffix=suffix, suffix_first=True))
