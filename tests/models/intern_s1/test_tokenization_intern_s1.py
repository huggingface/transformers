# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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


import copy
import os
from functools import lru_cache

from transformers import AddedToken, InternS1Tokenizer
from transformers.testing_utils import (
    get_tests_dir,
    require_sentencepiece,
    require_tokenizers,
    slow,
)

from ...test_tokenization_common import use_cache_if_possible
from ..qwen2.test_tokenization_qwen2 import Qwen2TokenizationTest


Qwen2TokenizationTest.__test__ = False

SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class InternS1TokenizationTest(Qwen2TokenizationTest):
    __test__ = True
    from_pretrained_id = "internlm/Intern-S1"
    tokenizer_class = InternS1Tokenizer
    rust_tokenizer_class = None
    # InternS1Tokenizer is a slow tokenizer
    test_rust_tokenizer = False
    test_tokenizer_from_extractor = False
    # InternS1Tokenizer encodes ' A' and 'A' differently
    test_pretokenized_inputs = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Create temporary spm.model files to facilitate fast testing without hybrid tokenization functionality
        model_names = [
            "tokenizer_SMILES.model",
            "tokenizer_IUPAC.model",
            "tokenizer_FASTA.model",
        ]
        for model_name in model_names:
            target_path = os.path.join(cls.tmpdirname, model_name)
            if not os.path.exists(target_path):
                os.symlink(SAMPLE_VOCAB, target_path)

        cls.special_tokens_map = {"eos_token": "<|endoftext|>"}

    @classmethod
    @use_cache_if_possible
    @lru_cache(maxsize=64)
    def get_tokenizer(cls, pretrained_name=None, **kwargs):
        _kwargs = copy.deepcopy(cls.special_tokens_map)
        _kwargs.update(kwargs)
        kwargs = _kwargs
        pretrained_name = pretrained_name or cls.tmpdirname
        return InternS1Tokenizer.from_pretrained(pretrained_name, **kwargs)

    def test_add_tokens_tokenizer(self):
        tokenizers = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests (but also otherwise) because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = [
                    AddedToken("aaaaa bbbbbb", rstrip=True, lstrip=True),
                    AddedToken("cccccccccdddddddd", rstrip=True, lstrip=True),
                ]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size_2, vocab_size)
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
                self.assertEqual(vocab_size_3, vocab_size_2)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    ">>>>|||<||<<|<< aaaaa bbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l",
                    add_special_tokens=False,
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])

                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    @slow
    def test_tokenizer_integration(self):
        # Including domain-specific test samples
        sequences = [
            "Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides "
            "general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet...) for Natural "
            "Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained "
            "models in 100+ languages and deep interoperability between Jax, PyTorch and TensorFlow.",
            "🤗 Transformers 提供了可以轻松地下载并且训练先进的预训练模型的 API 和工具。使用预训练模型可以减少计算消耗和碳排放，并且节省从头训练所需要的时间和资源。",
            """```python\ntokenizer = AutoTokenizer.from_pretrained("internlm/Intern-S1")\n"""
            """tokenizer("世界，你好！")```""",
        ]

        # fmt: off
        expected_encoding = {'input_ids': [[8963, 388, 320, 69514, 3881, 438, 4510, 27414, 32852, 388, 323, 4510, 27414, 21334, 35722, 1455, 529, 8, 5707, 4586, 58238, 77235, 320, 61437, 11, 479, 2828, 12, 17, 11, 11830, 61437, 64, 11, 1599, 10994, 11, 27604, 321, 33, 529, 11, 29881, 6954, 32574, 369, 18448, 11434, 45451, 320, 45, 23236, 8, 323, 18448, 11434, 23470, 320, 30042, 38, 8, 448, 916, 220, 18, 17, 10, 80669, 4119, 304, 220, 16, 15, 15, 10, 15459, 323, 5538, 94130, 2897, 1948, 619, 706, 11, 5355, 51, 21584, 323, 94986, 13], [144834, 80532, 93685, 83744, 34187, 73670, 104261, 29490, 62189, 103937, 104034, 102830, 98841, 104034, 104949, 9370, 5333, 58143, 102011, 1773, 37029, 98841, 104034, 104949, 73670, 101940, 100768, 104997, 33108, 100912, 105054, 90395, 100136, 106831, 45181, 64355, 104034, 113521, 101975, 33108, 85329, 1773, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643], [73594, 12669, 198, 85593, 284, 8979, 37434, 6387, 10442, 35722, 445, 55444, 17771, 14, 67916, 6222, 16, 1138, 85593, 445, 99489, 3837, 108386, 6313, 899, 73594, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}
        # fmt: on
        self.tokenizer_integration_test_util(
            expected_encoding=expected_encoding,
            model_name="internlm/Intern-S1",
            revision="edeb08ec5a413c5cf8acad72e8833e506d4090a0",  # TODO
            sequences=sequences,
        )

    @slow
    def test_hybrid_tokenization(self):
        # test encode & decode
        tokenizer = self.tokenizer_class.from_pretrained("internlm/Intern-S1")
        text = """**CC(C)(C)OC(=O)NC@@HC(=O)O, the SMILES representation of N-tert-butyloxycarbonyl-L-phenylalanine, often abbreviated as Boc-Phe-OH.
It's a pretty important molecule in organic chemistry and biochemistry. The `CC(C)(C)` at the beginning represents the tert-butyl group - that's the protecting group.
Then we have `OC(=O)` which is the carbonyl part of the ester. The `N` is the nitrogen atom from the amino group, and `[C@@H]` shows us we have a chiral carbon with
 specific stereochemistry (this is the L-form, which is the natural form found in proteins). The `(CC1=CC=CC=C1)` part is the benzyl side chain - that's what makes this
 phenylalanine instead of some other amino acid. It's basically a benzene ring attached to the main carbon chain. Finally, `C(=O)O` represents the carboxylic acid group
 that's still free and available for reactions."""

        self.assertEqual(text, tokenizer.decode(tokenizer.encode(text)))
        self.assertEqual(
            tokenizer.encode(text),
            tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)),
        )

    @slow
    def test_tokens_trie(self):
        """
        Test the `tokens_trie.split()` method, a component of the `tokenize()` process.

        This method performs an initial segmentation by splitting text based on special
        tokens. This test verifies that domain-specific trigger tokens (e.g., `<SMILES>`)
        are correctly isolated, as this pre-segmentation guides the subsequent tokenization steps.

        Notably, while certain protected tokens (`extra_tokenizer_protect_{begin/end}_tokens`)
        are separated here to act as markers, they are not assigned final token IDs and will be
        further segmented in a later stage.

        # TODO: slow
        """
        tokenizer = self.tokenizer_class.from_pretrained("internlm/Intern-S1")

        self.assertEqual(
            ["Describe ", "<SMILES>", "C1=CC=C(C=C1)C=O", "</SMILES>", " and CC1=CC=CC=C1C=O"],
            tokenizer.tokens_trie.split("Describe <SMILES>C1=CC=C(C=C1)C=O</SMILES> and CC1=CC=CC=C1C=O"),
        )
        self.assertEqual(
            ["<think>", "<MOLFORMULA>", "C6H15O6", "<MOLFORMULA>", "</think>"],
            tokenizer.tokens_trie.split("<think><MOLFORMULA>C6H15O6<MOLFORMULA></think>"),
        )

    @slow
    def test_auto_detection(self):
        """
        Part of the process of `tokenize()`, which processes a list of tokens to automatically detect and wrap specific
        content patterns such as SMILES chemical strings. The core logic ensures that this auto-detection
        does not interfere with content that is already enclosed in special tags.

        Example 1: Basic auto-detection
          Input:  ["Describe C1=CC=C(C=C1)C=O"]
          Output: ["Describe ", "<SMILES_AUTO_DETECT>", "C1=CC=C(C=C1)C=O", "</SMILES_AUTO_DETECT>"]

        Example 2: Avoid processing content in existing tags
          Input:  ["Describe ", "<SMILES>", "C1=CC=C(C=C1)C=O", "</SMILES>", " and CC1=CC=CC=C1C=O"]
          Output: ["Describe ", "<SMILES>", "C1=CC=C(C=C1)C=O", "</SMILES>", " and ",
                   "<SMILES_AUTO_DETECT>", "CC1=CC=CC=C1C=O", "</SMILES_AUTO_DETECT>"]

        Example 3: Protect content in protected tags to avoid unexpected specific tokenization
          Input:  ["<MOLFORMULA>C6H15O6<MOLFORMULA>"]
          Output: ["<MOLFORMULA>", "C6H15O6", "<MOLFORMULA>"]
        """
        tokenizer = self.tokenizer_class.from_pretrained("internlm/Intern-S1")
        # fmt: off
        self.assertEqual(
            ["Describe", "Ġ", "<SMILES_AUTO_DETECT>", "C", "1", "=CC=C(", "C=C", "1", ")C", "=O", "</SMILES_AUTO_DETECT>"],
            tokenizer.tokenize("Describe C1=CC=C(C=C1)C=O"),
        )
        self.assertEqual(["Describe", "Ġ", "<SMILES>", "C", "1", "=CC=C(", "C=C", "1", ")C", "=O", "</SMILES>", "Ġand", "Ġ",
                "<SMILES_AUTO_DETECT>", "CC", "1", "=CC=CC=C", "1", "C", "=O", "</SMILES_AUTO_DETECT>",],
            tokenizer.tokenize("Describe <SMILES>C1=CC=C(C=C1)C=O</SMILES> and CC1=CC=CC=C1C=O"),
        )
        # fmt: on
        self.assertEqual(
            ["<M", "OL", "FORM", "ULA", ">", "C", "6", "H", "1", "5", "O", "6", "<M", "OL", "FORM", "ULA", ">"],
            tokenizer.tokenize("<MOLFORMULA>C6H15O6<MOLFORMULA>"),
        )

    @slow
    def test_convert_tokens_to_ids_offset(self):
        """
        Tests the domain-specific encoding mechanism.

        When a special token is used to switch to a domain-specific context
        (e.g., SMILES representations), a corresponding vocabulary offset must be applied to
        the subsequent token IDs. This test verifies that behavior.
        """
        tokenizer = self.tokenizer_class.from_pretrained("internlm/Intern-S1")
        # fmt: off
        self.assertEqual(
            [74785, 220, 151925, 151854, 151860, 151698, 151707, 151860, 151690, 151726, 151926, 323, 220, 151672, 151860, 151701, 151860, 151854, 151726],
            tokenizer.convert_tokens_to_ids(["Describe", "Ġ", "<SMILES>", "C", "1", "=CC=C(", "C=C","1", ")C", "=O", "</SMILES>",
               "Ġand", "Ġ", "<SMILES_AUTO_DETECT>", "CC", "1", "=CC=CC=C", "1", "C", "=O", "</SMILES_AUTO_DETECT>",]),
        )
        # fmt: on
        self.assertEqual(
            [id - tokenizer.sp_model_SMILES.offset for id in [151672, 151860, 151701, 151860, 151854, 151726]],
            tokenizer.sp_model_SMILES.piece_to_id(["CC", "1", "=CC=CC=C", "1", "C", "=O"]),
        )
