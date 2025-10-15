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
import unittest

from transformers import AutoTokenizer

from transformers.models.electra.tokenization_electra import (
    VOCAB_FILES_NAMES,
    ElectraTokenizer
)
from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin, filter_non_english


@require_tokenizers
class ElectraTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google/electra-small-generator"
    tokenizer_class = ElectraTokenizer
    rust_tokenizer_class = ElectraTokenizer
    test_rust_tokenizer = False
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "google/electra-small-generator"

        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tok_auto]

        # Create custom vocab for testing
        vocab_tokens = [
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[MASK]",
            "want",
            "##want",
            "##ed",
            "wa",
            "un",
            "runn",
            "##ing",
            ",",
            "low",
            "lowest",
        ]
        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(cls.vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        cls.custom_tokenizer = cls.tokenizer_class(vocab_file=cls.vocab_file)
        cls.tokenizers.append(cls.custom_tokenizer)

  
    def test_clean_text(self):
        tokenizer = self.custom_tokenizer

        # Example taken from the issue https://github.com/huggingface/tokenizers/issues/340
        self.assertListEqual([tokenizer.tokenize(t) for t in ["Test", "\xad", "test"]], [["[UNK]"], [], ["[UNK]"]])


    def test_offsets_with_special_characters(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.get_tokenizer(pretrained_name, **kwargs)

                sentence = f"A, naïve {tokenizer_r.mask_token} AllenNLP sentence."
                tokens = tokenizer_r.encode_plus(
                    sentence,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )

                do_lower_case = tokenizer_r.do_lower_case if hasattr(tokenizer_r, "do_lower_case") else False
                expected_results = (
                    [
                        ((0, 0), tokenizer_r.cls_token),
                        ((0, 1), "A"),
                        ((1, 2), ","),
                        ((3, 5), "na"),
                        ((5, 6), "##ï"),
                        ((6, 8), "##ve"),
                        ((9, 15), tokenizer_r.mask_token),
                        ((16, 21), "Allen"),
                        ((21, 23), "##NL"),
                        ((23, 24), "##P"),
                        ((25, 33), "sentence"),
                        ((33, 34), "."),
                        ((0, 0), tokenizer_r.sep_token),
                    ]
                    if not do_lower_case
                    else [
                        ((0, 0), tokenizer_r.cls_token),
                        ((0, 1), "a"),
                        ((1, 2), ","),
                        ((3, 8), "naive"),
                        ((9, 15), tokenizer_r.mask_token),
                        ((16, 21), "allen"),
                        ((21, 23), "##nl"),
                        ((23, 24), "##p"),
                        ((25, 33), "sentence"),
                        ((33, 34), "."),
                        ((0, 0), tokenizer_r.sep_token),
                    ]
                )

                self.assertEqual(
                    [e[1] for e in expected_results], tokenizer_r.convert_ids_to_tokens(tokens["input_ids"])
                )
                self.assertEqual([e[0] for e in expected_results], tokens["offset_mapping"])

    def test_change_tokenize_chinese_chars(self):
        list_of_common_chinese_char = ["的", "人", "有"]
        text_with_chinese_char = "".join(list_of_common_chinese_char)
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                kwargs["tokenize_chinese_chars"] = True
                tokenizer_p = self.get_tokenizer(pretrained_name, **kwargs)

                ids_without_spe_char_p = tokenizer_p.encode(text_with_chinese_char, add_special_tokens=False)

                tokens_without_spe_char_r = tokenizer_r.convert_ids_to_tokens(ids_without_spe_char_r)

                # it is expected that each Chinese character is not preceded by "##"
                self.assertListEqual(tokens_without_spe_char_p, list_of_common_chinese_char)

                kwargs["tokenize_chinese_chars"] = False
                tokenizer_p = self.get_tokenizer(pretrained_name, **kwargs)

                ids_without_spe_char_p = tokenizer_p.encode(text_with_chinese_char, add_special_tokens=False)

                tokens_without_spe_char_p = tokenizer_p.convert_ids_to_tokens(ids_without_spe_char_p)

                # it is expected that only the first Chinese character is not preceded by "##".
                expected_tokens = [
                    f"##{token}" if idx != 0 else token for idx, token in enumerate(list_of_common_chinese_char)
                ]
                self.assertListEqual(tokens_without_spe_char_p, expected_tokens)
