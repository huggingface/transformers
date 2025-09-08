# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import json
import os
import unittest

from transformers import BatchEncoding, LEDTokenizer, LEDTokenizerFast
from transformers.models.led.tokenization_led import VOCAB_FILES_NAMES
from transformers.testing_utils import require_tokenizers, require_torch
from transformers.utils import cached_property

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class TestTokenizationLED(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "allenai/led-base-16384"
    tokenizer_class = LEDTokenizer
    rust_tokenizer_class = LEDTokenizerFast
    test_rust_tokenizer = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        cls.special_tokens_map = {"unk_token": "<unk>"}

        cls.vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        cls.merges_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(cls.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(cls.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    @classmethod
    def get_tokenizer(cls, pretrained_name=None, **kwargs):
        kwargs.update(cls.special_tokens_map)
        pretrained_name = pretrained_name or cls.tmpdirname
        return cls.tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    @classmethod
    def get_rust_tokenizer(cls, pretrained_name=None, **kwargs):
        kwargs.update(cls.special_tokens_map)
        pretrained_name = pretrained_name or cls.tmpdirname
        return cls.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

    def get_input_output_texts(self, tokenizer):
        return "lower newer", "lower newer"

    @cached_property
    def default_tokenizer(self):
        return LEDTokenizer.from_pretrained("allenai/led-base-16384")

    @cached_property
    def default_tokenizer_fast(self):
        return LEDTokenizerFast.from_pretrained("allenai/led-base-16384")

    @require_torch
    def test_prepare_batch(self):
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        expected_src_tokens = [0, 250, 251, 17818, 13, 39186, 1938, 4, 2]

        for tokenizer in [self.default_tokenizer, self.default_tokenizer_fast]:
            batch = tokenizer(src_text, max_length=len(expected_src_tokens), padding=True, return_tensors="pt")
            self.assertIsInstance(batch, BatchEncoding)

            self.assertEqual((2, 9), batch.input_ids.shape)
            self.assertEqual((2, 9), batch.attention_mask.shape)
            result = batch.input_ids.tolist()[0]
            self.assertListEqual(expected_src_tokens, result)

    @require_torch
    def test_prepare_batch_empty_target_text(self):
        src_text = ["A long paragraph for summarization.", "Another paragraph for summarization."]
        for tokenizer in [self.default_tokenizer, self.default_tokenizer_fast]:
            batch = tokenizer(src_text, padding=True, return_tensors="pt")
            self.assertIn("input_ids", batch)
            self.assertIn("attention_mask", batch)
            self.assertNotIn("labels", batch)
            self.assertNotIn("decoder_attention_mask", batch)

    @require_torch
    def test_tokenizer_as_target_length(self):
        tgt_text = [
            "Summary of the text.",
            "Another summary.",
        ]
        for tokenizer in [self.default_tokenizer, self.default_tokenizer_fast]:
            targets = tokenizer(text_target=tgt_text, max_length=32, padding="max_length", return_tensors="pt")
            self.assertEqual(32, targets["input_ids"].shape[1])

    @require_torch
    def test_prepare_batch_not_longer_than_maxlen(self):
        for tokenizer in [self.default_tokenizer, self.default_tokenizer_fast]:
            batch = tokenizer(
                ["I am a small frog" * 1024, "I am a small frog"], padding=True, truncation=True, return_tensors="pt"
            )
            self.assertIsInstance(batch, BatchEncoding)
            self.assertEqual(batch.input_ids.shape, (2, 5122))

    @require_torch
    def test_special_tokens(self):
        src_text = ["A long paragraph for summarization."]
        tgt_text = [
            "Summary of the text.",
        ]
        for tokenizer in [self.default_tokenizer, self.default_tokenizer_fast]:
            inputs = tokenizer(src_text, return_tensors="pt")
            targets = tokenizer(text_target=tgt_text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            labels = targets["input_ids"]
            self.assertTrue((input_ids[:, 0] == tokenizer.bos_token_id).all().item())
            self.assertTrue((labels[:, 0] == tokenizer.bos_token_id).all().item())
            self.assertTrue((input_ids[:, -1] == tokenizer.eos_token_id).all().item())
            self.assertTrue((labels[:, -1] == tokenizer.eos_token_id).all().item())

    @require_torch
    def test_global_attention_mask(self):
        for tokenizer in [self.default_tokenizer, self.default_tokenizer_fast]:
            src_text = ["Summary of the text.", "Another summary."]
            expected_global_attention_mask = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, -1]]

            encoded_output = tokenizer(src_text, padding=False)
            encoded_output["global_attention_mask"] = [[0] * len(x) for x in encoded_output["input_ids"]]
            outputs = tokenizer.pad(encoded_output)
            self.assertSequenceEqual(outputs["global_attention_mask"], expected_global_attention_mask)

    @unittest.skip
    def test_pretokenized_inputs(self):
        pass

    def test_embeded_special_tokens(self):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.get_rust_tokenizer(pretrained_name, **kwargs)
                tokenizer_p = self.get_tokenizer(pretrained_name, **kwargs)
                sentence = "A, <mask> AllenNLP sentence."
                tokens_r = tokenizer_r.encode_plus(sentence, add_special_tokens=True, return_token_type_ids=True)
                tokens_p = tokenizer_p.encode_plus(sentence, add_special_tokens=True, return_token_type_ids=True)
                self.assertEqual(sum(tokens_r["token_type_ids"]), sum(tokens_p["token_type_ids"]))
                self.assertEqual(
                    sum(tokens_r["attention_mask"]) / len(tokens_r["attention_mask"]),
                    sum(tokens_p["attention_mask"]) / len(tokens_p["attention_mask"]),
                )

                tokens_r_str = tokenizer_r.convert_ids_to_tokens(tokens_r["input_ids"])
                tokens_p_str = tokenizer_p.convert_ids_to_tokens(tokens_p["input_ids"])
                self.assertSequenceEqual(tokens_p["input_ids"], [0, 250, 6, 50264, 3823, 487, 21992, 3645, 4, 2])
                self.assertSequenceEqual(tokens_r["input_ids"], [0, 250, 6, 50264, 3823, 487, 21992, 3645, 4, 2])

                self.assertSequenceEqual(
                    tokens_p_str, ["<s>", "A", ",", "<mask>", "ĠAllen", "N", "LP", "Ġsentence", ".", "</s>"]
                )
                self.assertSequenceEqual(
                    tokens_r_str, ["<s>", "A", ",", "<mask>", "ĠAllen", "N", "LP", "Ġsentence", ".", "</s>"]
                )
