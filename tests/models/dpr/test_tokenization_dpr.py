# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
    DPRContextEncoderTokenizer,
    DPRQuestionEncoderTokenizer,
    DPRReaderOutput,
    DPRReaderTokenizer,
)
from transformers.testing_utils import require_tokenizers
from transformers.tokenization_utils_base import BatchEncoding

from ...test_tokenization_common import TokenizerTesterMixin


READER_CHECKPOINT = "facebook/dpr-reader-single-nq-base"


class DPREncoderTokenizationTesterMixin(TokenizerTesterMixin):
    """
    Shared expectations for the two DPR encoder tokenizers. Both are plain `BertTokenizer` subclasses over the
    `bert-base-uncased` vocabulary, so their tokenization is identical and the constants below are shared.
    """

    integration_expected_tokens = ['this', 'is', 'a', 'test', '[UNK]', 'i', 'was', 'born', 'in', '92', '##00', '##0', ',', 'and', 'this', 'is', 'false', '.', '生', '[UNK]', '的', '真', '[UNK]', '[UNK]', 'hi', 'hello', 'hi', 'hello', 'hello', '<', 's', '>', 'hi', '<', 's', '>', 'there', 'the', 'following', 'string', 'should', 'be', 'properly', 'encoded', ':', 'hello', '.', 'but', 'ir', '##d', 'and', '[UNK]', 'ir', '##d', '[UNK]', 'hey', 'how', 'are', 'you', 'doing']  # fmt: skip
    integration_expected_token_ids = [2023, 2003, 1037, 3231, 100, 1045, 2001, 2141, 1999, 6227, 8889, 2692, 1010, 1998, 2023, 2003, 6270, 1012, 1910, 100, 1916, 1921, 100, 100, 7632, 7592, 7632, 7592, 7592, 1026, 1055, 1028, 7632, 1026, 1055, 1028, 2045, 1996, 2206, 5164, 2323, 2022, 7919, 12359, 1024, 7592, 1012, 2021, 20868, 2094, 1998, 100, 20868, 2094, 100, 4931, 2129, 2024, 2017, 2725]  # fmt: skip
    integration_expected_decoded_text = "this is a test [UNK] i was born in 92000, and this is false. 生 [UNK] 的 真 [UNK] [UNK] hi hello hi hello hello < s > hi < s > there the following string should be properly encoded : hello. but ird and [UNK] ird [UNK] hey how are you doing"


@require_tokenizers
class DPRContextEncoderTokenizationTest(DPREncoderTokenizationTesterMixin, unittest.TestCase):
    from_pretrained_id = "facebook/dpr-ctx_encoder-single-nq-base"
    tokenizer_class = DPRContextEncoderTokenizer


@require_tokenizers
class DPRQuestionEncoderTokenizationTest(DPREncoderTokenizationTesterMixin, unittest.TestCase):
    from_pretrained_id = "facebook/dpr-question_encoder-single-nq-base"
    tokenizer_class = DPRQuestionEncoderTokenizer


@require_tokenizers
class DPRReaderTokenizationTest(unittest.TestCase):
    """
    `DPRReaderTokenizer` replaces `__call__` with a three-input (questions, titles, texts) signature and adds
    `decode_best_spans`, so it is covered here rather than through `TokenizerTesterMixin`, whose tests all assume
    the standard `(text, text_pair)` call.
    """

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = DPRReaderTokenizer.from_pretrained(READER_CHECKPOINT)

    def test_call_concatenates_question_title_and_text(self):
        tokenizer = self.tokenizer
        text_1 = tokenizer.encode("question sequence", add_special_tokens=False)
        text_2 = tokenizer.encode("title sequence", add_special_tokens=False)
        text_3 = tokenizer.encode("text sequence", add_special_tokens=False)
        expected_input_ids = (
            [tokenizer.cls_token_id] + text_1 + [tokenizer.sep_token_id] + text_2 + [tokenizer.sep_token_id] + text_3
        )

        encoded_input = tokenizer(questions=["question sequence"], titles=["title sequence"], texts=["text sequence"])

        self.assertIn("input_ids", encoded_input)
        self.assertIn("attention_mask", encoded_input)
        self.assertListEqual(encoded_input["input_ids"][0], expected_input_ids)
        self.assertListEqual(encoded_input["attention_mask"][0], [1] * len(expected_input_ids))

    def test_call_broadcasts_a_single_question_over_passages(self):
        # One question asked against several passages is the normal retrieval-then-read shape.
        encoded_input = self.tokenizer(
            questions="What is love ?",
            titles=["Haddaway", "Love"],
            texts=["'What Is Love' is a song by Haddaway", "Love is a feeling"],
        )
        self.assertEqual(len(encoded_input["input_ids"]), 2)

    def test_call_without_titles_and_texts_falls_back_to_plain_tokenization(self):
        # With neither titles nor texts, the reader tokenizer must behave like its BertTokenizer superclass.
        encoded_input = self.tokenizer(questions=["question sequence"])
        self.assertEqual(
            encoded_input["input_ids"][0], self.tokenizer.encode("question sequence", add_special_tokens=True)
        )

    def test_call_rejects_mismatched_titles_and_texts(self):
        with self.assertRaises(ValueError):
            self.tokenizer(questions="q", titles=["one title"], texts=["first text", "second text"])

    def test_decode_best_spans(self):
        tokenizer = self.tokenizer
        text_1 = tokenizer.encode("question sequence", add_special_tokens=False)
        text_2 = tokenizer.encode("title sequence", add_special_tokens=False)
        text_3 = tokenizer.encode("text sequence " * 4, add_special_tokens=False)
        input_ids = [
            [tokenizer.cls_token_id] + text_1 + [tokenizer.sep_token_id] + text_2 + [tokenizer.sep_token_id] + text_3
        ]
        reader_input = BatchEncoding({"input_ids": input_ids})

        start_logits = [[0] * len(input_ids[0])]
        end_logits = [[0] * len(input_ids[0])]
        relevance_logits = [0]
        reader_output = DPRReaderOutput(start_logits, end_logits, relevance_logits)

        start_index, end_index = 8, 9
        start_logits[0][start_index] = 10
        end_logits[0][end_index] = 10

        predicted_spans = tokenizer.decode_best_spans(reader_input, reader_output)

        self.assertEqual(predicted_spans[0].start_index, start_index)
        self.assertEqual(predicted_spans[0].end_index, end_index)
        self.assertEqual(predicted_spans[0].doc_id, 0)
        # The span must decode back to the slice of the passage it points at.
        self.assertEqual(predicted_spans[0].text, tokenizer.decode(input_ids[0][start_index : end_index + 1]))
