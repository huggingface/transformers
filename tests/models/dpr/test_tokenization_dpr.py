# coding=utf-8
# Copyright 2020 Huggingface
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

from transformers import (
    DPRContextEncoderTokenizer,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoderTokenizer,
    DPRQuestionEncoderTokenizerFast,
    DPRReaderOutput,
    DPRReaderTokenizer,
    DPRReaderTokenizerFast,
)
from transformers.testing_utils import require_tokenizers, slow
from transformers.tokenization_utils_base import BatchEncoding

from ..bert.test_tokenization_bert import BertTokenizationTest


@require_tokenizers
class DPRContextEncoderTokenizationTest(BertTokenizationTest):
    tokenizer_class = DPRContextEncoderTokenizer
    rust_tokenizer_class = DPRContextEncoderTokenizerFast
    test_rust_tokenizer = True
    from_pretrained_id = "facebook/dpr-ctx_encoder-single-nq-base"


@require_tokenizers
class DPRQuestionEncoderTokenizationTest(BertTokenizationTest):
    tokenizer_class = DPRQuestionEncoderTokenizer
    rust_tokenizer_class = DPRQuestionEncoderTokenizerFast
    test_rust_tokenizer = True
    from_pretrained_id = "facebook/dpr-ctx_encoder-single-nq-base"


@require_tokenizers
class DPRReaderTokenizationTest(BertTokenizationTest):
    tokenizer_class = DPRReaderTokenizer
    rust_tokenizer_class = DPRReaderTokenizerFast
    test_rust_tokenizer = True
    from_pretrained_id = "facebook/dpr-ctx_encoder-single-nq-base"

    @slow
    def test_decode_best_spans(self):
        tokenizer = self.tokenizer_class.from_pretrained("google-bert/bert-base-uncased")

        text_1 = tokenizer.encode("question sequence", add_special_tokens=False)
        text_2 = tokenizer.encode("title sequence", add_special_tokens=False)
        text_3 = tokenizer.encode("text sequence " * 4, add_special_tokens=False)
        input_ids = [[101] + text_1 + [102] + text_2 + [102] + text_3]
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

    @slow
    def test_call(self):
        tokenizer = self.tokenizer_class.from_pretrained("google-bert/bert-base-uncased")

        text_1 = tokenizer.encode("question sequence", add_special_tokens=False)
        text_2 = tokenizer.encode("title sequence", add_special_tokens=False)
        text_3 = tokenizer.encode("text sequence", add_special_tokens=False)
        expected_input_ids = [101] + text_1 + [102] + text_2 + [102] + text_3
        encoded_input = tokenizer(questions=["question sequence"], titles=["title sequence"], texts=["text sequence"])
        self.assertIn("input_ids", encoded_input)
        self.assertIn("attention_mask", encoded_input)
        self.assertListEqual(encoded_input["input_ids"][0], expected_input_ids)
