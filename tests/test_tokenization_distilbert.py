# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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


from transformers.tokenization_distilbert import DistilBertTokenizer

from .test_tokenization_bert import BertTokenizationTest
from .utils import slow


class DistilBertTokenizationTest(BertTokenizationTest):

    tokenizer_class = DistilBertTokenizer

    def get_tokenizer(self, **kwargs):
        return DistilBertTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    @slow
    def test_sequence_builders(self):
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [tokenizer.sep_token_id] + text_2 + [
            tokenizer.sep_token_id
        ]

    def test_encode_plus_with_padding(self):
        tokenizer = self.get_tokenizer()

        sequence = "Sequence"
        padding_size = 10
        padding_idx = tokenizer.pad_token_id

        encoded_sequence = tokenizer.encode_plus(sequence, return_special_tokens_mask=True)
        input_ids = encoded_sequence["input_ids"]
        attention_mask = encoded_sequence["attention_mask"]
        special_tokens_mask = encoded_sequence["special_tokens_mask"]
        sequence_length = len(input_ids)

        # Test right padding
        tokenizer.padding_side = "right"
        padded_sequence = tokenizer.encode_plus(
            sequence,
            max_length=sequence_length + padding_size,
            pad_to_max_length=True,
            return_special_tokens_mask=True,
        )
        padded_input_ids = padded_sequence["input_ids"]
        padded_attention_mask = padded_sequence["attention_mask"]
        padded_special_tokens_mask = padded_sequence["special_tokens_mask"]
        padded_sequence_length = len(padded_input_ids)

        assert sequence_length + padding_size == padded_sequence_length
        assert input_ids + [padding_idx] * padding_size == padded_input_ids
        assert attention_mask + [0] * padding_size == padded_attention_mask
        assert special_tokens_mask + [1] * padding_size == padded_special_tokens_mask

        # Test left padding
        tokenizer.padding_side = "left"
        padded_sequence = tokenizer.encode_plus(
            sequence,
            max_length=sequence_length + padding_size,
            pad_to_max_length=True,
            return_special_tokens_mask=True,
        )
        padded_input_ids = padded_sequence["input_ids"]
        padded_attention_mask = padded_sequence["attention_mask"]
        padded_special_tokens_mask = padded_sequence["special_tokens_mask"]
        padded_sequence_length = len(padded_input_ids)

        assert sequence_length + padding_size == padded_sequence_length
        assert [padding_idx] * padding_size + input_ids == padded_input_ids
        assert [0] * padding_size + attention_mask == padded_attention_mask
        assert [1] * padding_size + special_tokens_mask == padded_special_tokens_mask

    def test_mask_output(self):
        tokenizer = self.get_tokenizer()

        if tokenizer.build_inputs_with_special_tokens.__qualname__.split(".")[0] != "PreTrainedTokenizer":
            seq_0 = "Test this method."
            seq_1 = "With these inputs."
            information = tokenizer.encode_plus(seq_0, seq_1, return_token_type_ids=True, add_special_tokens=True)
            sequences, mask = information["input_ids"], information["token_type_ids"]
            self.assertEqual(len(sequences), len(mask))
