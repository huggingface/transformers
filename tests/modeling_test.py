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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import json
import random

import torch

import modeling


class BertModelTest(unittest.TestCase):
    class BertModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     is_training=True,
                     use_input_mask=True,
                     use_token_type_ids=True,
                     vocab_size=99,
                     hidden_size=32,
                     num_hidden_layers=5,
                     num_attention_heads=4,
                     intermediate_size=37,
                     hidden_act="gelu",
                     hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,
                     max_position_embeddings=512,
                     type_vocab_size=16,
                     initializer_range=0.02,
                     scope=None):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_input_mask = use_input_mask
            self.use_token_type_ids = use_token_type_ids
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.intermediate_size = intermediate_size
            self.hidden_act = hidden_act
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.scope = scope

        def create_model(self):
            input_ids = BertModelTest.ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = BertModelTest.ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            token_type_ids = None
            if self.use_token_type_ids:
                token_type_ids = BertModelTest.ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

            config = modeling.BertConfig(
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                type_vocab_size=self.type_vocab_size,
                initializer_range=self.initializer_range)

            model = modeling.BertModel(config=config)

            all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)

            outputs = {
                "sequence_output": all_encoder_layers[-1],
                "pooled_output": pooled_output,
                "all_encoder_layers": all_encoder_layers,
            }
            return outputs

        def check_output(self, result):
            self.parent.assertListEqual(
                list(result["sequence_output"].size()),
                [self.batch_size, self.seq_length, self.hidden_size])

            self.parent.assertListEqual(list(result["pooled_output"].size()), [self.batch_size, self.hidden_size])

    def test_default(self):
        self.run_tester(BertModelTest.BertModelTester(self))

    def test_config_to_json_string(self):
        config = modeling.BertConfig(vocab_size=99, hidden_size=37)
        obj = json.loads(config.to_json_string())
        self.assertEqual(obj["vocab_size"], 99)
        self.assertEqual(obj["hidden_size"], 37)

    def run_tester(self, tester):
        output_result = tester.create_model()
        tester.check_output(output_result)

    @classmethod
    def ids_tensor(cls, shape, vocab_size, rng=None, name=None):
        """Creates a random int32 tensor of the shape within the vocab size."""
        if rng is None:
            rng = random.Random()

        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(rng.randint(0, vocab_size - 1))

        return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


if __name__ == "__main__":
    unittest.main()
