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
import shutil

from transformers import is_torch_available

from .modeling_common_test import (CommonTestCases, ids_tensor, floats_tensor)
from .configuration_common_test import ConfigTester
from .utils import require_torch, slow, torch_device

if is_torch_available():
    from transformers import (UnilmConfig, UnilmForSeq2Seq, UnilmModel)
    from transformers.modeling_unilm import UNILM_PRETRAINED_MODEL_ARCHIVE_MAP


@require_torch
class UnilmModelTest(CommonTestCases.CommonModelTester):

    all_model_classes = ()

    class UnilmModelTest(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     is_training=True,
                     use_input_mask=True,
                     use_token_type_ids=True,
                     use_labels=True,
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
                     type_sequence_label_size=2,
                     initializer_range=0.02,
                     num_labels=3,
                     num_choices=4,
                     scope=None,
                     ):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_input_mask = use_input_mask
            self.use_token_type_ids = use_token_type_ids
            self.use_labels = use_labels
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
            self.type_sequence_label_size = type_sequence_label_size
            self.initializer_range = initializer_range
            self.num_labels = num_labels
            self.num_choices = num_choices
            self.scope = scope

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            token_type_ids = None
            if self.use_token_type_ids:
                token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

            sequence_labels = None
            token_labels = None
            choice_labels = None
            if self.use_labels:
                sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
                token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
                choice_labels = ids_tensor([self.batch_size], self.num_choices)

            config = UnilmConfig(
                vocab_size_or_config_json_file=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                type_vocab_size=self.type_vocab_size,
                is_decoder=False,
                initializer_range=self.initializer_range)

            return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids, token_type_ids, input_mask,
             sequence_labels, token_labels, choice_labels) = config_and_inputs
            inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': input_mask}
            return config, inputs_dict

        def prepare_config_and_inputs_for_seq2seq_finetuning(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            masked_lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            masked_pos = ids_tensor([self.batch_size, self.seq_length], self.seq_length)
            masked_weights = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            input_mask = None
            if self.use_input_mask:
                input_mask = ids_tensor([self.batch_size, self.seq_length, self.seq_length], vocab_size=2)

            token_type_ids = None
            if self.use_token_type_ids:
                token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

            config = UnilmConfig(
                vocab_size_or_config_json_file=self.vocab_size,
                hidden_size=self.hidden_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                max_position_embeddings=self.max_position_embeddings,
                type_vocab_size=self.type_vocab_size,
                is_decoder=False,
                initializer_range=self.initializer_range)

            return config, input_ids, token_type_ids, input_mask, masked_lm_labels, masked_pos, masked_weights

        def create_and_check_unilm_model_for_seq2seq_finetuning(self, config, input_ids, token_type_ids, input_mask, masked_lm_labels, masked_pos, masked_weights):
            model = UnilmForSeq2Seq(config=config)
            model.to(torch_device)
            model.eval()
            loss = model(
                input_ids, token_type_ids=token_type_ids, attention_mask=input_mask, 
                masked_lm_labels=masked_lm_labels, masked_pos=masked_pos, masked_weights=masked_weights)

            self.parent.assertListEqual(list(loss.size()), [])

    def setUp(self):
        self.model_tester = UnilmModelTest.UnilmModelTest(self)
        self.config_tester = ConfigTester(self, config_class=UnilmConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_for_seq2seq_finetuning(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_seq2seq_finetuning()
        self.model_tester.create_and_check_unilm_model_for_seq2seq_finetuning(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        cache_dir = "/tmp/transformers_test/"
        for model_name in list(UNILM_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = UnilmModel.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
