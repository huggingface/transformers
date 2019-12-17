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

from transformers import is_tf_available

if is_tf_available():
    import tensorflow as tf
    from transformers import (XLMConfig, TFXLMModel,
                                      TFXLMWithLMHeadModel,
                                      TFXLMForSequenceClassification,
                                      TFXLMForQuestionAnsweringSimple,
                                      TF_XLM_PRETRAINED_MODEL_ARCHIVE_MAP)

from .modeling_tf_common_test import (TFCommonTestCases, ids_tensor)
from .configuration_common_test import ConfigTester
from .utils import require_tf, slow


@require_tf
class TFXLMModelTest(TFCommonTestCases.TFCommonModelTester):

    all_model_classes = (TFXLMModel, TFXLMWithLMHeadModel,
                         TFXLMForSequenceClassification,
                         TFXLMForQuestionAnsweringSimple) if is_tf_available() else ()


    class TFXLMModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     is_training=True,
                     use_input_lengths=True,
                     use_token_type_ids=True,
                     use_labels=True,
                     gelu_activation=True,
                     sinusoidal_embeddings=False,
                     causal=False,
                     asm=False,
                     n_langs=2,
                     vocab_size=99,
                     n_special=0,
                     hidden_size=32,
                     num_hidden_layers=5,
                     num_attention_heads=4,
                     hidden_dropout_prob=0.1,
                     attention_probs_dropout_prob=0.1,
                     max_position_embeddings=512,
                     type_vocab_size=16,
                     type_sequence_label_size=2,
                     initializer_range=0.02,
                     num_labels=3,
                     num_choices=4,
                     summary_type="last",
                     use_proj=True,
                     scope=None,
                    ):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_input_lengths = use_input_lengths
            self.use_token_type_ids = use_token_type_ids
            self.use_labels = use_labels
            self.gelu_activation = gelu_activation
            self.sinusoidal_embeddings = sinusoidal_embeddings
            self.asm = asm
            self.n_langs = n_langs
            self.vocab_size = vocab_size
            self.n_special = n_special
            self.summary_type = summary_type
            self.causal = causal
            self.use_proj = use_proj
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.n_langs = n_langs
            self.type_sequence_label_size = type_sequence_label_size
            self.initializer_range = initializer_range
            self.summary_type = summary_type
            self.num_labels = num_labels
            self.num_choices = num_choices
            self.scope = scope

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            input_mask = ids_tensor([self.batch_size, self.seq_length], 2, dtype=tf.float32)

            input_lengths = None
            if self.use_input_lengths:
                input_lengths = ids_tensor([self.batch_size], vocab_size=2) + self.seq_length - 2  # small variation of seq_length

            token_type_ids = None
            if self.use_token_type_ids:
                token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.n_langs)

            sequence_labels = None
            token_labels = None
            is_impossible_labels = None
            if self.use_labels:
                sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
                token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
                is_impossible_labels = ids_tensor([self.batch_size], 2, dtype=tf.float32)

            config = XLMConfig(
                 vocab_size=self.vocab_size,
                 n_special=self.n_special,
                 emb_dim=self.hidden_size,
                 n_layers=self.num_hidden_layers,
                 n_heads=self.num_attention_heads,
                 dropout=self.hidden_dropout_prob,
                 attention_dropout=self.attention_probs_dropout_prob,
                 gelu_activation=self.gelu_activation,
                 sinusoidal_embeddings=self.sinusoidal_embeddings,
                 asm=self.asm,
                 causal=self.causal,
                 n_langs=self.n_langs,
                 max_position_embeddings=self.max_position_embeddings,
                 initializer_range=self.initializer_range,
                 summary_type=self.summary_type,
                 use_proj=self.use_proj)

            return config, input_ids, token_type_ids, input_lengths, sequence_labels, token_labels, is_impossible_labels, input_mask

        def create_and_check_xlm_model(self, config, input_ids, token_type_ids, input_lengths, sequence_labels, token_labels, is_impossible_labels, input_mask):
            model = TFXLMModel(config=config)
            inputs = {'input_ids': input_ids,
                      'lengths': input_lengths,
                      'langs': token_type_ids}
            outputs = model(inputs)

            inputs = [input_ids, input_mask]
            outputs = model(inputs)
            sequence_output = outputs[0]
            result = {
                "sequence_output": sequence_output.numpy(),
            }
            self.parent.assertListEqual(
                list(result["sequence_output"].shape),
                [self.batch_size, self.seq_length, self.hidden_size])


        def create_and_check_xlm_lm_head(self, config, input_ids, token_type_ids, input_lengths, sequence_labels, token_labels, is_impossible_labels, input_mask):
            model = TFXLMWithLMHeadModel(config)

            inputs = {'input_ids': input_ids,
                      'lengths': input_lengths,
                      'langs': token_type_ids}
            outputs = model(inputs)

            logits = outputs[0]

            result = {
                "logits": logits.numpy(),
            }

            self.parent.assertListEqual(
                list(result["logits"].shape),
                [self.batch_size, self.seq_length, self.vocab_size])


        def create_and_check_xlm_qa(self, config, input_ids, token_type_ids, input_lengths, sequence_labels, token_labels, is_impossible_labels, input_mask):
            model = TFXLMForQuestionAnsweringSimple(config)

            inputs = {'input_ids': input_ids,
                      'lengths': input_lengths}

            outputs = model(inputs)
            start_logits, end_logits = model(inputs)

            result = {
                "start_logits": start_logits.numpy(),
                "end_logits": end_logits.numpy(),
            }

            self.parent.assertListEqual(
                list(result["start_logits"].shape),
                [self.batch_size, self.seq_length])
            self.parent.assertListEqual(
                list(result["end_logits"].shape),
                [self.batch_size, self.seq_length])


        def create_and_check_xlm_sequence_classif(self, config, input_ids, token_type_ids, input_lengths, sequence_labels, token_labels, is_impossible_labels, input_mask):
            model = TFXLMForSequenceClassification(config)

            inputs = {'input_ids': input_ids,
                      'lengths': input_lengths}

            (logits,) = model(inputs)

            result = {
                "logits": logits.numpy(),
            }

            self.parent.assertListEqual(
                list(result["logits"].shape),
                [self.batch_size, self.type_sequence_label_size])


        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids, token_type_ids, input_lengths,
             sequence_labels, token_labels, is_impossible_labels, input_mask) = config_and_inputs
            inputs_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'langs': token_type_ids, 'lengths': input_lengths}
            return config, inputs_dict

    def setUp(self):
        self.model_tester = TFXLMModelTest.TFXLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XLMConfig, emb_dim=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_xlm_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_model(*config_and_inputs)

    def test_xlm_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_lm_head(*config_and_inputs)

    def test_xlm_qa(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_qa(*config_and_inputs)

    def test_xlm_sequence_classif(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlm_sequence_classif(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        cache_dir = "/tmp/transformers_test/"
        for model_name in list(TF_XLM_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = XLMModel.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
