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

import os
import unittest
import json
import random
import shutil
import pytest

from transformers import XLNetConfig, is_tf_available

if is_tf_available():
    import tensorflow as tf

    from transformers.modeling_tf_xlnet import (TFXLNetModel, TFXLNetLMHeadModel,
                                                        TFXLNetForSequenceClassification,
                                                        TFXLNetForQuestionAnsweringSimple,
                                                        TF_XLNET_PRETRAINED_MODEL_ARCHIVE_MAP)
else:
    pytestmark = pytest.mark.skip("Require TensorFlow")

from .modeling_tf_common_test import (TFCommonTestCases, ids_tensor)
from .configuration_common_test import ConfigTester

class TFXLNetModelTest(TFCommonTestCases.TFCommonModelTester):

    all_model_classes=(TFXLNetModel, TFXLNetLMHeadModel,
                       TFXLNetForSequenceClassification,
                       TFXLNetForQuestionAnsweringSimple) if is_tf_available() else ()
    test_pruning = False

    class TFXLNetModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     mem_len=10,
                     clamp_len=-1,
                     reuse_len=15,
                     is_training=True,
                     use_labels=True,
                     vocab_size=99,
                     cutoffs=[10, 50, 80],
                     hidden_size=32,
                     num_attention_heads=4,
                     d_inner=128,
                     num_hidden_layers=5,
                     max_position_embeddings=10,
                     type_sequence_label_size=2,
                     untie_r=True,
                     bi_data=False,
                     same_length=False,
                     initializer_range=0.05,
                     seed=1,
                     type_vocab_size=2,
            ):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.mem_len = mem_len
            # self.key_len = seq_length + mem_len
            self.clamp_len = clamp_len
            self.reuse_len = reuse_len
            self.is_training = is_training
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.cutoffs = cutoffs
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.d_inner = d_inner
            self.num_hidden_layers = num_hidden_layers
            self.max_position_embeddings = max_position_embeddings
            self.bi_data = bi_data
            self.untie_r = untie_r
            self.same_length = same_length
            self.initializer_range = initializer_range
            self.seed = seed
            self.type_vocab_size = type_vocab_size
            self.type_sequence_label_size = type_sequence_label_size

        def prepare_config_and_inputs(self):
            input_ids_1 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            input_ids_2 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            segment_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
            input_mask = ids_tensor([self.batch_size, self.seq_length], 2, dtype=tf.float32)

            input_ids_q = ids_tensor([self.batch_size, self.seq_length + 1], self.vocab_size)
            perm_mask = tf.zeros((self.batch_size, self.seq_length + 1, self.seq_length), dtype=tf.float32)
            perm_mask_last = tf.ones((self.batch_size, self.seq_length + 1, 1), dtype=tf.float32)
            perm_mask = tf.concat([perm_mask, perm_mask_last], axis=-1)
            # perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
            target_mapping = tf.zeros((self.batch_size, 1, self.seq_length), dtype=tf.float32)
            target_mapping_last = tf.ones((self.batch_size, 1, 1), dtype=tf.float32)
            target_mapping = tf.concat([target_mapping, target_mapping_last], axis=-1)
            # target_mapping[:, 0, -1] = 1.0  # predict last token

            sequence_labels = None
            lm_labels = None
            is_impossible_labels = None
            if self.use_labels:
                lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
                sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
                is_impossible_labels = ids_tensor([self.batch_size], 2, dtype=tf.float32)

            config = XLNetConfig(
                vocab_size_or_config_json_file=self.vocab_size,
                d_model=self.hidden_size,
                n_head=self.num_attention_heads,
                d_inner=self.d_inner,
                n_layer=self.num_hidden_layers,
                untie_r=self.untie_r,
                max_position_embeddings=self.max_position_embeddings,
                mem_len=self.mem_len,
                clamp_len=self.clamp_len,
                same_length=self.same_length,
                reuse_len=self.reuse_len,
                bi_data=self.bi_data,
                initializer_range=self.initializer_range,
                num_labels=self.type_sequence_label_size)

            return (config, input_ids_1, input_ids_2, input_ids_q, perm_mask, input_mask,
                    target_mapping, segment_ids, lm_labels, sequence_labels, is_impossible_labels)

        def set_seed(self):
            random.seed(self.seed)
            tf.random.set_seed(self.seed)

        def create_and_check_xlnet_base_model(self, config, input_ids_1, input_ids_2, input_ids_q, perm_mask, input_mask,
                target_mapping, segment_ids, lm_labels, sequence_labels, is_impossible_labels):
            model = TFXLNetModel(config)

            inputs = {'input_ids': input_ids_1,
                      'input_mask': input_mask,
                      'token_type_ids': segment_ids}

            _, _ = model(inputs)

            inputs = [input_ids_1, input_mask]

            outputs, mems_1 = model(inputs)

            result = {
                "mems_1": [mem.numpy() for mem in mems_1],
                "outputs": outputs.numpy(),
            }

            self.parent.assertListEqual(
                list(result["outputs"].shape),
                [self.batch_size, self.seq_length, self.hidden_size])
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers)

        def create_and_check_xlnet_lm_head(self, config, input_ids_1, input_ids_2, input_ids_q, perm_mask, input_mask,
                target_mapping, segment_ids, lm_labels, sequence_labels, is_impossible_labels):
            model = TFXLNetLMHeadModel(config)

            inputs_1 = {'input_ids': input_ids_1,
                      'token_type_ids': segment_ids}

            all_logits_1, mems_1 = model(inputs_1)

            inputs_2 = {'input_ids': input_ids_2,
                        'mems': mems_1,
                        'token_type_ids': segment_ids}

            all_logits_2, mems_2 = model(inputs_2)

            inputs_3 = {'input_ids': input_ids_q,
                        'perm_mask': perm_mask,
                        'target_mapping': target_mapping}

            logits, _ = model(inputs_3)

            result = {
                "mems_1": [mem.numpy() for mem in mems_1],
                "all_logits_1": all_logits_1.numpy(),
                "mems_2": [mem.numpy() for mem in mems_2],
                "all_logits_2": all_logits_2.numpy(),
            }

            self.parent.assertListEqual(
                list(result["all_logits_1"].shape),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers)

            self.parent.assertListEqual(
                list(result["all_logits_2"].shape),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_2"]),
                [[self.mem_len, self.batch_size, self.hidden_size]] * self.num_hidden_layers)

        def create_and_check_xlnet_qa(self, config, input_ids_1, input_ids_2, input_ids_q, perm_mask, input_mask,
                target_mapping, segment_ids, lm_labels, sequence_labels, is_impossible_labels):
            model = TFXLNetForQuestionAnsweringSimple(config)

            inputs = {'input_ids': input_ids_1,
                      'attention_mask': input_mask,
                      'token_type_ids': segment_ids}
            start_logits, end_logits, mems = model(inputs)

            result = {
                "start_logits": start_logits.numpy(),
                "end_logits": end_logits.numpy(),
                "mems": [m.numpy() for m in mems],
            }

            self.parent.assertListEqual(
                list(result["start_logits"].shape),
                [self.batch_size, self.seq_length])
            self.parent.assertListEqual(
                list(result["end_logits"].shape),
                [self.batch_size, self.seq_length])
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers)

        def create_and_check_xlnet_sequence_classif(self, config, input_ids_1, input_ids_2, input_ids_q, perm_mask, input_mask,
                target_mapping, segment_ids, lm_labels, sequence_labels, is_impossible_labels):
            model = TFXLNetForSequenceClassification(config)

            logits, mems_1 = model(input_ids_1)

            result = {
                "mems_1": [mem.numpy() for mem in mems_1],
                "logits": logits.numpy(),
            }

            self.parent.assertListEqual(
                list(result["logits"].shape),
                [self.batch_size, self.type_sequence_label_size])
            self.parent.assertListEqual(
                list(list(mem.shape) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.hidden_size]] * self.num_hidden_layers)

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids_1, input_ids_2, input_ids_q, perm_mask, input_mask,
                target_mapping, segment_ids, lm_labels,
                sequence_labels, is_impossible_labels) = config_and_inputs
            inputs_dict = {'input_ids': input_ids_1}
            return config, inputs_dict


    def setUp(self):
        self.model_tester = TFXLNetModelTest.TFXLNetModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XLNetConfig, d_inner=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_xlnet_base_model(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_base_model(*config_and_inputs)

    def test_xlnet_lm_head(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_lm_head(*config_and_inputs) 

    def test_xlnet_sequence_classif(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_sequence_classif(*config_and_inputs)

    def test_xlnet_qa(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_xlnet_qa(*config_and_inputs)

    @pytest.mark.slow
    def test_model_from_pretrained(self):
        cache_dir = "/tmp/transformers_test/"
        for model_name in list(TF_XLNET_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = TFXLNetModel.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
