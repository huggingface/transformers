# coding=utf-8
# Copyright 2018 Google T5 Authors and HuggingFace Inc. team.
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
    from transformers import (T5Config, T5Model, T5WithLMHeadModel)
    from transformers.modeling_t5 import T5_PRETRAINED_MODEL_ARCHIVE_MAP


@require_torch
class T5ModelTest(CommonTestCases.CommonModelTester):

    all_model_classes = (T5Model, T5WithLMHeadModel) if is_torch_available() else ()
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    is_encoder_decoder = True

    class T5ModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     encoder_seq_length=7,
                     decoder_seq_length=9,
                     is_training=True,
                     use_attention_mask=True,
                     use_labels=True,
                     vocab_size=99,
                     n_positions=14,
                     hidden_size=32,
                     num_hidden_layers=5,
                     num_attention_heads=4,
                     d_ff=37,
                     relative_attention_num_buckets=8,
                     dropout_rate=0.1,
                     initializer_factor=0.002,
                     scope=None,
                    ):
            self.parent = parent
            self.batch_size = batch_size
            self.encoder_seq_length = encoder_seq_length
            self.decoder_seq_length = decoder_seq_length
            self.is_training = is_training
            self.use_attention_mask = use_attention_mask
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.n_positions = n_positions
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.d_ff = d_ff
            self.relative_attention_num_buckets = relative_attention_num_buckets
            self.dropout_rate = dropout_rate
            self.initializer_factor = initializer_factor
            self.scope = scope

        def prepare_config_and_inputs(self):
            encoder_input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)
            decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

            encoder_attention_mask = None
            decoder_attention_mask = None
            if self.use_attention_mask:
                encoder_attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)
                decoder_attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)

            decoder_lm_labels = None
            if self.use_labels:
                decoder_lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

            config = T5Config(
                vocab_size=self.vocab_size,
                n_positions=self.n_positions,
                d_model=self.hidden_size,
                d_ff=self.d_ff,
                d_kv=self.hidden_size // self.num_attention_heads,
                num_layers=self.num_hidden_layers,
                num_heads=self.num_attention_heads,
                relative_attention_num_buckets=self.relative_attention_num_buckets,
                dropout_rate=self.dropout_rate,
                initializer_factor=self.initializer_factor)

            return (config, encoder_input_ids, decoder_input_ids, encoder_attention_mask, decoder_attention_mask, decoder_lm_labels)

        def check_loss_output(self, result):
            self.parent.assertListEqual(
                list(result["loss"].size()),
                [])

        def create_and_check_t5_model(self, config, encoder_input_ids, decoder_input_ids, encoder_attention_mask, decoder_attention_mask, decoder_lm_labels):
            model = T5Model(config=config)
            model.eval()
            decoder_output, encoder_output = model(encoder_input_ids=encoder_input_ids,
                                                   decoder_input_ids=decoder_input_ids,
                                                   encoder_attention_mask=encoder_attention_mask,
                                                   decoder_attention_mask=decoder_attention_mask)
            decoder_output, encoder_output = model(encoder_input_ids=encoder_input_ids,
                                                   decoder_input_ids=decoder_input_ids)

            result = {
                "encoder_output": encoder_output,
                "decoder_output": decoder_output,
            }
            self.parent.assertListEqual(
                list(result["encoder_output"].size()),
                [self.batch_size, self.encoder_seq_length, self.hidden_size])
            self.parent.assertListEqual(
                list(result["decoder_output"].size()),
                [self.batch_size, self.decoder_seq_length, self.hidden_size])


        def create_and_check_t5_with_lm_head(self, config, encoder_input_ids, decoder_input_ids, encoder_attention_mask, decoder_attention_mask, decoder_lm_labels):
            model = T5WithLMHeadModel(config=config)
            model.eval()
            outputs = model(encoder_input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask, decoder_lm_labels=decoder_lm_labels)
            loss, prediction_scores = outputs[0], outputs[1]
            result = {
                "loss": loss,
                "prediction_scores": prediction_scores,
            }
            self.parent.assertListEqual(
                list(result["prediction_scores"].size()),
                [self.batch_size, self.decoder_seq_length, self.vocab_size])
            self.check_loss_output(result)

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (config, encoder_input_ids, decoder_input_ids, encoder_attention_mask,
             decoder_attention_mask, decoder_lm_labels) = config_and_inputs
            inputs_dict = {'encoder_input_ids': encoder_input_ids,
                           'decoder_input_ids': decoder_input_ids,
                           'decoder_attention_mask': decoder_attention_mask,
                           'encoder_attention_mask': encoder_attention_mask}
            return config, inputs_dict

    def setUp(self):
        self.model_tester = T5ModelTest.T5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=T5Config, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_t5_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_t5_model(*config_and_inputs)

    def test_with_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_t5_with_lm_head(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        cache_dir = "/tmp/transformers_test/"
        for model_name in list(T5_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = T5Model.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()
