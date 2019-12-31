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


import unittest

from transformers import T5Config, is_tf_available

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor
from .utils import CACHE_DIR, require_tf, slow


if is_tf_available():
    from transformers.modeling_tf_t5 import TFT5Model, TFT5WithLMHeadModel


@require_tf
class TFT5ModelTest(TFModelTesterMixin, unittest.TestCase):

    is_encoder_decoder = True
    all_model_classes = (TFT5Model, TFT5WithLMHeadModel) if is_tf_available() else ()

    class TFT5ModelTester(object):
        def __init__(
            self,
            parent,
            batch_size=13,
            seq_length=7,
            is_training=True,
            use_input_mask=True,
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
            self.seq_length = seq_length
            self.is_training = is_training
            self.use_input_mask = use_input_mask
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
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

            token_labels = None
            if self.use_labels:
                token_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

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
                initializer_factor=self.initializer_factor,
            )

            return (config, input_ids, input_mask, token_labels)

        def create_and_check_t5_model(self, config, input_ids, input_mask, token_labels):
            model = TFT5Model(config=config)
            inputs = {
                "encoder_input_ids": input_ids,
                "decoder_input_ids": input_ids,
                "decoder_attention_mask": input_mask,
            }
            encoder_output, decoder_output = model(inputs)

            encoder_output, decoder_output = model(
                input_ids, decoder_attention_mask=input_mask, encoder_input_ids=input_ids
            )

            result = {
                "encoder_output": encoder_output.numpy(),
                "decoder_output": decoder_output.numpy(),
            }
            self.parent.assertListEqual(
                list(result["encoder_output"].shape), [self.batch_size, self.seq_length, self.hidden_size]
            )
            self.parent.assertListEqual(
                list(result["decoder_output"].shape), [self.batch_size, self.seq_length, self.hidden_size]
            )

        def create_and_check_t5_with_lm_head(self, config, input_ids, input_mask, token_labels):
            model = TFT5WithLMHeadModel(config=config)
            inputs = {
                "encoder_input_ids": input_ids,
                "decoder_input_ids": input_ids,
                "decoder_attention_mask": input_mask,
            }
            prediction_scores, decoder_output = model(inputs)
            result = {
                "prediction_scores": prediction_scores.numpy(),
            }
            self.parent.assertListEqual(
                list(result["prediction_scores"].shape), [self.batch_size, self.seq_length, self.vocab_size]
            )

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (config, input_ids, input_mask, token_labels) = config_and_inputs
            inputs_dict = {
                "encoder_input_ids": input_ids,
                "decoder_input_ids": input_ids,
                "decoder_attention_mask": input_mask,
            }
            return config, inputs_dict

    def setUp(self):
        self.model_tester = TFT5ModelTest.TFT5ModelTester(self)
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
        for model_name in ["t5-small"]:
            model = TFT5Model.from_pretrained(model_name, cache_dir=CACHE_DIR)
            self.assertIsNotNone(model)
