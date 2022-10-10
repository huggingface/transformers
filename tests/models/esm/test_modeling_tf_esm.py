# coding=utf-8
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


import unittest

from transformers import RobertaConfig, is_tf_available
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_tf_available():
    import numpy
    import tensorflow as tf

    from transformers.models.esm.modeling_tf_esm import (
        TF_ESM_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFEsmForMaskedLM,
        TFEsmForSequenceClassification,
        TFEsmForTokenClassification,
        TFEsmModel,
    )


# copied from tests.test_modeling_tf_roberta
class TFEsmModelTester:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_mask = True
        self.use_labels = True
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 16
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.num_labels = 3
        self.num_choices = 4
        self.scope = None

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = RobertaConfig(
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
            initializer_range=self.initializer_range,
        )

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_model(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFEsmModel(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask}
        result = model(inputs)

        inputs = [input_ids, input_mask]
        result = model(inputs)

        result = model(input_ids)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True

        model = TFEsmModel(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
        result = model(inputs)

        inputs = [input_ids, input_mask]
        result = model(inputs, encoder_hidden_states=encoder_hidden_states)

        # Also check the case where encoder outputs are not passed
        result = model(input_ids, attention_mask=input_mask)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFEsmForMaskedLM(config=config)
        result = model([input_ids, input_mask, token_type_ids])
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFEsmForTokenClassification(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_tf
class TFEsmModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            TFEsmModel,
            TFEsmForMaskedLM,
            TFEsmForSequenceClassification,
            TFEsmForTokenClassification,
        )
        if is_tf_available()
        else ()
    )
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFEsmModelTester(self)
        self.config_tester = ConfigTester(self, config_class=RobertaConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        """Test the base model"""
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_as_decoder(self):
        """Test the base model as a decoder (of an encoder-decoder architecture)

        is_deocder=True + cross_attention + pass encoder outputs
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        """Similar to `test_causal_lm_model_past_with_large_inputs` but with cross-attention"""
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)


    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_ESM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFEsmModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_tf
class TFRobertaModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = TFEsmForMaskedLM.from_pretrained("roberta-base")

        input_ids = tf.constant([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        expected_shape = [1, 11, 50265]
        self.assertEqual(list(output.numpy().shape), expected_shape)
        # compare the actual values for a slice.
        expected_slice = tf.constant(
            [[[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]]
        )
        self.assertTrue(numpy.allclose(output[:, :3, :3].numpy(), expected_slice.numpy(), atol=1e-4))

    @slow
    def test_inference_no_head(self):
        model = TFEsmModel.from_pretrained("roberta-base")

        input_ids = tf.constant([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        # compare the actual values for a slice.
        expected_slice = tf.constant(
            [[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]]
        )
        self.assertTrue(numpy.allclose(output[:, :3, :3].numpy(), expected_slice.numpy(), atol=1e-4))