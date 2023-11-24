# coding=utf-8
# Copyright 2022 The HuggingFace Inc. Team. All rights reserved.
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


from __future__ import annotations

import unittest

from transformers import EsmConfig, is_tf_available
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


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
        self.num_hidden_layers = 2
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

        config = EsmConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            pad_token_id=1,
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

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
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
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFEsmForMaskedLM(config=config)
        result = model([input_ids, input_mask])
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_token_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFEsmForTokenClassification(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_tf
class TFEsmModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
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
    pipeline_model_mapping = (
        {
            "feature-extraction": TFEsmModel,
            "fill-mask": TFEsmForMaskedLM,
            "text-classification": TFEsmForSequenceClassification,
            "token-classification": TFEsmForTokenClassification,
            "zero-shot": TFEsmForSequenceClassification,
        }
        if is_tf_available()
        else {}
    )
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFEsmModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EsmConfig, hidden_size=37)

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

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_ESM_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFEsmModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @unittest.skip("Protein models do not support embedding resizing.")
    def test_resize_token_embeddings(self):
        pass

    @unittest.skip("Protein models do not support embedding resizing.")
    def test_save_load_after_resize_token_embeddings(self):
        pass

    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            assert isinstance(model.get_input_embeddings(), tf.keras.layers.Layer)
            if model_class is TFEsmForMaskedLM:
                # Output embedding test differs from the main test because they're a matrix, not a layer
                name = model.get_bias()
                assert isinstance(name, dict)
                for k, v in name.items():
                    assert isinstance(v, tf.Variable)
            else:
                x = model.get_output_embeddings()
                assert x is None
                name = model.get_bias()
                assert name is None


@require_tf
class TFEsmModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = TFEsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")

        input_ids = tf.constant([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]
        expected_shape = [1, 6, 33]
        self.assertEqual(list(output.numpy().shape), expected_shape)
        # compare the actual values for a slice.
        expected_slice = tf.constant(
            [
                [
                    [8.921518, -10.589814, -6.4671307],
                    [-6.3967156, -13.911377, -1.1211915],
                    [-7.781247, -13.951557, -3.740592],
                ]
            ]
        )
        self.assertTrue(numpy.allclose(output[:, :3, :3].numpy(), expected_slice.numpy(), atol=1e-2))

    @slow
    def test_inference_no_head(self):
        model = TFEsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

        input_ids = tf.constant([[0, 6, 4, 13, 5, 4, 16, 12, 11, 7, 2]])
        output = model(input_ids)[0]
        # compare the actual values for a slice.
        expected_slice = tf.constant(
            [
                [
                    [0.14443092, 0.54125327, 0.3247739],
                    [0.30340484, 0.00526676, 0.31077722],
                    [0.32278043, -0.24987096, 0.3414628],
                ]
            ]
        )
        self.assertTrue(numpy.allclose(output[:, :3, :3].numpy(), expected_slice.numpy(), atol=1e-4))
