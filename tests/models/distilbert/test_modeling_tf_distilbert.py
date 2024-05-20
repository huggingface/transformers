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


from __future__ import annotations

import unittest

from transformers import DistilBertConfig, is_tf_available
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers.models.distilbert.modeling_tf_distilbert import (
        TFDistilBertForMaskedLM,
        TFDistilBertForMultipleChoice,
        TFDistilBertForQuestionAnswering,
        TFDistilBertForSequenceClassification,
        TFDistilBertForTokenClassification,
        TFDistilBertModel,
    )


class TFDistilBertModelTester:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_mask = True
        self.use_token_type_ids = False
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

        config = DistilBertConfig(
            vocab_size=self.vocab_size,
            dim=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            hidden_dim=self.intermediate_size,
            hidden_act=self.hidden_act,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def create_and_check_distilbert_model(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFDistilBertModel(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask}

        result = model(inputs)

        inputs = [input_ids, input_mask]

        result = model(inputs)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_distilbert_for_masked_lm(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFDistilBertForMaskedLM(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_distilbert_for_question_answering(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFDistilBertForQuestionAnswering(config=config)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }
        result = model(inputs)
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_distilbert_for_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFDistilBertForSequenceClassification(config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_distilbert_for_multiple_choice(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = TFDistilBertForMultipleChoice(config)
        multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids, 1), (1, self.num_choices, 1))
        multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1, self.num_choices, 1))
        inputs = {
            "input_ids": multiple_choice_inputs_ids,
            "attention_mask": multiple_choice_input_mask,
        }
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def create_and_check_distilbert_for_token_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFDistilBertForTokenClassification(config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask}
        result = model(inputs)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_tf
class TFDistilBertModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            TFDistilBertModel,
            TFDistilBertForMaskedLM,
            TFDistilBertForQuestionAnswering,
            TFDistilBertForSequenceClassification,
            TFDistilBertForTokenClassification,
            TFDistilBertForMultipleChoice,
        )
        if is_tf_available()
        else None
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": TFDistilBertModel,
            "fill-mask": TFDistilBertForMaskedLM,
            "question-answering": TFDistilBertForQuestionAnswering,
            "text-classification": TFDistilBertForSequenceClassification,
            "token-classification": TFDistilBertForTokenClassification,
            "zero-shot": TFDistilBertForSequenceClassification,
        }
        if is_tf_available()
        else {}
    )
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFDistilBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DistilBertConfig, dim=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_distilbert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_masked_lm(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_sequence_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_multiple_choice(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_distilbert_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "distilbert/distilbert-base-cased"
        model = TFDistilBertModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_tf
class TFDistilBertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
        input_ids = tf.constant([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        expected_shape = [1, 6, 768]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = tf.constant(
            [
                [
                    [0.19261885, -0.13732955, 0.4119799],
                    [0.22150156, -0.07422661, 0.39037204],
                    [0.22756018, -0.0896414, 0.3701467],
                ]
            ]
        )
        tf.debugging.assert_near(output[:, :3, :3], expected_slice, atol=1e-4)
