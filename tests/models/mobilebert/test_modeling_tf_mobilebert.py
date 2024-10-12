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

from transformers import MobileBertConfig, is_tf_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_tf, slow

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TF_MODEL_FOR_PRETRAINING_MAPPING,
        TFMobileBertForMaskedLM,
        TFMobileBertForMultipleChoice,
        TFMobileBertForNextSentencePrediction,
        TFMobileBertForPreTraining,
        TFMobileBertForQuestionAnswering,
        TFMobileBertForSequenceClassification,
        TFMobileBertForTokenClassification,
        TFMobileBertModel,
    )


@require_tf
class TFMobileBertModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            TFMobileBertModel,
            TFMobileBertForMaskedLM,
            TFMobileBertForNextSentencePrediction,
            TFMobileBertForPreTraining,
            TFMobileBertForQuestionAnswering,
            TFMobileBertForSequenceClassification,
            TFMobileBertForTokenClassification,
            TFMobileBertForMultipleChoice,
        )
        if is_tf_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": TFMobileBertModel,
            "fill-mask": TFMobileBertForMaskedLM,
            "question-answering": TFMobileBertForQuestionAnswering,
            "text-classification": TFMobileBertForSequenceClassification,
            "token-classification": TFMobileBertForTokenClassification,
            "zero-shot": TFMobileBertForSequenceClassification,
        }
        if is_tf_available()
        else {}
    )
    test_head_masking = False
    test_onnx = False

    # special case for ForPreTraining model, same as BERT tests
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class in get_values(TF_MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict["next_sentence_label"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)

        return inputs_dict

    class TFMobileBertModelTester:
        def __init__(
            self,
            parent,
            batch_size=13,
            seq_length=7,
            is_training=True,
            use_input_mask=True,
            use_token_type_ids=True,
            use_labels=True,
            vocab_size=99,
            hidden_size=32,
            embedding_size=32,
            num_hidden_layers=2,
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
            self.embedding_size = embedding_size

        def prepare_config_and_inputs(self):
            input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            input_mask = None
            if self.use_input_mask:
                input_mask = random_attention_mask([self.batch_size, self.seq_length])

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

            config = MobileBertConfig(
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
                embedding_size=self.embedding_size,
            )

            return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

        def create_and_check_mobilebert_model(
            self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
        ):
            model = TFMobileBertModel(config=config)
            inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
            result = model(inputs)

            inputs = [input_ids, input_mask]
            result = model(inputs)

            result = model(input_ids)

            self.parent.assertEqual(
                result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size)
            )
            self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

        def create_and_check_mobilebert_for_masked_lm(
            self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
        ):
            model = TFMobileBertForMaskedLM(config=config)
            inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
            result = model(inputs)
            self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

        def create_and_check_mobilebert_for_next_sequence_prediction(
            self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
        ):
            model = TFMobileBertForNextSentencePrediction(config=config)
            inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
            result = model(inputs)
            self.parent.assertEqual(result.logits.shape, (self.batch_size, 2))

        def create_and_check_mobilebert_for_pretraining(
            self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
        ):
            model = TFMobileBertForPreTraining(config=config)
            inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
            result = model(inputs)
            self.parent.assertEqual(
                result.prediction_logits.shape, (self.batch_size, self.seq_length, self.vocab_size)
            )
            self.parent.assertEqual(result.seq_relationship_logits.shape, (self.batch_size, 2))

        def create_and_check_mobilebert_for_sequence_classification(
            self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
        ):
            config.num_labels = self.num_labels
            model = TFMobileBertForSequenceClassification(config=config)
            inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
            result = model(inputs)
            self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

        def create_and_check_mobilebert_for_multiple_choice(
            self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
        ):
            config.num_choices = self.num_choices
            model = TFMobileBertForMultipleChoice(config=config)
            multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids, 1), (1, self.num_choices, 1))
            multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1, self.num_choices, 1))
            multiple_choice_token_type_ids = tf.tile(tf.expand_dims(token_type_ids, 1), (1, self.num_choices, 1))
            inputs = {
                "input_ids": multiple_choice_inputs_ids,
                "attention_mask": multiple_choice_input_mask,
                "token_type_ids": multiple_choice_token_type_ids,
            }
            result = model(inputs)
            self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

        def create_and_check_mobilebert_for_token_classification(
            self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
        ):
            config.num_labels = self.num_labels
            model = TFMobileBertForTokenClassification(config=config)
            inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
            result = model(inputs)
            self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

        def create_and_check_mobilebert_for_question_answering(
            self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
        ):
            model = TFMobileBertForQuestionAnswering(config=config)
            inputs = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": token_type_ids}
            result = model(inputs)
            self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
            self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

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
            inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
            return config, inputs_dict

    def setUp(self):
        self.model_tester = TFMobileBertModelTest.TFMobileBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MobileBertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mobilebert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_multiple_choice(*config_and_inputs)

    def test_for_next_sequence_prediction(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_next_sequence_prediction(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_pretraining(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_mobilebert_for_token_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        #     model_name = 'google/mobilebert-uncased'
        for model_name in ["google/mobilebert-uncased"]:
            model = TFMobileBertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_tf
class TFMobileBertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = TFMobileBertForPreTraining.from_pretrained("google/mobilebert-uncased")
        input_ids = tf.constant([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        expected_shape = [1, 6, 30522]
        self.assertEqual(output.shape, expected_shape)

        expected_slice = tf.constant(
            [
                [
                    [-4.5919547, -9.248295, -9.645256],
                    [-6.7306175, -6.440284, -6.6052837],
                    [-7.2743506, -6.7847915, -6.024673],
                ]
            ]
        )
        tf.debugging.assert_near(output[:, :3, :3], expected_slice, atol=1e-4)
