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


import unittest

from transformers import DistilBertConfig, is_tf_available
from transformers.testing_utils import require_tf

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor


if is_tf_available():
    import tensorflow as tf
    from transformers.modeling_tf_distilbert import (
        TFDistilBertModel,
        TFDistilBertForMaskedLM,
        TFDistilBertForQuestionAnswering,
        TFDistilBertForSequenceClassification,
        TFDistilBertForTokenClassification,
        TFDistilBertForMultipleChoice,
    )


class TFDistilBertModelTester:
    def __init__(
        self, parent,
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
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

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

        outputs = model(inputs)
        sequence_output = outputs[0]

        inputs = [input_ids, input_mask]

        (sequence_output,) = model(inputs)

        result = {
            "sequence_output": sequence_output.numpy(),
        }
        self.parent.assertListEqual(
            list(result["sequence_output"].shape), [self.batch_size, self.seq_length, self.hidden_size]
        )

    def create_and_check_distilbert_for_masked_lm(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFDistilBertForMaskedLM(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask}
        (prediction_scores,) = model(inputs)
        result = {
            "prediction_scores": prediction_scores.numpy(),
        }
        self.parent.assertListEqual(
            list(result["prediction_scores"].shape), [self.batch_size, self.seq_length, self.vocab_size]
        )

    def create_and_check_distilbert_for_question_answering(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = TFDistilBertForQuestionAnswering(config=config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask}
        start_logits, end_logits = model(inputs)
        result = {
            "start_logits": start_logits.numpy(),
            "end_logits": end_logits.numpy(),
        }
        self.parent.assertListEqual(list(result["start_logits"].shape), [self.batch_size, self.seq_length])
        self.parent.assertListEqual(list(result["end_logits"].shape), [self.batch_size, self.seq_length])

    def create_and_check_distilbert_for_sequence_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFDistilBertForSequenceClassification(config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask}
        (logits,) = model(inputs)
        result = {
            "logits": logits.numpy(),
        }
        self.parent.assertListEqual(list(result["logits"].shape), [self.batch_size, self.num_labels])

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
        (logits,) = model(inputs)
        result = {
            "logits": logits.numpy(),
        }
        self.parent.assertListEqual(list(result["logits"].shape), [self.batch_size, self.num_choices])

    def create_and_check_distilbert_for_token_classification(
        self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = TFDistilBertForTokenClassification(config)
        inputs = {"input_ids": input_ids, "attention_mask": input_mask}
        (logits,) = model(inputs)
        result = {
            "logits": logits.numpy(),
        }
        self.parent.assertListEqual(list(result["logits"].shape), [self.batch_size, self.seq_length, self.num_labels])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask, sequence_labels, token_labels, choice_labels) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_tf
class TFDistilBertModelTest(TFModelTesterMixin, unittest.TestCase):

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
    test_pruning = True
    test_torchscript = True
    test_head_masking = True

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

    # @slow
    # def test_model_from_pretrained(self):
    #     for model_name in list(DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
    #         model = DistilBertModesss.from_pretrained(model_name)
    #         self.assertIsNotNone(model)
