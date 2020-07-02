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

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    from transformers import (
        AlbertConfig,
        AlbertModel,
        AlbertForPreTraining,
        AlbertForMaskedLM,
        AlbertForMultipleChoice,
        AlbertForSequenceClassification,
        AlbertForTokenClassification,
        AlbertForQuestionAnswering,
    )
    from transformers.modeling_albert import ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST


class AlbertModelTester:
    def __init__(
        self, parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_mask = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.vocab_size = 99
        self.embedding_size = 16
        self.hidden_size = 36
        self.num_hidden_layers = 6
        self.num_hidden_groups = 6
        self.num_attention_heads = 6
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

        config = AlbertConfig(
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
            num_hidden_groups=self.num_hidden_groups,
        )

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def check_loss_output(self, result):
        self.parent.assertListEqual(list(result["loss"].size()), [])

    def create_and_check_albert_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = AlbertModel(config=config)
        model.to(torch_device)
        model.eval()
        sequence_output, pooled_output = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        sequence_output, pooled_output = model(input_ids, token_type_ids=token_type_ids)
        sequence_output, pooled_output = model(input_ids)

        result = {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
        }
        self.parent.assertListEqual(
            list(result["sequence_output"].size()), [self.batch_size, self.seq_length, self.hidden_size]
        )
        self.parent.assertListEqual(list(result["pooled_output"].size()), [self.batch_size, self.hidden_size])

    def create_and_check_albert_for_pretraining(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = AlbertForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        loss, prediction_scores, sop_scores = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
            sentence_order_label=sequence_labels,
        )
        result = {
            "loss": loss,
            "prediction_scores": prediction_scores,
            "sop_scores": sop_scores,
        }
        self.parent.assertListEqual(
            list(result["prediction_scores"].size()), [self.batch_size, self.seq_length, self.vocab_size]
        )
        self.parent.assertListEqual(list(result["sop_scores"].size()), [self.batch_size, config.num_labels])
        self.check_loss_output(result)

    def create_and_check_albert_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = AlbertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        loss, prediction_scores = model(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels
        )
        result = {
            "loss": loss,
            "prediction_scores": prediction_scores,
        }
        self.parent.assertListEqual(
            list(result["prediction_scores"].size()), [self.batch_size, self.seq_length, self.vocab_size]
        )
        self.check_loss_output(result)

    def create_and_check_albert_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = AlbertForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        loss, start_logits, end_logits = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        result = {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
        self.parent.assertListEqual(list(result["start_logits"].size()), [self.batch_size, self.seq_length])
        self.parent.assertListEqual(list(result["end_logits"].size()), [self.batch_size, self.seq_length])
        self.check_loss_output(result)

    def create_and_check_albert_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = AlbertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        loss, logits = model(
            input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels
        )
        result = {
            "loss": loss,
            "logits": logits,
        }
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.num_labels])
        self.check_loss_output(result)

    def create_and_check_albert_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = AlbertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        loss, logits = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        result = {
            "loss": loss,
            "logits": logits,
        }
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.seq_length, self.num_labels])
        self.check_loss_output(result)

    def create_and_check_albert_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = AlbertForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        loss, logits = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            labels=choice_labels,
        )
        result = {
            "loss": loss,
            "logits": logits,
        }
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.num_choices])

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


@require_torch
class AlbertModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            AlbertModel,
            AlbertForPreTraining,
            AlbertForMaskedLM,
            AlbertForMultipleChoice,
            AlbertForSequenceClassification,
            AlbertForTokenClassification,
            AlbertForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = AlbertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AlbertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_albert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_albert_model(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_albert_for_pretraining(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_albert_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_albert_for_multiple_choice(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_albert_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_albert_for_sequence_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = AlbertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
