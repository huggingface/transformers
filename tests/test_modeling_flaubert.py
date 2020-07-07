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
        FlaubertConfig,
        FlaubertModel,
        FlaubertWithLMHeadModel,
        FlaubertForQuestionAnswering,
        FlaubertForQuestionAnsweringSimple,
        FlaubertForSequenceClassification,
    )
    from transformers.modeling_flaubert import FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST


class FlaubertModelTester(object):
    def __init__(
        self, parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_lengths = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.gelu_activation = True
        self.sinusoidal_embeddings = False
        self.causal = False
        self.asm = False
        self.n_langs = 2
        self.vocab_size = 99
        self.n_special = 0
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 12
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.num_labels = 3
        self.num_choices = 4
        self.summary_type = "last"
        self.use_proj = None
        self.scope = None

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = ids_tensor([self.batch_size, self.seq_length], 2).float()

        input_lengths = None
        if self.use_input_lengths:
            input_lengths = (
                ids_tensor([self.batch_size], vocab_size=2) + self.seq_length - 2
            )  # small variation of seq_length

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.n_langs)

        sequence_labels = None
        token_labels = None
        is_impossible_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            is_impossible_labels = ids_tensor([self.batch_size], 2).float()

        config = FlaubertConfig(
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
            use_proj=self.use_proj,
        )

        return (
            config,
            input_ids,
            token_type_ids,
            input_lengths,
            sequence_labels,
            token_labels,
            is_impossible_labels,
            input_mask,
        )

    def check_loss_output(self, result):
        self.parent.assertListEqual(list(result["loss"].size()), [])

    def create_and_check_flaubert_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        input_mask,
    ):
        model = FlaubertModel(config=config)
        model.to(torch_device)
        model.eval()
        outputs = model(input_ids, lengths=input_lengths, langs=token_type_ids)
        outputs = model(input_ids, langs=token_type_ids)
        outputs = model(input_ids)
        sequence_output = outputs[0]
        result = {
            "sequence_output": sequence_output,
        }
        self.parent.assertListEqual(
            list(result["sequence_output"].size()), [self.batch_size, self.seq_length, self.hidden_size]
        )

    def create_and_check_flaubert_lm_head(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        input_mask,
    ):
        model = FlaubertWithLMHeadModel(config)
        model.to(torch_device)
        model.eval()

        loss, logits = model(input_ids, token_type_ids=token_type_ids, labels=token_labels)

        result = {
            "loss": loss,
            "logits": logits,
        }

        self.parent.assertListEqual(list(result["loss"].size()), [])
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.seq_length, self.vocab_size])

    def create_and_check_flaubert_simple_qa(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        input_mask,
    ):
        model = FlaubertForQuestionAnsweringSimple(config)
        model.to(torch_device)
        model.eval()

        outputs = model(input_ids)

        outputs = model(input_ids, start_positions=sequence_labels, end_positions=sequence_labels)
        loss, start_logits, end_logits = outputs

        result = {
            "loss": loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }
        self.parent.assertListEqual(list(result["start_logits"].size()), [self.batch_size, self.seq_length])
        self.parent.assertListEqual(list(result["end_logits"].size()), [self.batch_size, self.seq_length])
        self.check_loss_output(result)

    def create_and_check_flaubert_qa(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        input_mask,
    ):
        model = FlaubertForQuestionAnswering(config)
        model.to(torch_device)
        model.eval()

        outputs = model(input_ids)
        start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = outputs

        outputs = model(
            input_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            cls_index=sequence_labels,
            is_impossible=is_impossible_labels,
            p_mask=input_mask,
        )

        outputs = model(
            input_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
            cls_index=sequence_labels,
            is_impossible=is_impossible_labels,
        )

        (total_loss,) = outputs

        outputs = model(input_ids, start_positions=sequence_labels, end_positions=sequence_labels)

        (total_loss,) = outputs

        result = {
            "loss": total_loss,
            "start_top_log_probs": start_top_log_probs,
            "start_top_index": start_top_index,
            "end_top_log_probs": end_top_log_probs,
            "end_top_index": end_top_index,
            "cls_logits": cls_logits,
        }

        self.parent.assertListEqual(list(result["loss"].size()), [])
        self.parent.assertListEqual(
            list(result["start_top_log_probs"].size()), [self.batch_size, model.config.start_n_top]
        )
        self.parent.assertListEqual(
            list(result["start_top_index"].size()), [self.batch_size, model.config.start_n_top]
        )
        self.parent.assertListEqual(
            list(result["end_top_log_probs"].size()),
            [self.batch_size, model.config.start_n_top * model.config.end_n_top],
        )
        self.parent.assertListEqual(
            list(result["end_top_index"].size()), [self.batch_size, model.config.start_n_top * model.config.end_n_top],
        )
        self.parent.assertListEqual(list(result["cls_logits"].size()), [self.batch_size])

    def create_and_check_flaubert_sequence_classif(
        self,
        config,
        input_ids,
        token_type_ids,
        input_lengths,
        sequence_labels,
        token_labels,
        is_impossible_labels,
        input_mask,
    ):
        model = FlaubertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()

        (logits,) = model(input_ids)
        loss, logits = model(input_ids, labels=sequence_labels)

        result = {
            "loss": loss,
            "logits": logits,
        }

        self.parent.assertListEqual(list(result["loss"].size()), [])
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.type_sequence_label_size])

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_lengths,
            sequence_labels,
            token_labels,
            is_impossible_labels,
            input_mask,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "lengths": input_lengths}
        return config, inputs_dict


@require_torch
class FlaubertModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            FlaubertModel,
            FlaubertWithLMHeadModel,
            FlaubertForQuestionAnswering,
            FlaubertForQuestionAnsweringSimple,
            FlaubertForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = FlaubertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FlaubertConfig, emb_dim=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_flaubert_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_flaubert_model(*config_and_inputs)

    def test_flaubert_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_flaubert_lm_head(*config_and_inputs)

    def test_flaubert_simple_qa(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_flaubert_simple_qa(*config_and_inputs)

    def test_flaubert_qa(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_flaubert_qa(*config_and_inputs)

    def test_flaubert_sequence_classif(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_flaubert_sequence_classif(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = FlaubertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
