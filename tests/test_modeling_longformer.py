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
    import torch
    from transformers import (
        LongformerConfig,
        LongformerModel,
        LongformerForMaskedLM,
        LongformerForSequenceClassification,
        LongformerForTokenClassification,
        LongformerForQuestionAnswering,
        LongformerForMultipleChoice,
    )


class LongformerModelTester:
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
        self.attention_window = 4

        # `ModelTesterMixin.test_attention_outputs` is expecting attention tensors to be of size
        # [num_attention_heads, encoder_seq_length, encoder_key_length], but LongformerSelfAttention
        # returns attention of shape [num_attention_heads, encoder_seq_length, self.attention_window + 1]
        # because its local attention only attends to `self.attention_window + 1` locations
        self.key_length = self.attention_window + 1

        # because of padding `encoder_seq_length`, is different from `seq_length`. Relevant for
        # the `test_attention_outputs` and `test_hidden_states_output` tests
        self.encoder_seq_length = (
            self.seq_length + (self.attention_window - self.seq_length % self.attention_window) % self.attention_window
        )

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

        config = LongformerConfig(
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
            attention_window=self.attention_window,
        )

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def check_loss_output(self, result):
        self.parent.assertListEqual(list(result["loss"].size()), [])

    def create_and_check_attention_mask_determinism(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LongformerModel(config=config)
        model.to(torch_device)
        model.eval()

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        output_with_mask = model(input_ids, attention_mask=attention_mask)[0]
        output_without_mask = model(input_ids)[0]
        self.parent.assertTrue(torch.allclose(output_with_mask[0, 0, :5], output_without_mask[0, 0, :5], atol=1e-4))

    def create_and_check_longformer_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LongformerModel(config=config)
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

    def create_and_check_longformer_model_with_global_attention_mask(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LongformerModel(config=config)
        model.to(torch_device)
        model.eval()
        global_attention_mask = input_mask.clone()
        global_attention_mask[:, input_mask.shape[-1] // 2] = 0
        global_attention_mask = global_attention_mask.to(torch_device)

        sequence_output, pooled_output = model(
            input_ids,
            attention_mask=input_mask,
            global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output, pooled_output = model(
            input_ids, token_type_ids=token_type_ids, global_attention_mask=global_attention_mask
        )
        sequence_output, pooled_output = model(input_ids, global_attention_mask=global_attention_mask)

        result = {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
        }
        self.parent.assertListEqual(
            list(result["sequence_output"].size()), [self.batch_size, self.seq_length, self.hidden_size]
        )
        self.parent.assertListEqual(list(result["pooled_output"].size()), [self.batch_size, self.hidden_size])

    def create_and_check_longformer_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LongformerForMaskedLM(config=config)
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

    def create_and_check_longformer_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LongformerForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        loss, start_logits, end_logits = model(
            input_ids,
            attention_mask=input_mask,
            global_attention_mask=input_mask,
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

    def create_and_check_longformer_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = LongformerForSequenceClassification(config)
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

    def create_and_check_longformer_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = LongformerForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        loss, logits = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        result = {
            "loss": loss,
            "logits": logits,
        }
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.seq_length, self.num_labels])
        self.check_loss_output(result)

    def create_and_check_longformer_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = LongformerForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        loss, logits = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            global_attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            labels=choice_labels,
        )
        result = {
            "loss": loss,
            "logits": logits,
        }
        self.parent.assertListEqual(list(result["logits"].size()), [self.batch_size, self.num_choices])
        self.check_loss_output(result)

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
        global_attention_mask = torch.zeros_like(input_ids)
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
            "global_attention_mask": global_attention_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_question_answering(self):
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

        # Replace sep_token_id by some random id
        input_ids[input_ids == config.sep_token_id] = torch.randint(0, config.vocab_size, (1,)).item()
        # Make sure there are exactly three sep_token_id
        input_ids[:, -3:] = config.sep_token_id
        input_mask = torch.ones_like(input_ids)

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels


@require_torch
class LongformerModelTest(ModelTesterMixin, unittest.TestCase):
    test_pruning = False  # pruning is not supported
    test_headmasking = False  # head masking is not supported
    test_torchscript = False

    all_model_classes = (
        (
            LongformerModel,
            LongformerForMaskedLM,
            LongformerForSequenceClassification,
            LongformerForQuestionAnswering,
            LongformerForTokenClassification,
            LongformerForMultipleChoice,
        )
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = LongformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LongformerConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_longformer_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_longformer_model(*config_and_inputs)

    def test_longformer_model_attention_mask_determinism(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_attention_mask_determinism(*config_and_inputs)

    def test_longformer_model_global_attention_mask(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_longformer_model_with_global_attention_mask(*config_and_inputs)

    def test_longformer_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_longformer_for_masked_lm(*config_and_inputs)

    def test_longformer_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_question_answering()
        self.model_tester.create_and_check_longformer_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_longformer_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_longformer_for_token_classification(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_longformer_for_multiple_choice(*config_and_inputs)


class LongformerModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        model.to(torch_device)

        # 'Hello world!'
        input_ids = torch.tensor([[0, 20920, 232, 328, 1437, 2]], dtype=torch.long, device=torch_device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        output = model(input_ids, attention_mask=attention_mask)[0]
        output_without_mask = model(input_ids)[0]

        expected_output_slice = torch.tensor([0.0549, 0.1087, -0.1119, -0.0368, 0.0250], device=torch_device)
        self.assertTrue(torch.allclose(output[0, 0, -5:], expected_output_slice, atol=1e-4))
        self.assertTrue(torch.allclose(output_without_mask[0, 0, -5:], expected_output_slice, atol=1e-4))

    @slow
    def test_inference_no_head_long(self):
        model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        model.to(torch_device)

        # 'Hello world! ' repeated 1000 times
        input_ids = torch.tensor(
            [[0] + [20920, 232, 328, 1437] * 1000 + [2]], dtype=torch.long, device=torch_device
        )  # long input

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
        global_attention_mask[:, [1, 4, 21]] = 1  # Set global attention on a few random positions

        output = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)[0]

        expected_output_sum = torch.tensor(74585.8594, device=torch_device)
        expected_output_mean = torch.tensor(0.0243, device=torch_device)
        self.assertTrue(torch.allclose(output.sum(), expected_output_sum, atol=1e-4))
        self.assertTrue(torch.allclose(output.mean(), expected_output_mean, atol=1e-4))

    @slow
    def test_inference_masked_lm_long(self):
        model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
        model.to(torch_device)

        # 'Hello world! ' repeated 1000 times
        input_ids = torch.tensor(
            [[0] + [20920, 232, 328, 1437] * 1000 + [2]], dtype=torch.long, device=torch_device
        )  # long input

        loss, prediction_scores = model(input_ids, labels=input_ids)

        expected_loss = torch.tensor(0.0074, device=torch_device)
        expected_prediction_scores_sum = torch.tensor(-6.1048e08, device=torch_device)
        expected_prediction_scores_mean = torch.tensor(-3.0348, device=torch_device)
        input_ids = input_ids.to(torch_device)

        self.assertTrue(torch.allclose(loss, expected_loss, atol=1e-4))
        self.assertTrue(torch.allclose(prediction_scores.sum(), expected_prediction_scores_sum, atol=1e-4))
        self.assertTrue(torch.allclose(prediction_scores.mean(), expected_prediction_scores_mean, atol=1e-4))
