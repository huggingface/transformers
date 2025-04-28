# coding=utf-8
# Copyright 2024 ConvaiInnovations and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch HindiCausalLM model."""

import unittest

from transformers import HindiCausalLMConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        HindiCausalLMForCausalLM,
        HindiCausalLMForSequenceClassification,
        HindiCausalLMModel,
    )


class HindiCausalLMModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
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
        self.num_key_value_heads = num_key_value_heads
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
        self.pad_token_id = pad_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels

    def get_config(self):
        return HindiCausalLMConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs_for_decoder(self):
        config, input_ids, input_mask, sequence_labels = self.prepare_config_and_inputs()

        config.is_decoder = True
        encoder_hidden_states = ids_tensor([self.batch_size, self.seq_length, self.hidden_size], self.vocab_size)
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels):
        model = HindiCausalLMModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = HindiCausalLMModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
    ):
        model = HindiCausalLMForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        input_mask,
        sequence_labels,
    ):
        config.num_labels = self.num_labels
        model = HindiCausalLMForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class HindiCausalLMModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            HindiCausalLMModel,
            HindiCausalLMForCausalLM,
            HindiCausalLMForSequenceClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (HindiCausalLMForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": HindiCausalLMModel,
            "text-classification": HindiCausalLMForSequenceClassification,
            "text-generation": HindiCausalLMForCausalLM,
            "zero-shot": HindiCausalLMForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = HindiCausalLMModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=HindiCausalLMConfig, hidden_size=37, vocab_size=99, num_attention_heads=4
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "convaiinnovations/hindi-causal-lm"
        model = HindiCausalLMModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_torch
class HindiCausalLMModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head(self):
        model = HindiCausalLMModel.from_pretrained("convaiinnovations/hindi-causal-lm")
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        output = model(input_ids)[0]

        expected_shape = torch.Size([1, 6, 768])
        self.assertEqual(output.shape, expected_shape)

    @slow
    def test_inference_lm_head(self):
        model = HindiCausalLMForCausalLM.from_pretrained("convaiinnovations/hindi-causal-lm")

        # Test with Hindi text input
        input_ids = torch.tensor([[1, 100, 200, 300, 2]])  # Example token IDs
        output = model(input_ids)[0]

        expected_shape = torch.Size([1, 5, model.config.vocab_size])
        self.assertEqual(output.shape, expected_shape)
