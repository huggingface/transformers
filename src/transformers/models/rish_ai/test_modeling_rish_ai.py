# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""Testing RishAI model."""

import unittest

import torch

from transformers import RishAIConfig, is_torch_available
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from .modeling_rish_ai import (
    RishAICausalLM,
    RishAIModel,
)


if is_torch_available():
    import torch


class RishAIModelTester:
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
        num_experts=7,
        num_experts_per_tok=5,
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
        self.scope = scope
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # RishAI specific
        self.pad_token_id = vocab_size - 1
        self.bos_token_id = vocab_size - 2
        self.eos_token_id = vocab_size - 3

    def prepare_config_and_inputs(self):
        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_length), device=torch_device
        )

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.ones(self.batch_size, self.seq_length, device=torch_device)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = torch.randint(
                0, self.type_vocab_size, (self.batch_size, self.seq_length), device=torch_device
            )

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = torch.randint(0, self.num_labels, (self.batch_size,), device=torch_device)
            token_labels = torch.randint(
                0, self.num_labels, (self.batch_size, self.seq_length), device=torch_device
            )
            choice_labels = torch.randint(0, self.num_choices, (self.batch_size,), device=torch_device)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return RishAIConfig(
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
            is_encoder_decoder=False,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = RishAIModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_causal_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = RishAICausalLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        result = model(input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

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
class RishAIModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = RishAIModelTester(self)

    def test_config(self):
        config = self.model_tester.get_config()
        self.assertIsNotNone(config)
        self.assertEqual(config.model_type, "rish_ai")

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_causal_lm(*config_and_inputs)

    def test_model_from_pretrained(self):
        config = self.model_tester.get_config()
        model = RishAIModel(config)
        self.assertIsNotNone(model)

    def test_causal_lm_from_pretrained(self):
        config = self.model_tester.get_config()
        model = RishAICausalLM(config)
        self.assertIsNotNone(model)

    def test_forward_signature(self):
        config = self.model_tester.get_config()
        model = RishAIModel(config)
        model.to(torch_device)

        input_ids = torch.randint(0, config.vocab_size, (2, 10), device=torch_device)
        result = model(input_ids)
        self.assertIsNotNone(result.last_hidden_state)

    def test_causal_lm_forward_signature(self):
        config = self.model_tester.get_config()
        model = RishAICausalLM(config)
        model.to(torch_device)

        input_ids = torch.randint(0, config.vocab_size, (2, 10), device=torch_device)
        result = model(input_ids)
        self.assertIsNotNone(result.logits)

    def test_past_key_values(self):
        config = self.model_tester.get_config()
        model = RishAIModel(config)
        model.to(torch_device)

        input_ids = torch.randint(0, config.vocab_size, (2, 10), device=torch_device)
        result = model(input_ids, use_cache=True)
        self.assertIsNotNone(result.past_key_values)

    def test_attention_mask(self):
        config = self.model_tester.get_config()
        model = RishAIModel(config)
        model.to(torch_device)

        input_ids = torch.randint(0, config.vocab_size, (2, 10), device=torch_device)
        attention_mask = torch.ones(2, 10, device=torch_device)
        result = model(input_ids, attention_mask=attention_mask)
        self.assertIsNotNone(result.last_hidden_state)

    def test_generation(self):
        config = self.model_tester.get_config()
        model = RishAICausalLM(config)
        model.to(torch_device)

        input_ids = torch.randint(0, config.vocab_size, (1, 5), device=torch_device)
        generated = model.generate(input_ids, max_length=10)
        self.assertIsNotNone(generated)
        self.assertEqual(generated.shape[0], 1)
        self.assertGreaterEqual(generated.shape[1], 5)
