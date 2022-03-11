# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch DecisionTransformer model. """


import inspect
import unittest

from transformers import DecisionTransformerConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_generation_utils import GenerationTesterMixin
from .test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    # import torch

    from transformers import DecisionTransformerModel
    from transformers.models.decision_transformer.modeling_decision_transformer import (
        DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


class DecisionTransformerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        act_dim=6,
        state_dim=17,
        hidden_size=23,
        max_length=11,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        states = floats_tensor((self.batch_size, self.seq_length, self.state_dim))
        actions = floats_tensor((self.batch_size, self.seq_length, self.act_dim))
        rewards = floats_tensor((self.batch_size, self.seq_length, 1))
        returns_to_go = floats_tensor((self.batch_size, self.seq_length, 1))
        timesteps = ids_tensor((self.batch_size, self.seq_length), vocab_size=1000)
        attention_mask = random_attention_mask((self.batch_size, self.seq_length))

        config = self.get_config()

        return (
            config,
            states,
            actions,
            rewards,
            returns_to_go,
            timesteps,
            attention_mask,
        )

    def get_config(self):
        return DecisionTransformerConfig(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            act_dim=self.act_dim,
            state_dim=self.state_dim,
            hidden_size=self.hidden_size,
            max_length=self.max_length,
        )

    def create_and_check_model(
        self,
        config,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask,
    ):
        model = DecisionTransformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(states, actions, rewards, returns_to_go, timesteps, attention_mask)

        self.parent.assertEqual(result.state_preds.shape, states.shape)
        self.parent.assertEqual(result.action_preds.shape, actions.shape)
        self.parent.assertEqual(result.return_preds.shape, returns_to_go.shape)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (
                self.batch_size,
                self.seq_length * 3,
                self.hidden_size,
            ),  # seq length *3 as there are 3 modelities: states, returns and actions
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            states,
            actions,
            rewards,
            returns_to_go,
            timesteps,
            attention_mask,
        ) = config_and_inputs
        inputs_dict = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "returns_to_go": returns_to_go,
            "timesteps": timesteps,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class DecisionTransformerModelTest(  # ModelTesterMixin is removed as this model as the input / outputs of this model do not conform to a typical NLP model.
    ModelTesterMixin, GenerationTesterMixin, unittest.TestCase
):

    all_model_classes = (DecisionTransformerModel,) if is_torch_available() else ()
    all_generative_model_classes = ()

    # Ignoring of a failing test from GenerationTesterMixin, as the model does not use inputs_ids
    test_generate_without_input_ids = False

    # Ignoring of a failing tests from ModelTesterMixin, as the model does not implement these features
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_attention_outputs = False
    test_hidden_states_output = False
    test_inputs_embeds = False
    test_model_common_attributes = False

    def setUp(self):
        self.model_tester = DecisionTransformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DecisionTransformerConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = DecisionTransformerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "states",
                "actions",
                "rewards",
                "returns_to_go",
                "timesteps",
                "attention_mask",
            ]

            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)
