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
""" Testing suite for the PyTorch TrajectoryTransformer model. """


import inspect
import unittest

import numpy as np

from transformers import TrajectoryTransformerConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_generation_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, random_attention_mask


if is_torch_available():
    import torch

    from transformers import TrajectoryTransformerModel
    from transformers.models.trajectory_transformer.modeling_trajectory_transformer import (
        TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


class TrajectoryTransformerModelTester:
    def __init__(self, parent, batch_size=13, n_embd=128, action_dim=6, observation_dim=17, is_training=True):
        self.parent = parent
        self.batch_size = batch_size
        self.n_embd = n_embd
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.is_training = is_training
        self.seq_length = self.action_dim + self.observation_dim + 1

    def prepare_config_and_inputs(self):
        trajectories = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(self.batch_size)]).to(
            torch_device
        )
        attention_mask = random_attention_mask((self.batch_size, self.seq_length)).to(torch_device)
        targets = torch.LongTensor([np.random.permutation(self.seq_length) for _ in range(self.batch_size)]).to(
            torch_device
        )

        config = self.get_config()
        return config, trajectories, attention_mask, targets

    def get_config(self):
        return TrajectoryTransformerConfig(
            batch_size=self.batch_size,
            n_embd=self.n_embd,
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
        )

    def create_and_check_model(self, config, input_dict):
        model = TrajectoryTransformerModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(trajectories=input_dict["trajectories"], attention_mask=input_dict["attention_mask"])
        result = model(
            trajectories=input_dict["trajectories"],
            output_hidden_states=True,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )

        self.parent.assertEqual(result.hidden_states[-1].shape, (self.batch_size, self.seq_length, self.n_embd))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, trajectories, attention_mask, targets) = config_and_inputs
        inputs_dict = {"trajectories": trajectories, "attention_mask": attention_mask, "targets": targets}
        return config, inputs_dict


@require_torch
class TrajectoryTransformerModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):

    all_model_classes = (TrajectoryTransformerModel,) if is_torch_available() else ()

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
    test_torchscript = False

    def setUp(self):
        self.model_tester = TrajectoryTransformerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TrajectoryTransformerConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_conditional_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["trajectories"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    # # Input is 'trajectories' not 'input_ids'
    def test_model_main_input_name(self):
        model_signature = inspect.signature(getattr(TrajectoryTransformerModel, "forward"))
        # The main input is the name of the argument after `self`
        observed_main_input_name = list(model_signature.parameters.keys())[1]
        self.assertEqual(TrajectoryTransformerModel.main_input_name, observed_main_input_name)

    def test_retain_grad_hidden_states_attentions(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        model = TrajectoryTransformerModel(config)
        model.to(torch_device)

        outputs = model(
            trajectories=input_dict["trajectories"],
            attention_mask=input_dict["attention_mask"],
            targets=input_dict["targets"],
            output_hidden_states=True,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )

        output = outputs[0]
        hidden_states = outputs.hidden_states[0]
        hidden_states.retain_grad()

        if self.has_attentions:
            attentions = outputs.attentions[0]
            attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)

        if self.has_attentions:
            self.assertIsNotNone(attentions.grad)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        model = TrajectoryTransformerModel(config)
        model.to(torch_device)
        model.train()
        loss = model(
            trajectories=input_dict["trajectories"],
            attention_mask=input_dict["attention_mask"],
            targets=input_dict["targets"],
            output_hidden_states=True,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        ).loss
        loss.backward()

    def test_training_gradient_checkpointing(self):
        if not self.model_tester.is_training:
            return

        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        model = TrajectoryTransformerModel(config)
        model.gradient_checkpointing_enable()
        model.to(torch_device)
        model.train()
        loss = model(
            trajectories=input_dict["trajectories"],
            attention_mask=input_dict["attention_mask"],
            targets=input_dict["targets"],
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        ).loss
        loss.backward()

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    @slow
    def test_model_from_pretrained(self):
        for model_name in TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TrajectoryTransformerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class TrajectoryTransformerModelIntegrationTest(unittest.TestCase):
    @slow
    def test_prediction(self):
        batch_size = 1

        config = TrajectoryTransformerConfig.from_pretrained("CarlCochet/trajectory-transformer-halfcheetah-medium-v2")
        model = TrajectoryTransformerModel.from_pretrained(
            "CarlCochet/trajectory-transformer-halfcheetah-medium-v2", config=config
        )
        model.to(torch_device)
        model.eval()

        seq_length = model.config.action_dim + model.config.observation_dim + 1

        trajectories = torch.LongTensor(
            [[3, 19, 20, 22, 9, 7, 23, 10, 18, 14, 13, 4, 17, 11, 5, 6, 15, 21, 2, 8, 1, 0, 12, 16]]
        ).to(torch_device)
        outputs = model(
            trajectories=trajectories,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )

        output = outputs.logits

        expected_shape = torch.Size((batch_size, seq_length, model.config.vocab_size + 1))
        expected_slice = torch.tensor(
            [[[-0.7193, -0.2532, -0.0898], [1.9429, 2.0434, 2.3975], [-3.3651, -2.8744, -2.4532]]]
        ).to(torch_device)
        output_slice = output[:, :3, :3]

        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.allclose(output_slice, expected_slice, atol=1e-4))
