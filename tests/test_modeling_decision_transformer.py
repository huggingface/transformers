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


from os import stat
import unittest

from tests.test_modeling_common import floats_tensor
from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from transformers import DecisionTransformerConfig
from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask, floats_tensor


if is_torch_available():
    import torch

    from transformers import (
        DecisionTransformerModel,
    )
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
            state_dim=17
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.act_dim = act_dim
        self.state_dim = state_dim
        
        

    def prepare_config_and_inputs(self):
        states = floats_tensor((self.batch_size, self.seq_length, self.state_dim))
        actions = floats_tensor((self.batch_size, self.seq_length, self.act_dim))
        rewards = floats_tensor((self.batch_size, self.seq_length, 1))
        #dones = ids_tensor((self.batch_size, self.seq_length, 1), vocab_size=2)
        rtg = floats_tensor((self.batch_size, self.seq_length, 1))
        timesteps = ids_tensor((self.batch_size, self.seq_length, 1), vocab_size=1000)
        attention_mask = random_attention_mask((self.batch_size, self.seq_length, 1))
        
        config = self.get_config()

        return config, states, actions, rewards, rtg, timesteps, attention_mask

    def get_config(self):
        return DecisionTransformerConfig(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            act_dim=self.act_dim,
            state_dim=self.state_dim,
        )

    def create_and_check_model(
            self, config, states, actions, rewards, rtg, timesteps, attention_mask
    ):
        model = DecisionTransformerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(states, actions, rewards, rtg, timesteps, attention_mask)
        
        self.parent.assertEqual(result.state_preds.shape, states.shape)
        self.parent.assertEqual(result.action_preds.shape, actions.shape)
        self.parent.assertEqual(result.return_preds.shape, rtg.shape)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))


    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config, states, actions, rewards, rtg, timesteps, attention_mask
        ) = config_and_inputs
        inputs_dict = {"states": states, "actions": actions, "rewards": rewards,
            "rtg": rtg, "timesteps": timesteps, "attention_mask":attention_mask,}
        return config, inputs_dict


@require_torch
class DecisionTransformerModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            DecisionTransformerModel,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = () 

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


# @require_torch
# class DecisionTransformerModelIntegrationTest(unittest.TestCase):
#     @slow
#     def test_inference_masked_lm(self):
#         model = DecisionTransformerForMaskedLM.from_pretrained("decision_transformer")
#         input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
#         output = model(input_ids)[0]

#         # TODO Replace vocab size
#         vocab_size = 32000

#         expected_shape = torch.Size((1, 6, vocab_size))
#         self.assertEqual(output.shape, expected_shape)

#         # TODO Replace values below with what was printed above.
#         expected_slice = torch.tensor(
#             [[[-0.0483, 0.1188, -0.0313], [-0.0606, 0.1435, 0.0199], [-0.0235, 0.1519, 0.0175]]]
#         )

#         self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))


