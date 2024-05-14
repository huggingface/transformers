# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Iris model. """


import inspect
import unittest

from transformers import IrisConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import IrisModel


class IrisModelTester:
    def __init__(
        self,
        parent,
        batch_size = 16,
        batch_size_tokenizer=16,
        batch_size_world_model=4,
        batch_size_actor_critic=4,
        seq_length_tokenizer=1,
        seq_length_world_model=20,
        seq_length_actor_critic=21,
        num_actions=4,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.batch_size_tokenizer = batch_size_tokenizer
        self.batch_size_world_model = batch_size_world_model
        self.batch_size_actor_critic = batch_size_actor_critic
        self.seq_length_tokenizer = seq_length_tokenizer
        self.seq_length_world_model = seq_length_world_model
        self.seq_length_actor_critic = seq_length_actor_critic
        self.num_actions = num_actions
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        config = self.get_config()

        self.hidden_size = config.embed_dim_world_model
        
        observations_tokenizer = floats_tensor((self.batch_size_tokenizer,self.seq_length_tokenizer,config.in_channels,config.resolution,config.resolution))
        actions_tokenizer = ids_tensor((self.batch_size_tokenizer,self.seq_length_tokenizer),vocab_size =4).long()
        rewards_tokenizer = ids_tensor((self.batch_size_tokenizer,self.seq_length_tokenizer),vocab_size =8)
        zeros = torch.zeros_like(rewards_tokenizer)
        # Rewards are given depending on which color brick is broken in 'Breakout' Atari env
        zeros[((rewards_tokenizer==7)|( rewards_tokenizer==4)|(rewards_tokenizer==1))]=1
        rewards_tokenizer = torch.mul(zeros,rewards_tokenizer).float()
        ends_tokenizer = torch.zeros(self.batch_size_tokenizer,self.seq_length_tokenizer).long()
        for i in range(self.batch_size_tokenizer):
            ends_tokenizer[i,ids_tensor((1,),vocab_size=1).item()]=1 if floats_tensor((1,)).item()<0.5 else 0
        mask_padding_tokenizer = torch.ones(self.batch_size_tokenizer,self.seq_length_tokenizer).bool()

        observations_world_model = floats_tensor((self.batch_size_world_model,self.seq_length_world_model,config.in_channels,config.resolution,config.resolution))
        actions_world_model = ids_tensor((self.batch_size_world_model,self.seq_length_world_model),vocab_size =4).long()
        rewards_world_model = ids_tensor((self.batch_size_world_model,self.seq_length_world_model),vocab_size =8)
        zeros = torch.zeros_like(rewards_world_model)
        # Rewards are given depending on which color brick is broken in 'Breakout' Atari env
        zeros[((rewards_world_model==7)|( rewards_world_model==4)|(rewards_world_model==1))]=1
        rewards_world_model = torch.mul(zeros,rewards_world_model).float()
        ends_world_model = torch.zeros(self.batch_size_world_model,self.seq_length_world_model).long()
        for i in range(self.batch_size_world_model):
            ends_world_model[i,ids_tensor((1,),vocab_size=1).item()]=1 if floats_tensor((1,)).item()<0.5 else 0
        mask_padding_world_model = torch.ones(self.batch_size_world_model,self.seq_length_world_model).bool()

        observations_actor_critic = floats_tensor((self.batch_size_actor_critic,self.seq_length_actor_critic,config.in_channels,config.resolution,config.resolution))
        actions_actor_critic = ids_tensor((self.batch_size_actor_critic,self.seq_length_actor_critic),vocab_size =4).long()
        rewards_actor_critic = ids_tensor((self.batch_size_actor_critic,self.seq_length_actor_critic),vocab_size =8)
        zeros = torch.zeros_like(rewards_actor_critic)
        # Rewards are given depending on which color brick is broken in 'Breakout' Atari env
        zeros[((rewards_actor_critic==7)|( rewards_actor_critic==4)|(rewards_actor_critic==1))]=1
        rewards_actor_critic = torch.mul(zeros,rewards_actor_critic).float()
        ends_actor_critic = torch.zeros(self.batch_size_actor_critic,self.seq_length_actor_critic).long()
        for i in range(self.batch_size_actor_critic):
            ends_actor_critic[i,ids_tensor((1,),vocab_size=1).item()]=1 if floats_tensor((1,)).item()<0.5 else 0
        mask_padding_actor_critic = torch.ones(self.batch_size_actor_critic,self.seq_length_actor_critic).bool()
        
        observations = [observations_tokenizer,observations_world_model,observations_actor_critic]
        actions = [actions_tokenizer,actions_world_model,actions_actor_critic]
        rewards = [rewards_tokenizer,rewards_world_model,rewards_actor_critic]
        ends = [ends_tokenizer,ends_world_model,ends_actor_critic]
        mask_padding = [mask_padding_tokenizer,mask_padding_world_model,mask_padding_actor_critic]

        return (
            config,
            observations,
            actions,
            rewards,
            ends,
            mask_padding,
        )

    def get_config(self):
        return IrisConfig(
            num_actions = self.num_actions,
        )

    def create_and_check_model(
        self,
        config,
        observations,
        actions,
        rewards,
        ends,
        mask_padding,
    ):
        model = IrisModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(observations, actions, rewards, ends, mask_padding)

        act_pred_expected_shape = torch.Size((self.batch_size_actor_critic,1, self.num_actions))
        reward_pred_expected_shape = torch.Size((self.batch_size_world_model,self.seq_length_world_model, 3))
        ep_end_pred_expected_shape = torch.Size((self.batch_size_world_model,self.seq_length_world_model, 2))
        obs_pred_expected_shape = torch.Size((self.batch_size_world_model,320,config.vocab_size))

        self.parent.assertEqual(result.reconstructed_img.shape, observations[0].squeeze(1).shape)
        self.parent.assertEqual(result.action_preds.shape, act_pred_expected_shape)
        self.parent.assertEqual(result.reward_preds.shape, reward_pred_expected_shape)
        self.parent.assertEqual(result.epsiode_end.shape, ep_end_pred_expected_shape)
        self.parent.assertEqual(result.obs_preds.shape, obs_pred_expected_shape)
        

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            observations,
            actions,
            rewards,
            ends,
            mask_padding,
        ) = config_and_inputs
        inputs_dict = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "ends": ends,
            "mask_padding": mask_padding,
        }
        return config, inputs_dict


@require_torch
class IrisModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (IrisModel,) if is_torch_available() else ()
    all_generative_model_classes = ()
    pipeline_model_mapping = {"feature-extraction": IrisModel} if is_torch_available() else {}
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
    test_gradient_checkpointing = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = IrisModelTester(self)
        self.config_tester = ConfigTester(self, config_class=IrisConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "ruffy369/iris-breakout"
        model = IrisModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "observations",
                "actions",
                "rewards",
                "ends",
                "mask_padding",
            ]

            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    # @unittest.skip(
    #     reason="Training is not supported yet, so the test is ignored."
    # )
    # def test_retain_grad_hidden_states_attentions(self):
    #     pass

@require_torch
class IrisModelIntegrationTest(unittest.TestCase):
    @slow
    def test_autoregressive_prediction(self):
        """
        An integration test that performs reconstruction of images as observations, action prediction with policy and autoregressive prediction of 
        new frame tokens, rewards and  potential episode termination from a sequence of interleaved frame and action tokens Test is performed over two timesteps.

        """

        NUM_STEPS = 1  # number of steps of prediction with all the three components i.e., tokenizer, world model & actor critic we will perform
        model = IrisModel.from_pretrained("ruffy369/iris-breakout")
        model = model.to(torch_device)
        config = model.config
        torch.manual_seed(0)
        # state = torch.randn(1, 1, config.state_dim).to(device=torch_device, dtype=torch.float32)  # env.reset()

        expected_outputs = torch.tensor([[[-2.1720,  0.1622, -4.3326,  4.6579]],
                                        [[-3.4466, -5.1955,  4.0899,  3.4191]],
                                        [[-1.3850,  1.0004, -3.9815,  3.1493]],
                                        [[-3.1353, -5.3218, -2.6076, 10.3681]]], device=torch_device)

        observations_tokenizer = floats_tensor((16,1,config.in_channels,config.resolution,config.resolution))
        actions_tokenizer = ids_tensor((16,1),vocab_size =4).long()
        rewards_tokenizer = ids_tensor((16,1),vocab_size =8)
        zeros = torch.zeros_like(rewards_tokenizer)
        # Rewards are given depending on which color brick is broken in 'Breakout' Atari env
        zeros[((rewards_tokenizer==7)|( rewards_tokenizer==4)|(rewards_tokenizer==1))]=1
        rewards_tokenizer = torch.mul(zeros,rewards_tokenizer).float()
        ends_tokenizer = torch.zeros(16,1).long()
        for i in range(16):
            ends_tokenizer[i,ids_tensor((1,),vocab_size=1).item()]=1 if floats_tensor((1,)).item()<0.5 else 0
        mask_padding_tokenizer = torch.ones(16,1).bool()

        observations_world_model = floats_tensor((4,20,config.in_channels,config.resolution,config.resolution))
        actions_world_model = ids_tensor((4,20),vocab_size =4).long()
        rewards_world_model = ids_tensor((4,20),vocab_size =8)
        zeros = torch.zeros_like(rewards_world_model)
        # Rewards are given depending on which color brick is broken in 'Breakout' Atari env
        zeros[((rewards_world_model==7)|( rewards_world_model==4)|(rewards_world_model==1))]=1
        rewards_world_model = torch.mul(zeros,rewards_world_model).float()
        ends_world_model = torch.zeros(4,20).long()
        for i in range(4):
            ends_world_model[i,ids_tensor((1,),vocab_size=20).item()]=1 if floats_tensor((1,)).item()<0.5 else 0
        mask_padding_world_model = torch.ones(4,20).bool()

        observations_actor_critic = floats_tensor((4,21,config.in_channels,config.resolution,config.resolution))
        actions_actor_critic = ids_tensor((4,21),vocab_size =4).long()
        rewards_actor_critic = ids_tensor((4,21),vocab_size =8)
        zeros = torch.zeros_like(rewards_actor_critic)
        # Rewards are given depending on which color brick is broken in 'Breakout' Atari env
        zeros[((rewards_actor_critic==7)|( rewards_actor_critic==4)|(rewards_actor_critic==1))]=1
        rewards_actor_critic = torch.mul(zeros,rewards_actor_critic).float()
        ends_actor_critic = torch.zeros(4,21).long()
        for i in range(4):
            ends_actor_critic[i,ids_tensor((1,),vocab_size=21).item()]=1 if floats_tensor((1,)).item()<0.5 else 0
        mask_padding_actor_critic = torch.ones(4,21).bool()
        
        observations = [observations_tokenizer,observations_world_model,observations_actor_critic]
        actions = [actions_tokenizer,actions_world_model,actions_actor_critic]
        rewards = [rewards_tokenizer,rewards_world_model,rewards_actor_critic]
        ends = [ends_tokenizer,ends_world_model,ends_actor_critic]
        mask_padding = [mask_padding_tokenizer,mask_padding_world_model,mask_padding_actor_critic]

        for step in range(NUM_STEPS):
            

            with torch.no_grad():
                model_pred = model(
                    observations = observations,
                    actions = actions,
                    rewards = rewards,
                    ends = ends,
                    mask_padding = mask_padding,
                    should_preprocess = True,
                    should_postprocess = True,
                    return_dict = True,
                )
            
            act_pred_expected_shape = torch.Size((4,1, config.num_actions))
            self.assertEqual(model_pred.action_preds.shape, act_pred_expected_shape)
            self.assertTrue(torch.allclose(model_pred.action_preds, expected_outputs[step], atol=1e-4))
            
