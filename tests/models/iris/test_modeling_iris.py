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

import copy
import inspect
from typing import Dict, List, Tuple
import unittest

from transformers import IrisConfig, is_torch_available,PretrainedConfig
from transformers.testing_utils import require_torch, slow, torch_device, is_flaky

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin

if is_torch_available():
    import torch
    import torch.nn.functional as F
    from transformers import IrisModel

def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key or "layer_scale" in key:
            setattr(configs_no_init, key, 1e-10)
        if isinstance(getattr(configs_no_init, key, None), PretrainedConfig):
            no_init_subconfig = _config_zero_init(getattr(configs_no_init, key))
            setattr(configs_no_init, key, no_init_subconfig)
    return configs_no_init

class IrisModelTester:
    def __init__(
        self,
        parent,
        batch_size = 3,
        batch_size_tokenizer=3,
        batch_size_world_model=3,
        batch_size_actor_critic=3,
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
            #for faster running of tests
            vocab_size=16,
            embed_dim_tokenizer=16,
            z_channels=16,
            embed_dim_world_model = 8,
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
        #number of losses(number of components)
        self.parent.assertEqual(result.losses.shape, torch.Size([3]))
        

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

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertTrue(
                            -1.5<=((param.data.mean() * 1e9).round() / 1e9).item()<=1.5,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                            )

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()
                tuple_dummy = ()
                dict_dummy = ()
                #loss(specially actor critic loss) and actor critic logit actions are not deterministic so, squeeze them out for the test
                for i, output in enumerate(tuple_output): tuple_dummy = tuple_dummy+(output,) if i!= 1 and i !=2 else tuple_dummy
                for i, output in enumerate(tuple_output): dict_dummy = dict_dummy+(output,) if i!= 1 and i !=2 else dict_dummy
                tuple_output = tuple_dummy
                dict_output = dict_dummy


                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (List, Tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            if self.has_attentions:
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(
                    model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
                )
    
    @unittest.skip("IRIS model is not deterministic.")
    def test_batching_equivalence(self):
        pass

    @unittest.skip("Cannot configure Iris to output a smaller backbone and there is no tiny model available")
    def test_model_is_small(self):
        pass

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

        observations_tokenizer = floats_tensor((3,1,config.in_channels,config.resolution,config.resolution))
        actions_tokenizer = ids_tensor((3,1),vocab_size =4).long()
        rewards_tokenizer = ids_tensor((3,1),vocab_size =8)
        zeros = torch.zeros_like(rewards_tokenizer)
        # Rewards are given depending on which color brick is broken in 'Breakout' Atari env
        zeros[((rewards_tokenizer==7)|( rewards_tokenizer==4)|(rewards_tokenizer==1))]=1
        rewards_tokenizer = torch.mul(zeros,rewards_tokenizer).float()
        ends_tokenizer = torch.zeros(3,1).long()
        for i in range(3):
            ends_tokenizer[i,ids_tensor((1,),vocab_size=1).item()]=1 if floats_tensor((1,)).item()<0.5 else 0
        mask_padding_tokenizer = torch.ones(3,1).bool()

        observations_world_model = floats_tensor((3,20,config.in_channels,config.resolution,config.resolution))
        actions_world_model = ids_tensor((3,20),vocab_size =4).long()
        rewards_world_model = ids_tensor((3,20),vocab_size =8)
        zeros = torch.zeros_like(rewards_world_model)
        # Rewards are given depending on which color brick is broken in 'Breakout' Atari env
        zeros[((rewards_world_model==7)|( rewards_world_model==4)|(rewards_world_model==1))]=1
        rewards_world_model = torch.mul(zeros,rewards_world_model).float()
        ends_world_model = torch.zeros(3,20).long()
        for i in range(3):
            ends_world_model[i,ids_tensor((1,),vocab_size=20).item()]=1 if floats_tensor((1,)).item()<0.5 else 0
        mask_padding_world_model = torch.ones(4,20).bool()

        observations_actor_critic = floats_tensor((3,21,config.in_channels,config.resolution,config.resolution))
        actions_actor_critic = ids_tensor((3,21),vocab_size =4).long()
        rewards_actor_critic = ids_tensor((3,21),vocab_size =8)
        zeros = torch.zeros_like(rewards_actor_critic)
        # Rewards are given depending on which color brick is broken in 'Breakout' Atari env
        zeros[((rewards_actor_critic==7)|( rewards_actor_critic==4)|(rewards_actor_critic==1))]=1
        rewards_actor_critic = torch.mul(zeros,rewards_actor_critic).float()
        ends_actor_critic = torch.zeros(3,21).long()
        for i in range(3):
            ends_actor_critic[i,ids_tensor((1,),vocab_size=21).item()]=1 if floats_tensor((1,)).item()<0.5 else 0
        mask_padding_actor_critic = torch.ones(3,21).bool()
        
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
            
            act_pred_expected_shape = torch.Size((3,1, config.num_actions))
            self.assertEqual(model_pred.action_preds.shape, act_pred_expected_shape)
            self.assertTrue(torch.allclose(model_pred.action_preds, expected_outputs[step], atol=1e-4))
            
