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
""" Testing suite for the PyTorch R3M model. """


import unittest

from ..test_modeling_common import floats_tensor
from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from transformers import R3MConfig
from ..test_configuration_common import ConfigTester
from ..test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        R3MModel,
    )
    from transformers.models.r3m.modeling_r3m import (
        R3M_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


class R3MModelTester:
    def __init__(
            self,
            parent,
            batch_size=32,
            resnet_size=34,
            is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.resnet_size = resnet_size
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        input_images = torch.randint(0, 255, (self.batch_size, 3, 224, 224))
        input_images.to(torch_device) 
        config = self.get_config()

        return config, input_images
      

    def get_config(self):
        return R3MConfig(
            resnet_size=self.resnet_size,
        )


    def create_and_check_model(
            self, config, input_images
    ):
        model = R3MModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_images)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, model.output_size))


@require_torch
class R3MModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            R3MModel,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = ()
    
    # Ignoring of a failing test from GenerationTesterMixin, as the model does not use inputs_ids
    test_generate_without_input_ids = False
    test_correct_missing_keys = False

    # Ignoring of a failing tests from ModelTesterMixin, as the model does not implement these features
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_attention_outputs = False
    test_hidden_states_output = False
    test_inputs_embeds = False
    test_model_common_attributes = False
    test_gradient_checkpointing = False

    def setUp(self):
        self.model_tester = R3MModelTester(self)
        self.config_tester = ConfigTester(self, config_class=R3MConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in R3M_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = R3MModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class R3MModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = R3MModel.from_pretrained("surajnair/r3m-50")
        input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        # TODO Replace vocab size
        vocab_size = 32000

        expected_shape = torch.Size((1, 6, vocab_size))
        self.assertEqual(output.shape, expected_shape)

        # TODO Replace values below with what was printed above.
        expected_slice = torch.tensor(
            [[[-0.0483, 0.1188, -0.0313], [-0.0606, 0.1435, 0.0199], [-0.0235, 0.1519, 0.0175]]]
        )

        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))


