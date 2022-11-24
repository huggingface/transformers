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
""" Testing suite for the PyTorch AltCLIP model. """


import unittest

import numpy as np

from ...test_modeling_common import floats_tensor
from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from transformers import AltCLIPConfig, AltCLIPTextConfig
from ...test_configuration_common import ConfigTester
from ...models.clip.test_modeling_clip import CLIPVisionModelTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


if is_torch_available():
    import torch

    from transformers import (
        AltCLIPModel,
        AltCLIPTextModel
    )
    from transformers.models.altclip.modeling_altclip import (
        ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
    )

if is_vision_available():
    from PIL import Image

    from transformers import AltCLIPProcessor


class AltCLIPTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        projection_dim=32,
        project_dim=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.project_dim = project_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return AltCLIPTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            project_dim = self.project_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=1,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = AltCLIPTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.projection_dim))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class AltCLIPTextModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (AltCLIPTextModel,) if is_torch_available() else ()
    fx_compatible = True
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = AltCLIPTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AltCLIPTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_gradient_checkpointing_enable_disable(self):
        # config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
    
        # if not AltCLIPTextModel.supports_gradient_checkpointing:
        #     pass

        # # at init model should have gradient checkpointing disabled
        # model = AltCLIPTextModel(config)
        # self.assertFalse(model.is_gradient_checkpointing)

        # # check enable works
        # model.gradient_checkpointing_enable()
        # print(model)
        # print(model.is_gradient_checkpointing)
        # self.assertTrue(model.is_gradient_checkpointing)
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_torch_fx_output_loss(self):
        pass
#----------------------------------------------------------
    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="Result of the model is a dict")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="AltCLIP does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="AltCLIPTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="AltCLIPTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = AltCLIPTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class AltCLIPModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):

        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = AltCLIPTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = CLIPVisionModelTester(parent, **vision_kwargs)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config(text_config, vision_config)
        return config, input_ids, attention_mask, pixel_values

    def get_config(self, text_config, vision_config):
        return AltCLIPConfig(text_config_dict=text_config, vision_config_dict=vision_config.to_dict(), projection_dim=64)

    def create_and_check_model(
            self, config, input_ids, pixel_values, attention_mask):
        model = AltCLIPModel(config=config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            result = model(input_ids, pixel_values, attention_mask)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "return_loss": True,
        }
        return config, inputs_dict


@require_torch
class AltCLIPModelTest(ModelTesterMixin, unittest.TestCase):

    model_class = AltCLIPModel
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = AltCLIPModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AltCLIPConfig, hidden_size=37)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="CLIPModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass
    
    # override as the `logit_scale` parameter initilization is different for AltCLIP
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        model = self.model_class(config=configs_no_init)
        for name, param in model.named_parameters():
            if param.requires_grad:
                # check if `logit_scale` is initilized as per the original implementation
                if name == "logit_scale":
                    self.assertAlmostEqual(
                        param.data.item(),
                        np.log(1 / 0.07),
                        delta=1e-3,
                        msg=f"Parameter {name} of model {self.model_class} seems not properly initialized",
                    )
                else:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {self.model_class} seems not properly initialized",
                    )

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        model = self.model_class(config=configs_no_init)
        model.to(torch_device)
        model.eval()

        try:
            input_ids = inputs_dict["input_ids"]
            pixel_values = inputs_dict["pixel_values"]  # CLIP needs pixel_values
            traced_model = torch.jit.trace(model, (input_ids, pixel_values))
        except RuntimeError:
            self.fail("Couldn't trace module.")

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

            try:
                torch.jit.save(traced_model, pt_file_name)
            except Exception:
                self.fail("Couldn't save module.")

            try:
                loaded_model = torch.jit.load(pt_file_name)
            except Exception:
                self.fail("Couldn't load module.")

        model.to(torch_device)
        model.eval()

        loaded_model.to(torch_device)
        loaded_model.eval()

        model_state_dict = model.state_dict()
        loaded_model_state_dict = loaded_model.state_dict()

        self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

        models_equal = True
        for layer_name, p1 in model_state_dict.items():
            p2 = loaded_model_state_dict[layer_name]
            if p1.data.ne(p2.data).sum() > 0:
                models_equal = False

        self.assertTrue(models_equal)

    @slow
    def test_model_from_pretrained(self):
        for model_name in ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = AltCLIPModel.from_pretrained(model_name)
            self.assertIsNotNone(model)