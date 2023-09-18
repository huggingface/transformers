# coding=utf-8
# Copyright 2023 The Intel Team Authors, The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch TVP model. """


import inspect
import unittest

from transformers import TVPConfig, TVPVisionConfig
from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_torch_available

from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import TVPModel


class TVPVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        num_frames=48,
        image_size=448,
        input_format="BGR",
        features=["res5"],
        resnets_depth=50,
        resnets_num_groups=1,
        resnets_width_per_group=64,
        resnets_stem_input_channels=3,
        resnets_stem_out_channels=64,
        resnets_res_out_channels=256,
        resnets_res_dilation=1,
        backbone_freeze_at=2,
        grid_encoder_conv_input_size=2048,
        grid_encoder_conv_output_size=768,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.input_format = input_format
        self.features = features
        self.resnets_depth = resnets_depth
        self.resnets_num_groups = resnets_num_groups
        self.resnets_width_per_group = resnets_width_per_group
        self.resnets_stem_input_channels = resnets_stem_input_channels
        self.resnets_stem_out_channels = resnets_stem_out_channels
        self.resnets_res_out_channels = resnets_res_out_channels
        self.resnets_res_dilation = resnets_res_dilation
        self.backbone_freeze_at = backbone_freeze_at
        self.grid_encoder_conv_input_size = grid_encoder_conv_input_size
        self.grid_encoder_conv_output_size = grid_encoder_conv_output_size
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_channels = 3

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [self.batch_size, self.num_frames, self.num_channels, self.image_size, self.image_size]
        )
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return TVPVisionConfig(
            input_format=self.input_format,
            features=self.features,
            resnets_depth=self.resnets_depth,
            resnets_num_groups=self.resnets_num_groups,
            resnets_width_per_group=self.resnets_width_per_group,
            resnets_stem_input_channels=self.resnets_stem_input_channels,
            resnets_stem_out_channels=self.resnets_stem_out_channels,
            resnets_res_out_channels=self.resnets_res_out_channels,
            resnets_res_dilation=self.resnets_res_dilation,
            backbone_freeze_at=self.backbone_freeze_at,
            grid_encoder_conv_input_size=self.grid_encoder_conv_input_size,
            grid_encoder_conv_output_size=self.grid_encoder_conv_output_size,
        )


class TVPModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=8,
        vision_kwargs=None,
        alpha=1.0,
        beta=0.1,
        vp_type="framepad",
        vp_apply="replace",
        max_img_size=448,
        pad_size=96,
        num_frm=48,
        vocab_size=30522,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=512,
        max_grid_col_position_embeddings=100,
        max_grid_row_position_embeddings=100,
        hidden_dropout_prob=0.1,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        pad_token_id=0,
        type_vocab_size=2,
        attention_probs_dropout_prob=0.1,
    ):
        if vision_kwargs is None:
            vision_kwargs = {}

        vision_kwargs["batch_size"] = batch_size
        vision_kwargs["num_frames"] = num_frm
        vision_kwargs["image_size"] = max_img_size

        self.parent = parent
        self.vision_model_tester = TVPVisionModelTester(parent, **vision_kwargs)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.alpha = alpha
        self.beta = beta
        self.vp_type = vp_type
        self.vp_apply = vp_apply
        self.max_img_size = max_img_size
        self.pad_size = pad_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_grid_col_position_embeddings = max_grid_col_position_embeddings
        self.max_grid_row_position_embeddings = max_grid_row_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.type_vocab_size = type_vocab_size
        self.is_training = False

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return (config, input_ids, pixel_values, attention_mask)

    def get_config(self):
        return TVPConfig.from_vision_configs(
            vision_config=self.vision_model_tester.get_config(),
            alpha=self.alpha,
            beta=self.beta,
            vp_type=self.vp_type,
            vp_apply=self.vp_apply,
            max_img_size=self.max_img_size,
            pad_size=self.pad_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_grid_col_position_embeddings=self.max_grid_col_position_embeddings,
            max_grid_row_position_embeddings=self.max_grid_row_position_embeddings,
            layer_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            type_vocab_size=self.type_vocab_size,
        )

    def create_and_check_model(self, config, input_ids, pixel_values, attention_mask):
        model = TVPModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, pixel_values, attention_mask)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, 2))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, pixel_values, attention_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "pixel_values": pixel_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class TVPModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as TVP does not use, inputs_embeds.
    The seq_length in TVP contain textual and visual inputs, and prompt.
    """

    all_model_classes = (TVPModel,) if is_torch_available() else ()
    pipeline_model_mapping = {"temporal-video-grounding": TVPModel} if is_torch_available() else {}

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False
    is_training = False

    def setUp(self):
        self.model_tester = TVPModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="TVP does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="TVPModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="TVPModel does not have small model")
    def test_model_is_small(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_ids"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    # override as the `logit_scale` parameter initilization is different for TVP
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # cnn params and prompt params are randomly initialized.
                    if name.startswith("cnn") or name.startswith("tp") or "prompt" in name:
                        self.assertAlmostEqual(
                            param.data.mean().item(),
                            0,
                            delta=1e-1,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_attention_outputs(self):
        if not self.has_attentions:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 1, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            expected_num_layers = self.model_tester.num_hidden_layers + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertEqual(hidden_states[0].shape[-1], self.model_tester.hidden_size)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)
