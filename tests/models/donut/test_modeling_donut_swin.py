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
""" Testing suite for the PyTorch Donut Swin model. """

import collections
import unittest

from transformers import DonutSwinConfig
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import DonutSwinModel


class DonutSwinModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        patch_size=2,
        num_channels=3,
        embed_dim=16,
        depths=[1, 2, 1],
        num_heads=[2, 2, 4],
        window_size=2,
        mlp_ratio=2.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        patch_norm=True,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        is_training=True,
        scope=None,
        use_labels=True,
        type_sequence_label_size=10,
        encoder_stride=8,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.patch_norm = patch_norm
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.is_training = is_training
        self.scope = scope
        self.use_labels = use_labels
        self.type_sequence_label_size = type_sequence_label_size
        self.encoder_stride = encoder_stride

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DonutSwinConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            drop_path_rate=self.drop_path_rate,
            hidden_act=self.hidden_act,
            use_absolute_embeddings=self.use_absolute_embeddings,
            path_norm=self.patch_norm,
            layer_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            encoder_stride=self.encoder_stride,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = DonutSwinModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        expected_seq_len = ((config.image_size // config.patch_size) ** 2) // (4 ** (len(config.depths) - 1))
        expected_dim = int(config.embed_dim * 2 ** (len(config.depths) - 1))

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, expected_seq_len, expected_dim))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class DonutSwinModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (DonutSwinModel,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-feature-extraction": DonutSwinModel} if is_torch_available() else {}
    fx_compatible = True

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = DonutSwinModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DonutSwinConfig, embed_dim=37)

    def test_config(self):
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        return

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_inputs_embeds(self):
        # DonutSwin does not use inputs_embeds
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            expected_num_attentions = len(self.model_tester.depths)
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            window_size_squared = config.window_size**2
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_heads[0], window_size_squared, window_size_squared],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            else:
                # also another +1 for reshaped_hidden_states
                added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), expected_num_attentions)

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_heads[0], window_size_squared, window_size_squared],
            )

    def check_hidden_states_output(self, inputs_dict, config, model_class, image_size):
        model = model_class(config)
        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

        hidden_states = outputs.hidden_states

        expected_num_layers = getattr(
            self.model_tester, "expected_num_hidden_layers", len(self.model_tester.depths) + 1
        )
        self.assertEqual(len(hidden_states), expected_num_layers)

        # DonutSwin has a different seq_length
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )

        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        self.assertListEqual(
            list(hidden_states[0].shape[-2:]),
            [num_patches, self.model_tester.embed_dim],
        )

        reshaped_hidden_states = outputs.reshaped_hidden_states
        self.assertEqual(len(reshaped_hidden_states), expected_num_layers)

        batch_size, num_channels, height, width = reshaped_hidden_states[0].shape
        reshaped_hidden_states = (
            reshaped_hidden_states[0].view(batch_size, num_channels, height * width).permute(0, 2, 1)
        )
        self.assertListEqual(
            list(reshaped_hidden_states.shape[-2:]),
            [num_patches, self.model_tester.embed_dim],
        )

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        image_size = (
            self.model_tester.image_size
            if isinstance(self.model_tester.image_size, collections.abc.Iterable)
            else (self.model_tester.image_size, self.model_tester.image_size)
        )

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            self.check_hidden_states_output(inputs_dict, config, model_class, image_size)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            self.check_hidden_states_output(inputs_dict, config, model_class, image_size)

    def test_hidden_states_output_with_padding(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.patch_size = 3

        image_size = (
            self.model_tester.image_size
            if isinstance(self.model_tester.image_size, collections.abc.Iterable)
            else (self.model_tester.image_size, self.model_tester.image_size)
        )
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )

        padded_height = image_size[0] + patch_size[0] - (image_size[0] % patch_size[0])
        padded_width = image_size[1] + patch_size[1] - (image_size[1] % patch_size[1])

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            self.check_hidden_states_output(inputs_dict, config, model_class, (padded_height, padded_width))

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            self.check_hidden_states_output(inputs_dict, config, model_class, (padded_height, padded_width))

    @slow
    def test_model_from_pretrained(self):
        model_name = "naver-clova-ix/donut-base"
        model = DonutSwinModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if "embeddings" not in name and param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )
