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
""" Testing suite for the PyTorch Swin2SR model. """
import inspect
import unittest

from transformers import Swin2SRConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
    from torch import nn

    from transformers import Swin2SRForImageSuperResolution, Swin2SRModel
    from transformers.models.swin2sr.modeling_swin2sr import SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST

if is_vision_available():
    from PIL import Image


class Swin2SRModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        patch_size=1,
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
        use_labels=False,
        upscale=2,
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
        self.upscale = upscale

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return Swin2SRConfig(
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
            upscale=self.upscale,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = Swin2SRModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        expected_seq_len = ((config.image_size // config.patch_size) ** 2) // (4 ** (len(config.depths) - 1))
        expected_dim = int(config.embed_dim * 2 ** (len(config.depths) - 1))

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, expected_seq_len, expected_dim))

    def create_and_check_for_image_super_resolution(self, config, pixel_values, labels):
        model = Swin2SRForImageSuperResolution(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        expected_image_size = self.image_size * self.upscale
        self.parent.assertEqual(
            result.reconstruction.shape, (self.batch_size, self.num_channels, expected_image_size, expected_image_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class Swin2SRModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (Swin2SRModel, Swin2SRForImageSuperResolution) if is_torch_available() else ()

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = Swin2SRModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Swin2SRConfig, embed_dim=37)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_for_image_super_resolution(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_super_resolution(*config_and_inputs)

    @unittest.skip(reason="Swin2SR does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Swin2SR does not support training yet")
    def test_training(self):
        pass

    @unittest.skip(reason="Swin2SR does not support training yet")
    def test_training_gradient_checkpointing(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    @slow
    def test_model_from_pretrained(self):
        for model_name in SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = Swin2SRModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_vision
@require_torch
class Swin2SRModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_image_super_resolution_head(self):
        # TODO update to appropriate organization
        model = Swin2SRForImageSuperResolution.from_pretrained("nielsr/swin2SR-classical-sr-x2-64").to(torch_device)
        feature_extractor = self.default_feature_extractor

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)
        expected_slice = torch.tensor([-0.3947, -0.4306, 0.0026]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
