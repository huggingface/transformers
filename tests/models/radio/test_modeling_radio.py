# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Testing suite for the PyTorch RADIO model."""

import unittest

from transformers import RadioConfig
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import RadioModel


class RadioModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=32,
        patch_size=4,
        num_channels=3,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        mlp_ratio=2.0,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        drop_path_rate=0.0,
        layerscale_value=1.0,
        max_img_size=32,
        num_cls_tokens=2,
        num_registers=3,
        summary_idxs=None,
        initializer_range=0.02,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.layerscale_value = layerscale_value
        self.max_img_size = max_img_size
        self.num_cls_tokens = num_cls_tokens
        self.num_registers = num_registers
        self.summary_idxs = summary_idxs if summary_idxs is not None else [0, 1]
        self.initializer_range = initializer_range
        self.is_training = is_training

        self.num_patches = (image_size // patch_size) ** 2
        self.num_prefix_tokens = num_cls_tokens + num_registers
        self.seq_length = self.num_prefix_tokens + self.num_patches

    def get_config(self):
        return RadioConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            mlp_ratio=self.mlp_ratio,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
            drop_path_rate=self.drop_path_rate,
            layerscale_value=self.layerscale_value,
            num_channels=self.num_channels,
            patch_size=self.patch_size,
            image_size=self.image_size,
            max_img_size=self.max_img_size,
            num_cls_tokens=self.num_cls_tokens,
            num_registers=self.num_registers,
            summary_idxs=self.summary_idxs,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def create_and_check_model(self, config, pixel_values):
        model = RadioModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)

        expected_summary_size = len(self.summary_idxs) * self.hidden_size
        self.parent.assertEqual(result.summary.shape, (self.batch_size, expected_summary_size))
        self.parent.assertEqual(result.features.shape, (self.batch_size, self.num_patches, self.hidden_size))
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_layer_scale_init(self, config, pixel_values):
        model = RadioModel(config=config)
        for layer in model.encoder.layer:
            self.parent.assertTrue(
                torch.allclose(layer.layer_scale1.lambda1, torch.ones_like(layer.layer_scale1.lambda1)),
                "layer_scale1.lambda1 should be initialized to 1.0",
            )
            self.parent.assertTrue(
                torch.allclose(layer.layer_scale2.lambda1, torch.ones_like(layer.layer_scale2.lambda1)),
                "layer_scale2.lambda1 should be initialized to 1.0",
            )

    def create_and_check_variable_resolution(self, config, pixel_values):
        model = RadioModel(config=config)
        model.to(torch_device)
        model.eval()

        # Test a different resolution (2x): num_patches quadruples
        large_size = self.image_size * 2
        large_pixel_values = floats_tensor([self.batch_size, self.num_channels, large_size, large_size])
        large_pixel_values = large_pixel_values.to(torch_device)
        expected_patches = (large_size // self.patch_size) ** 2

        with torch.no_grad():
            result = model(large_pixel_values)

        self.parent.assertEqual(result.features.shape, (self.batch_size, expected_patches, self.hidden_size))


@require_torch
class RadioModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as RadioModel
    does not use input_ids, inputs_embeds, or attention_mask.
    """

    all_model_classes = (RadioModel,) if is_torch_available() else ()
    pipeline_model_mapping = {}
    test_resize_embeddings = False
    test_head_masking = False
    test_pruning = False

    def setUp(self):
        self.model_tester = RadioModelTester(self)
        self.config_tester = ConfigTester(self, config_class=RadioConfig, has_text_modality=False, hidden_size=16)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, pixel_values)

    def test_layer_scale_init(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_layer_scale_init(config, pixel_values)

    def test_variable_resolution(self):
        config, pixel_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_variable_resolution(config, pixel_values)

    @unittest.skip(reason="RadioModel does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="RadioModel does not use inputs_embeds")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="RadioModel does not support feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="RadioModel uses pixel_values, not token embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(
        reason="The shared 'radio' conversion mapping includes a video_embedder rename for video-capable "
        "checkpoints; the image-only RadioModel has no matching key, so the reverse-mapping check does not apply."
    )
    def test_reverse_loading_mapping(self):
        pass

    @unittest.skip(
        reason="RadioModel has no classification head, so the test body is a no-op; its `_config_zero_init` helper "
        "also sets the `_std`-suffixed `norm_std` config field to a scalar, which the strict RadioConfig rejects."
    )
    def test_can_load_ignoring_mismatched_shapes(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = RadioModel.from_pretrained("nvidia/C-RADIOv4-H")
        self.assertIsNotNone(model)

    @slow
    def test_inference(self):
        model = RadioModel.from_pretrained("nvidia/C-RADIOv4-H").to(torch_device)
        model.eval()

        torch.manual_seed(42)
        pixel_values = torch.randn(1, 3, 224, 224, device=torch_device)

        with torch.no_grad():
            outputs = model(pixel_values)

        self.assertEqual(outputs.summary.shape, (1, 2560))
        self.assertEqual(outputs.features.shape, (1, 196, 1280))
        self.assertFalse(outputs.summary.isnan().any(), "summary contains NaN")
        self.assertFalse(outputs.features.isnan().any(), "features contain NaN")
