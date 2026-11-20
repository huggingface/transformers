# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Pixio model."""

import unittest
from functools import cached_property

from transformers import PixioConfig
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...model_tester_config import ModelTesterConfigMixin
from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import PixioBackbone, PixioModel


if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class PixioModelTester(ModelTesterConfigMixin):
    defaults = {}
    small_config_overrides = {
        "batch_size": 13,
        "image_size": 30,
        "patch_size": 2,
        "num_channels": 3,
        "is_training": True,
        "use_labels": True,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 37,
        "n_cls_tokens": 1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "type_sequence_label_size": 10,
        "initializer_range": 0.02,
        "scope": None,
        "attn_implementation": "eager",
    }

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        # in Pixio, the seq length equals the number of patches + class tokens
        num_patches = (values["image_size"] // values["patch_size"]) ** 2
        values["seq_length"] = num_patches + values["n_cls_tokens"]
        return values

    def __init__(self, parent, **kwargs):
        self.parent = parent
        for key, value in self.get_small_config_defaults(**kwargs).items():
            setattr(self, key, value)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return PixioConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            n_cls_tokens=self.n_cls_tokens,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            attn_implementation=self.attn_implementation,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = PixioModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = PixioBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify hidden states
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        expected_size = self.image_size // config.patch_size
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape), [self.batch_size, model.channels[0], expected_size, expected_size]
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))

        # verify backbone works with out_features=None
        config.out_features = None
        model = PixioBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape), [self.batch_size, model.channels[0], expected_size, expected_size]
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)

        # verify backbone works with apply_layernorm=False and reshape_hidden_states=False
        config.apply_layernorm = False
        config.reshape_hidden_states = False

        model = PixioBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape), [self.batch_size, self.seq_length, self.hidden_size]
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        self.parent.skipTest(reason="Pixio currently exposes only the base model and backbone.")

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
class PixioModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Pixio does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    model_tester_class = PixioModelTester

    test_torch_exportable = True

    all_model_classes = (
        (
            PixioModel,
            PixioBackbone,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = {"image-feature-extraction": PixioModel} if is_torch_available() else {}

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = PixioModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PixioConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_backbone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_batching_equivalence(self, atol=1e-4, rtol=1e-4):
        super().test_batching_equivalence(atol=atol, rtol=rtol)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class PixioModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return AutoImageProcessor.from_pretrained("LiheYoung/pixio-vith16") if is_vision_available() else None

    @slow
    def test_inference_no_head(self):
        model = PixioModel.from_pretrained("LiheYoung/pixio-vith16").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the last hidden states
        expected_shape = torch.Size((1, 264, 1280))
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        slice_expectations = Expectations(
            {
                (None, None): [[0.7420, -1.4219, 0.1581], [0.3935, -1.4387, 0.2880], [0.2888, -1.4017, 0.3672]],
                ("cuda", (8, 0)): [[0.7420, -1.4219, 0.1581], [0.3935, -1.4387, 0.2880], [0.2888, -1.4017, 0.3672]],
                ("xpu", None): [[0.7420, -1.4219, 0.1581], [0.3935, -1.4387, 0.2880], [0.2888, -1.4017, 0.3672]],
            }
        )
        expected_slice = torch.tensor(slice_expectations.get_expectation(), device=torch_device)
        # valid the first three patch tokens
        torch.testing.assert_close(outputs.last_hidden_state[0, 8:11, :3], expected_slice, rtol=1e-4, atol=1e-4)


@require_torch
class PixioBackboneTest(unittest.TestCase, BackboneTesterMixin):
    model_tester_class = PixioModelTester

    all_model_classes = (PixioBackbone,) if is_torch_available() else ()
    config_class = PixioConfig

    has_attentions = False

    def setUp(self):
        self.model_tester = PixioModelTester(self)
