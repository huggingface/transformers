# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch SwiftFormer model. """


import inspect
import unittest

from transformers import SwiftFormerConfig
from transformers.testing_utils import (
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import SwiftFormerForImageClassification, SwiftFormerModel
    from transformers.models.swiftformer.modeling_swiftformer import SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import ViTImageProcessor


class SwiftFormerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        num_channels=3,
        is_training=True,
        use_labels=True,
        # hidden_size=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        ###
        layers=[3, 3, 6, 4],
        embed_dims=[48, 56, 112, 220],
        mlp_ratios=4,
        downsamples=[True, True, True, True],
        vit_num=1,
        act_layer="gelu",
        num_labels=1000,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
    ):
        self.parent = parent

        self.batch_size = batch_size

        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels

        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.type_sequence_label_size = num_labels

        self.image_size = 224

        # self.

        # in SwiftFormer, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return SwiftFormerConfig(
            layers=[3, 3, 6, 4],
            embed_dims=[48, 56, 112, 220],
            mlp_ratios=4,
            downsamples=[True, True, True, True],
            vit_num=1,
            act_layer="gelu",
            num_labels=1000,
            down_patch_size=3,
            down_stride=2,
            down_pad=1,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_layer_scale=True,
            layer_scale_init_value=1e-5,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = SwiftFormerForImageClassification(config=config)
        model.to(torch_device)
        model.eval()
        model(pixel_values)
        # self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size)) #TODO

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.type_sequence_label_size
        model = SwiftFormerForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        model = SwiftFormerForImageClassification(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

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
class SwiftFormerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SwiftFormer does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (SwiftFormerModel, SwiftFormerForImageClassification) if is_torch_available() else ()

    pipeline_model_mapping = (
        {"feature-extraction": SwiftFormerModel, "image-classification": SwiftFormerForImageClassification}
        if is_torch_available()
        else {}
    )

    fx_compatible = False

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    has_attentions = False

    def setUp(self):
        self.model_tester = SwiftFormerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=SwiftFormerConfig,
            has_text_modality=False,
            hidden_size=37,
            num_attention_heads=12,
            num_hidden_layers=12,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SwiftFormer does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            # self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
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

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = SwiftFormerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    ####
    @unittest.skip(reason="SwiftFormer does not output attentions")
    def test_attention_outputs(self):
        pass

    def test_hidden_states_output(self):
        pass

    def test_retain_grad_hidden_states_attentions(self):
        pass

    def test_initialization(self):
        pass


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class SwiftFormerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return ViTImageProcessor.from_pretrained("shehan97/swiftformer-xs") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = SwiftFormerForImageClassification.from_pretrained("shehan97/swiftformer-xs").to(torch_device)

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        # expected_slice = torch.tensor([-0.2744, 0.8215, -0.0836]).to(torch_device)

        # self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
