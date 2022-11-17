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
""" Testing suite for the PyTorch FAN model. """


import inspect
import unittest

from PIL import Image

from transformers import FANConfig
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor

if is_torch_available():
    import torch
    from torch import nn

    from transformers import FANFeatureExtractor, FANForImageClassification, FANForSemanticSegmentation, FANModel
    from transformers.models.fan.modeling_fan import FAN_PRETRAINED_MODEL_ARCHIVE_LIST


# Copied and adapted from transformers.tests.models.deit.test_modeling_deit.py
class FANModelTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        image_size=224,
        patch_size=16,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=384,
        num_hidden_layers=16,
        num_attention_heads=8,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=1000,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.num_labels = num_labels

        # in DeiT, the seq length equals the number of patches + 2 (we add 2 for the [CLS] and distilation tokens)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches
        self.num_hidden_states_types = 2  # Optional Backbone Hidden States

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def get_config(self):
        return FANConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = FANModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        # seq_length +1 to account for class tokens
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.seq_length + 1, self.hidden_size)
        )

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.type_sequence_label_size
        model = FANForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        # test greyscale images
        config.num_channels = 1
        model = FANForImageClassification(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    # Copied and adapted from transformers.tests.models.deit.test_modeling_deit.py


class FANModelSegmenentationTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        image_size=224,
        patch_size=16,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=448,
        num_hidden_layers=16,
        num_attention_heads=8,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=1000,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
        sharpen_attn=False,
        depths=[3, 3],
        dims=[128, 256, 512, 1024],
        use_head=False,
        backbone="hybrid",
        segmentation_in_channels=[128, 256, 448, 448],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=256,
        decoder_dropout=0.1,
        decoder_hidden_size=768,
        out_index=15,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.num_labels = num_labels
        # Added For Segmentation
        self.sharpen_attn = sharpen_attn
        self.depths = depths
        self.dims = dims
        self.use_head = use_head
        self.backbone = backbone
        self.segmentation_in_channels = segmentation_in_channels
        self.in_index = in_index
        self.feature_strides = feature_strides
        self.channels = channels
        self.decoder_dropout = decoder_dropout
        self.decoder_hidden_size = decoder_hidden_size
        self.out_index = out_index

        # in DeiT, the seq length equals the number of patches + 2 (we add 2 for the [CLS] and distilation tokens)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches
        self.num_hidden_states_types = 2  # Optional Backbone Hidden States

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def get_config(self):
        return FANConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            sharpen_attn=self.sharpen_attn,
            depths=self.depths,
            dims=self.dims,
            use_head=self.use_head,
            backbone=self.backbone,
            segmentation_in_channels=self.segmentation_in_channels,
            in_index=self.in_index,
            feature_strides=self.feature_strides,
            channels=self.channels,
            decoder_dropout=self.decoder_dropout,
            decoder_hidden_size=self.decoder_hidden_size,
            out_index=self.out_index,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = FANModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        # seq_length +1 to account for class tokens
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.seq_length + 1, self.hidden_size)
        )


@require_torch
class FANModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            FANModel,
            FANForImageClassification,
        )
        if is_torch_available()
        else ()
    )
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = FANModelTester(self)
        self.config_tester = ConfigTester(self, config_class=FANConfig, hidden_size=37, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="FAN does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    @slow
    def test_model_from_pretrained(self):
        for model_name in FAN_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = FANModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)


@require_torch
class FANModelSegmentationTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (FANForSemanticSegmentation,) if is_torch_available() else ()
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = FANModelSegmenentationTester(self)
        self.config_tester = ConfigTester(self, config_class=FANConfig, hidden_size=37, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="FAN does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    @slow
    def test_model_from_pretrained(self):
        for model_name in FAN_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = FANModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
# Copied from transformers.tests.models.deit.test_modeling_deit.py
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
class FANModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return FANFeatureExtractor() if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        IMG_CLF_PATH = "ksmcg/fan_base_18_p16_224"
        model = FANForImageClassification.from_pretrained(IMG_CLF_PATH).to(torch_device)
        model.eval()
        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([0.4830, -0.7349, -0.4465]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
