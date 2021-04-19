# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch PiT model. """


import inspect
import math
import unittest

from transformers.file_utils import cached_property, is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import PiTConfig, PiTForImageClassification, PiTModel
    from transformers.models.pit.modeling_pit import PIT_PRETRAINED_MODEL_ARCHIVE_LIST, to_2tuple


if is_vision_available():
    from PIL import Image

    from transformers import PiTFeatureExtractor


class PiTModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        heads=(2, 2, 2),
        depths=(2, 2, 2),
        base_dims=(16, 16, 16),
        stride=4,
        conv_pooling_stride=2,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_labels=10,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.base_dims = base_dims
        self.depths = depths
        self.heads = heads
        self.stride = stride
        self.conv_pooling_stride = conv_pooling_stride
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_labels = num_labels
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = PiTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            depths=self.depths,
            base_dims=self.base_dims,
            heads=self.heads,
            stride=self.stride,
            conv_pooling_stride=self.conv_pooling_stride,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

        return config, pixel_values, labels

    def create_and_check_model(self, config, pixel_values, labels):
        model = PiTModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = to_2tuple(self.image_size)
        patch_size = to_2tuple(self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = PiTForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

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
class PiTModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as PiT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (
            PiTModel,
            PiTForImageClassification,
        )
        if is_torch_available()
        else ()
    )

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = PiTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PiTConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_inputs_embeds(self):
        # PiT does not use inputs_embeds
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (torch.nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, torch.nn.Linear))

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

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        def get_attn_shapes():
            # get intital seq_len
            # in PiT, the seq_len equals the number of patches + 1 (we add 1 for the [CLS] token)#
            seq_len = ((config.image_size + 2 - config.patch_size) // config.stride + 1) ** 2

            attention_shapes = []
            for num_heads in config.heads:
                attention_shapes.append([num_heads, seq_len + 1, seq_len + 1])
                seq_len = math.ceil(int(seq_len ** 0.5) / config.conv_pooling_stride)
                seq_len = seq_len ** 2
            return attention_shapes

        attention_shapes = get_attn_shapes()

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

            self.assertEqual(len(attentions), len(self.model_tester.depths))
            for i, attn in enumerate(attentions):
                self.assertEqual(len(attn), self.model_tester.depths[i])

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions

            self.assertEqual(len(attentions), len(self.model_tester.depths))
            for i, attn in enumerate(attentions):
                self.assertEqual(len(attn), self.model_tester.depths[i])
                self.assertListEqual(list(attn[0].shape[-3:]), attention_shapes[i])

            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            added_hidden_states = 3
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), len(self.model_tester.depths))
            for i, attn in enumerate(self_attentions):
                self.assertEqual(len(attn), self.model_tester.depths[i])
                self.assertListEqual(list(attn[0].shape[-3:]), attention_shapes[i])

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_blocks = len(config.base_dims)
            self.assertEqual(len(hidden_states), expected_num_blocks)

            def get_hidden_shapes():
                # get intital seq_len
                # in PiT, the seq_len equals the number of patches + 1 (we add 1 for the [CLS] token)#
                seq_len = ((config.image_size + 2 - config.patch_size) // config.stride + 1) ** 2

                hidden_shapes = []
                for i, (num_heads, base_dim) in enumerate(zip(config.heads, config.base_dims)):
                    hidden_shapes.append([seq_len + 1, num_heads * base_dim])
                    seq_len = math.ceil(int(seq_len ** 0.5) / config.conv_pooling_stride)
                    seq_len = seq_len ** 2
                return hidden_shapes

            excped_hidden_shapes = get_hidden_shapes()

            for i, states in enumerate(hidden_states):
                self.assertListEqual(list(states[0].shape[-2:]), excped_hidden_shapes[i])

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in PIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = PiTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/cats.png")
    return image


@require_vision
class PiTModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return PiTFeatureExtractor.from_pretrained("google/pit-base-patch16-224") if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = PiTForImageClassification.from_pretrained("google/pit-base-patch16-224").to(torch_device)

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-0.2744, 0.8215, -0.0836]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))
