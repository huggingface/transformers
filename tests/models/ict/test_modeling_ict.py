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
""" Testing suite for the PyTorch ICT model. """


import inspect
import unittest

from transformers import IctConfig
from transformers.testing_utils import (
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    torch.manual_seed(3)

    from transformers import IctModel
    from transformers.models.ict.modeling_ict import ICT_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import IctImageProcessor


class IctModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        vocab_size=512,
        hidden_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_residual_blocks=8,
        intermediate_size=37,
        activation_function="gelu",
        embedding_dropout_prob=0.0,
        residual_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=32,
        num_channels=3,
        qkv_bias=False,
        temperature=1.0,
        top_k=50,
        gan_loss_function="nsgan",
        output_image_size=256,
        scope=None,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_residual_blocks = num_residual_blocks
        self.intermediate_size = intermediate_size
        self.activation_function = activation_function
        self.embedding_dropout_prob = embedding_dropout_prob
        self.residual_dropout_prob = residual_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.temperature = temperature
        self.top_k = top_k
        self.gan_loss_function = gan_loss_function
        self.output_image_size = output_image_size

        self.seq_length = image_size * image_size
        self.scope = scope
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        pixel_values = ids_tensor([self.batch_size, self.image_size * self.image_size], self.vocab_size)
        bool_masked_pos = torch.randint(low=0, high=2, size=(1, pixel_values.shape[1])).bool()

        clusters = torch.rand(512, 3)

        config = self.get_config()

        return config, pixel_values, bool_masked_pos, clusters

    def get_config(self):
        return IctConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_residual_blocks=self.num_residual_blocks,
            intermediate_size=self.intermediate_size,
            activation_function=self.activation_function,
            embedding_dropout_prob=self.embedding_dropout_prob,
            residual_dropout_prob=self.residual_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            image_size=self.image_size,
            num_channels=self.num_channels,
            qkv_bias=self.qkv_bias,
            temperature=self.temperature,
            top_k=self.top_k,
            gan_loss_function=self.gan_loss_function,
            output_image_size=self.output_image_size,
        )

    def create_and_check_model(self, config, pixel_values, bool_masked_pos, clusters):
        model = IctModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, bool_masked_pos, clusters)
        self.parent.assertEqual(
            result.reconstruction.shape,
            (self.batch_size, self.num_channels, self.output_image_size, self.output_image_size),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, pixel_values, bool_masked_pos, clusters) = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "bool_masked_pos": bool_masked_pos,
            "clusters": clusters,
        }
        return config, inputs_dict


@require_torch
class IctModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ICT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (IctModel,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": IctModel} if is_torch_available() else {}
    fx_compatible = False

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = IctModelTester(self)
        self.config_tester = ConfigTester(self, config_class=IctConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="ICT does not use inputs_embeds")
    def test_inputs_embeds(self):
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

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in ICT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = IctModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class IctModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return IctImageProcessor.from_pretrained("sheonhan/ict-imagenet-256") if is_vision_available() else None

    # @slow
    def test_inference_masked_image_modeling(self):
        model = IctModel.from_pretrained("sheonhan/ict-imagenet-256").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")

        pixel_values = inputs.pixel_values
        image_size = pixel_values.shape[1]

        bool_masked_pos = torch.randint(low=0, high=2, size=(1, image_size)).bool()
        clusters = inputs.clusters

        # forward pass
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                bool_masked_pos=bool_masked_pos,
                clusters=clusters,
            )

        # verify the logits
        expected_shape = torch.Size((1, 3, 256, 256))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[2.3445, 2.6889, 2.7313], [1.0530, 1.2416, 0.5699], [0.2205, 0.7749, 0.3953]]
        ).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3, :3], expected_slice, atol=1e-4))
