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
""" Testing suite for the PyTorch CCT model. """


import inspect
import unittest
from math import floor
from typing import Dict, List, Tuple
import torch.nn as nn

from transformers import CctConfig
from transformers.file_utils import cached_property, is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import CctForImageClassification, CctModel
    from transformers.models.cct.modeling_cct import CCT_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import ConvNextFeatureExtractor


class CctConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "embed_dim"))
        self.parent.assertTrue(hasattr(config, "num_heads"))


class CctModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        in_channels=3,
        out_channels=[64, 384],
        conv_kernel_size=7,
        conv_stride=2,
        conv_padding=3,
        conv_bias=False,
        pool_kernel_size=3,
        pool_stride=2,
        pool_padding=1,
        num_conv_layers=2,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=3,
        attention_drop_rate=0.1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        qkv_bias=[False, False, False],
        qkv_projection_method=["avg", "avg", "avg"],
        num_transformer_layers=14,
        pos_emb_type='learnable',
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        is_training=True,
        use_labels=True,
        num_labels=2,  # Check
    ):
        
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.conv_bias = conv_bias
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_kernel_size
        self.pool_padding = pool_padding
        self.num_conv_layers = num_conv_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attention_drop_rate = attention_drop_rate
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.qkv_projection_method = qkv_projection_method
        self.num_transformer_layers = num_transformer_layers
        self.pos_emb_type = pos_emb_type
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_training = is_training
        self.use_labels = use_labels
        self.num_labels = num_labels
 
    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.in_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()
        return config, pixel_values, labels

    def get_config(self):
        return CctConfig(
            img_size=self.image_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            conv_kernel_size=self.conv_kernel_size,
            conv_stride=self.conv_stride,
            conv_padding=self.conv_padding,
            conv_bias=self.conv_bias,
            pool_kernel_size=self.pool_kernel_size,
            pool_stride=self.pool_stride,
            pool_padding=self.pool_padding,
            num_conv_layers=self.num_conv_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            attention_drop_rate=self.attention_drop_rate,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            qkv_bias=self.qkv_bias,
            qkv_projection_method=self.qkv_projection_method,
            num_transformer_layers=self.num_transformer_layers,
            pos_emb_type=self.pos_emb_type,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
        )

    def create_and_check_hidden_state(self, post_pool):
        sz = self.image_size
        for _ in range(self.num_conv_layers):
            sz = floor((sz + 2*self.conv_padding - self.conv_kernel_size)/self.conv_stride) + 1
            sz = floor((sz + 2*self.pool_padding - self.pool_kernel_size)/self.pool_stride) + 1

        if post_pool == False:
            expected_size = (self.batch_size, sz*sz, self.embed_dim)
        else:
            expected_size = (self.batch_size, self.embed_dim)

        return expected_size

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.num_labels
        model = CctForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class CctModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py
    """

    all_model_classes = (CctModel, CctForImageClassification) if is_torch_available() else ()

    pipeline_model_mapping = (
        {"feature-extraction": CctModel, "image-classification": CctForImageClassification}
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_resize_position_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = CctModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CctConfig, has_text_modality=False)

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

    @unittest.skip(reason="Cct does not output attentions")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="Cct does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Cct does not support input and output embeddings")
    def test_model_common_attributes(self):
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

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            for key, value in model.named_parameters():
                if 'weight' in key and value.requires_grad:
                    if 'norm' in key:
                        bound = 1.0
                    else:
                        bound = 0.0
                    self.assertTrue(
                        bound-0.01 <= ((value.data.mean() * 1e9).round() / 1e9).item() <= bound+0.01,
                        msg=f"Parameter {key} of model {model_class} seems not properly initialized with value {value.data.mean()}") 
                        
                if 'bias' in key and value.requires_grad: 
                    self.assertEqual(
                        value.data.mean().item(),
                        0.0,
                        msg=f"Parameter {key} of model {model_class} seems not properly initialized with value {value.data.mean()}",
                    )
                
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = self.model_tester.num_transformer_layers + 2 ## output from conv tokenizer + num_transformer_layers + output of seq_pool
            self.assertEqual(len(hidden_states), expected_num_layers)

            ## check input hidden state
            expected_size = self.model_tester.create_and_check_hidden_state(post_pool=False)
            result = outputs.hidden_states[0]
            self.assertEqual(result.shape, expected_size)

            ## check final hidden state
            expected_size = self.model_tester.create_and_check_hidden_state(post_pool=True)
            result = outputs.hidden_states[-1]
            self.assertEqual(result.shape, expected_size)

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
        for model_name in CCT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CctModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class CctModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return ConvNextFeatureExtractor.from_pretrained(CCT_PRETRAINED_MODEL_ARCHIVE_LIST[0])

    @slow
    def test_inference_image_classification_head(self):
        model = CctForImageClassification.from_pretrained(CCT_PRETRAINED_MODEL_ARCHIVE_LIST[0]).to(torch_device)

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([0.1484, 1.1873, 0.3872, 0.1801, 0.5467]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :5], expected_slice, atol=1e-4))
