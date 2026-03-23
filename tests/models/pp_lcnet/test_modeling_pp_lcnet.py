# coding = utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PP-LCNet model."""

import inspect
import unittest

import requests
from parameterized import parameterized

from transformers import (
    PPLCNetBackbone,
    PPLCNetConfig,
    PPLCNetForImageClassification,
    PPLCNetImageProcessor,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_vision,
    slow,
    torch_device,
)

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class PPLCNetModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        image_size=128,
        num_channels=3,
        num_stages=5,
        is_training=False,
        scale=1.0,
        reduction=4,
        dropout_prob=0.2,
        class_expand=1280,
        use_last_convolution=True,
        hidden_act="hardswish",
        num_labels=4,
        out_features=["stage2", "stage3", "stage4"],
        out_indices=[2, 3, 4],
        stem_channels=16,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.num_stages = num_stages
        self.scale = scale
        self.reduction = reduction
        self.dropout_prob = dropout_prob
        self.class_expand = class_expand
        self.use_last_convolution = use_last_convolution
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.out_features = out_features
        self.out_indices = out_indices
        self.stem_channels = stem_channels
        self.block_configs = [
            [[3, 16, 32, 1, False]],
            [[3, 32, 32, 2, False], [3, 32, 32, 1, False]],
            [[3, 32, 32, 2, False], [3, 32, 32, 1, False]],
            [
                [3, 32, 32, 2, False],
                [5, 32, 32, 1, False],
                [5, 32, 32, 1, False],
                [5, 32, 32, 1, False],
                [5, 32, 32, 1, False],
                [5, 32, 32, 1, False],
            ],
            [[5, 32, 32, 2, True], [5, 32, 32, 1, True]],
        ]

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self) -> PPLCNetConfig:
        id2label = {"0": "0", "1": "90", "2": "180", "3": "270"}
        config = PPLCNetConfig(
            scale=self.scale,
            reduction=self.reduction,
            dropout_prob=self.dropout_prob,
            class_expand=self.class_expand,
            use_last_conv=self.use_last_convolution,
            hidden_act=self.hidden_act,
            id2label=id2label,
            out_features=self.out_features,
            out_indices=self.out_indices,
            block_configs=self.block_configs,
        )

        return config


@require_torch
class PPLCNetBackboneTest(BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (PPLCNetBackbone,) if is_torch_available() else ()
    has_attentions = False
    config_class = PPLCNetConfig

    def setUp(self):
        self.model_tester = PPLCNetModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=PPLCNetConfig,
            has_text_modality=False,
            common_properties=[],
        )


@require_torch
class PPLCNetModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PPLCNetForImageClassification,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-classification": PPLCNetForImageClassification} if is_torch_available() else {}

    has_attentions = False
    test_inputs_embeds = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = PPLCNetModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=PPLCNetConfig,
            has_text_modality=False,
            common_properties=[],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def create_and_check_pp_lcnet_image_classification(self, config, pixel_values):
        model = PPLCNetForImageClassification(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values)

        self.assertEqual(result.last_hidden_state.shape, (self.model_tester.batch_size, model.config.num_labels))

    def test_pp_lcnet_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.create_and_check_pp_lcnet_image_classification(*config_and_inputs)

    @unittest.skip(reason="PPLCNet does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="PPLCNet does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="PPLCNet does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="PPLCNet does not support attention")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="PPLCNet does not support train")
    def test_problem_types(self):
        pass

    @unittest.skip(reason="PPLCNet does not support model parallelism")
    def test_model_parallelism(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    @parameterized.expand(["float32", "float16", "bfloat16"])
    @require_torch_accelerator
    @slow
    def test_inference_with_different_dtypes(self, dtype_str):
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype_str]

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device).to(dtype)
            model.eval()
            for key, tensor in inputs_dict.items():
                if tensor.dtype == torch.float32:
                    inputs_dict[key] = tensor.to(dtype)
            with torch.no_grad():
                _ = model(**self._prepare_for_class(inputs_dict, model_class))

    # PPLCNet have no seq_length
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            expected_num_stages = self.model_tester.num_stages
            scale = self.model_tester.scale

            self.assertEqual(len(hidden_states), expected_num_stages + 1)

            self.assertEqual(hidden_states[0].shape[1], self.model_tester.stem_channels)
            for i in range(expected_num_stages):
                self.assertEqual(
                    hidden_states[i + 1].shape[1],
                    self.model_tester.block_configs[i][-1][2] * scale,
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)


@require_torch
@require_vision
@slow
class PPLCNetModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors"
        self.model = PPLCNetForImageClassification.from_pretrained(model_path).to(torch_device)
        self.image_processor = PPLCNetImageProcessor.from_pretrained(model_path) if is_vision_available() else None
        url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)

    def test_inference_image_classification_head(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        expected_shape_logits = torch.Size((1, 4))
        expected_logits = torch.tensor([[-0.3655, -1.0573, 2.4883, -1.0640]]).to(torch_device)

        self.assertEqual(outputs.last_hidden_state.shape, expected_shape_logits)
        torch.testing.assert_close(outputs.last_hidden_state, expected_logits, rtol=2e-2, atol=2e-2)

        expected_labels = torch.tensor([2]).to(torch_device)
        predicted_label = outputs.last_hidden_state.argmax(-1).item()

        self.assertEqual(predicted_label, expected_labels)
