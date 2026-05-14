# coding = utf-8
# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the SLANet model."""

import inspect
import unittest

from transformers import (
    AutoImageProcessor,
    AutoModelForTableRecognition,
    SLANetConfig,
    SLANetForTableRecognition,
    is_torch_available,
)
from transformers.image_utils import load_image
from transformers.testing_utils import (
    require_torch,
    require_vision,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch


class SLANetModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=488,
        num_channels=3,
        post_conv_out_channels=16,
        out_channels=1,
        hidden_size=16,
        max_text_length=1,
        num_stages=5,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.post_conv_out_channels = post_conv_out_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.max_text_length = max_text_length
        self.num_stages = num_stages
        self.is_training = is_training

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self) -> SLANetConfig:
        backbone_config = {
            "model_type": "pp_lcnet",
            "scale": 1,
            "out_features": ["stage2", "stage3", "stage4", "stage5"],
            "out_indices": [2, 3, 4, 5],
            "block_configs": [
                [[3, 16, 16, 1, False]],
                [[3, 16, 16, 2, False], [3, 16, 16, 1, False]],
                [[3, 16, 16, 2, False], [3, 16, 16, 1, False]],
                [
                    [3, 16, 16, 2, False],
                    [5, 16, 16, 1, False],
                    [5, 16, 16, 1, False],
                    [5, 16, 16, 1, False],
                    [5, 16, 16, 1, False],
                    [5, 16, 16, 1, False],
                ],
                [[5, 16, 16, 2, True], [5, 16, 16, 1, True]],
            ],
        }
        config = SLANetConfig(
            backbone_config=backbone_config,
            out_channels=self.out_channels,
            hidden_size=self.hidden_size,
            max_text_length=self.max_text_length,
            post_conv_out_channels=self.post_conv_out_channels,
        )

        return config


@require_torch
class SLANetModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (SLANetForTableRecognition,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-feature-extraction": SLANetForTableRecognition} if is_torch_available() else {}

    has_attentions = False
    test_resize_embeddings = False
    test_torch_exportable = False

    def setUp(self):
        self.model_tester = SLANetModelTester(
            self,
            batch_size=1,
            image_size=488,
        )
        self.config_tester = ConfigTester(
            self,
            config_class=SLANetConfig,
            has_text_modality=False,
            common_properties=[],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SLANet does not use inputs_embeds")
    def test_enable_input_require_grads(self):
        pass

    @unittest.skip(reason="SLANet does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="SLANet does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="SLANet does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    # SLANet have no seq_length
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            expected_num_stages = self.model_tester.num_stages

            self.assertEqual(len(hidden_states), expected_num_stages + 1)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict.copy(), config, model_class)

            # Check that output_hidden_states also works via config (including backbone subconfig)
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            if config.backbone_config is not None:
                config.backbone_config.output_hidden_states = True
            check_hidden_states_output(inputs_dict.copy(), config, model_class)


@require_torch
@require_vision
@slow
class SLANetModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/SLANet_plus_safetensors"
        self.model = AutoModelForTableRecognition.from_pretrained(model_path, dtype=torch.float32).to(torch_device)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        img_url = url_to_local_path(
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg"
        )
        self.image = load_image(img_url)

    def test_inference_table_recognition_head(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        pred_table_structure = self.image_processor.post_process_table_recognition(outputs)["structure"]
        expected_table_structure = [
            "<html>",
            "<body>",
            "<table>",
            "<tr>",
            "<td",
            ' colspan="4"',
            ">",
            "</td>",
            "</tr>",
            "<tr>",
            "<td></td>",
            "<td></td>",
            "<td></td>",
            "<td></td>",
            "</tr>",
            "<tr>",
            "<td></td>",
            "<td></td>",
            "<td></td>",
            "<td></td>",
            "</tr>",
            "<tr>",
            "<td></td>",
            "<td></td>",
            "<td></td>",
            "<td></td>",
            "</tr>",
            "</table>",
            "</body>",
            "</html>",
        ]

        self.assertEqual(pred_table_structure, expected_table_structure)
