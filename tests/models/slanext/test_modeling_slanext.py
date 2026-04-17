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
"""Testing suite for the SLANeXt model."""

import copy
import inspect
import tempfile
import unittest

import requests
from parameterized import parameterized

from transformers import (
    AutoImageProcessor,
    AutoModelForTableRecognition,
    SLANeXtConfig,
    SLANeXtForTableRecognition,
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

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class SLANeXtModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=512,
        num_channels=3,
        is_training=False,
        vision_config=None,
    ):
        self.parent = parent
        if vision_config is None:
            vision_config = {
                "hidden_size": 1,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "global_attn_indexes": [1, 1, 1, 1],
                "mlp_dim": 4,
            }
        self.vision_config = vision_config
        self.num_hidden_layers = vision_config["num_hidden_layers"]
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self) -> SLANeXtConfig:
        config = SLANeXtConfig(
            vision_config=self.vision_config,
            out_channels=1,
            hidden_size=1,
            max_text_length=1,
        )

        return config


@require_torch
class SLANeXtModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (SLANeXtForTableRecognition,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-feature-extraction": SLANeXtForTableRecognition} if is_torch_available() else {}

    test_resize_embeddings = False
    test_torch_exportable = False

    def setUp(self):
        self.model_tester = SLANeXtModelTester(
            self,
            batch_size=1,
            image_size=512,
        )
        self.config_tester = ConfigTester(
            self,
            config_class=SLANeXtConfig,
            has_text_modality=False,
            common_properties=[],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SLANeXt can at minimum only have roughly 1.7M parameters")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="SLANeXt does not use inputs_embeds")
    def test_enable_input_require_grads(self):
        pass

    @unittest.skip(reason="SLANeXt does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="SLANeXt does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="SLANeXt does not support input and output embeddings")
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

    def test_hidden_states_output(self):
        """
        Overriden because vision hidden states behave in a unique way

        NOTE: We ignore the head hidden states as they can be dynamic
        """

        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(copy.deepcopy(config))
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = self.model_tester.num_hidden_layers + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            patched_image_size = config.vision_config.image_size // config.vision_config.patch_size
            self.assertListEqual(
                list(hidden_states[0].shape[-3:]),
                [patched_image_size, patched_image_size, config.vision_config.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            self._set_subconfig_attributes(config, "output_hidden_states", True)
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_attention_outputs(self):
        """
        Overriden because vision attentions behave in a unique way

        NOTE: We ignore the head attentions as they can be dynamic
        """
        if not self.has_attentions:
            self.skipTest(reason="Model does not output attentions")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"

        # Window partitioned lengt based on the window size
        seq_len = config.vision_config.window_size * config.vision_config.window_size
        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            self._set_subconfig_attributes(config, "output_attentions", True)
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            # Ignoring batch size for now as it is dynamically changed during window partitioning
            self.assertListEqual(
                list(attentions[0].shape[-2:]),
                [seq_len, seq_len],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            # hidden states are also within the head
            self.assertEqual(out_len + 2, len(outputs))

            self_attentions = outputs.attentions
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            # Ignoring batch size for now as it is dynamically changed during window partitioning
            self.assertListEqual(
                list(attentions[0].shape[-2:]),
                [seq_len, seq_len],
            )

    @parameterized.expand(["float32", "float16", "bfloa16"])
    @require_torch_accelerator
    @slow
    def test_inference_with_different_dtypes(self, dtype_str):
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloa16": torch.bfloat16,
        }[dtype_str]

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device).to(dtype)

            # Save and reload to make use of keep in fp32 modules
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model.from_pretrained(tmpdirname).to(torch_device)
                model.eval()

                for key, tensor in inputs_dict.items():
                    if tensor.dtype == torch.float32:
                        inputs_dict[key] = tensor.to(dtype)
                with torch.no_grad():
                    _ = model(**self._prepare_for_class(inputs_dict, model_class))


@require_torch
@require_vision
@slow
class SLANeXtModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/SLANeXt_wired_safetensors"
        self.model = AutoModelForTableRecognition.from_pretrained(model_path, dtype=torch.float32).to(torch_device)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)

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
