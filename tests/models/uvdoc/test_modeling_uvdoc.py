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
"""Testing suite for the UVDoc model."""

import inspect
import unittest

from parameterized import parameterized

from transformers import (
    UVDocConfig,
    UVDocForDocumentRectification,
    UVDocImageProcessorFast,
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


class UVDocModelTester:
    def __init__(
        self,
        batch_size=3,
        image_size=128,
        num_channels=3,
        num_stages=6,
        is_training=False,
    ):
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.num_stages = num_stages

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self) -> UVDocConfig:
        dilation_values = {
            "bridge_block_1": 1,
            "bridge_block_2": 2,
            "bridge_block_3": 5,
            "bridge_block_4": [8, 3, 2],
            "bridge_block_5": [12, 7, 4],
            "bridge_block_6": [18, 12, 6],
        }

        self.dilation_values = dilation_values

        config = UVDocConfig(
            num_filter=32,
            in_channels=3,
            kernel_size=5,
            block_stride_values=[1, 2, 2, 2],
            feature_map_multipliers=[1, 1, 1, 2, 2],
            block_counts_per_stage=[1, 1, 1, 1],
            dilation_values=dilation_values,
            padding_mode="reflect",
            upsample_size=[712, 488],
            upsample_mode="bilinear",
        )

        return config

    def create_and_check_uvdoc_document_rectification(self, config, pixel_values):
        model = UVDocForDocumentRectification(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.num_channels, self.image_size, self.image_size)
        )


@require_torch
class UVDocModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (UVDocForDocumentRectification,) if is_torch_available() else ()

    has_attentions = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = UVDocModelTester(
            batch_size=3,
            is_training=False,
            image_size=128,
        )
        self.model_tester.parent = self
        self.config_tester = ConfigTester(
            self,
            config_class=UVDocConfig,
            has_text_modality=False,
            common_properties=[],
        )

    def test_config(self):
        # Skip create_and_test_config_with_num_labels: UVDoc has fixed single class (image)
        self.config_tester.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_from_and_save_pretrained_subfolder()
        self.config_tester.create_and_test_config_from_and_save_pretrained_composite()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()
        self.config_tester.create_and_test_config_from_pretrained_custom_kwargs()

    def test_uvdoc_document_rectification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_uvdoc_document_rectification(*config_and_inputs)

    @unittest.skip(reason="UVDoc does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="UVDoc does not support hidden_states")
    def test_hidden_states_output(self):
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

    @unittest.skip(reason="UVDoc does not support training")
    def test_retain_grad_hidden_states_attentions(self):
        pass


@require_torch
@require_vision
@slow
class UVDocModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "/workspace/model_weight_torch/UVDoc"
        self.model = UVDocForDocumentRectification.from_pretrained(model_path).to(torch_device)
        self.image_processor = UVDocImageProcessorFast.from_pretrained(model_path) if is_vision_available() else None
        path = "/workspace/PaddleX/paddlex/inference/models/image_unwarping/modeling/doc_test.jpg"
        self.image = Image.open(path)

    def test_inference_document_rectification(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)
        bs = inputs["pixel_values"].shape[0]

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.image_processor.post_process_document_rectification(outputs.last_hidden_state)

        expected_shape_logits = torch.Size((bs, 3, 708, 1100))
        expected_logits = torch.tensor(
            [
                [0.5279, 0.5213, 0.5157],
                [0.5331, 0.5261, 0.5187],
                [0.5330, 0.5291, 0.5239],
            ],
            device=torch_device,
        )

        self.assertEqual(outputs.last_hidden_state.shape, expected_shape_logits)
        torch.testing.assert_close(outputs.last_hidden_state[0, 0, :3, :3], expected_logits, rtol=2e-4, atol=2e-4)

        expected_images = torch.tensor(
            [
                [131, 130, 128],
                [131, 129, 127],
                [130, 129, 127],
            ],
            device=torch_device,
            dtype=torch.uint8,
        )
        torch.testing.assert_close(results[0]["images"][:3, :3, 0], expected_images, rtol=2e-4, atol=2e-4)
        self.assertEqual(results[0]["labels"].shape, (1,))
        self.assertTrue((results[0]["labels"] == 0).all())
