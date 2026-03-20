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

import requests
from parameterized import parameterized

from transformers import (
    UVDocConfig,
    UVDocImageProcessor,
    UVDocModel,
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
        is_training=False,
        bridge_in_channels=32,
        kernel_size=5,
        stage_layer_num=(3, 4, 6),
        resnet_head=((3, 8), (8, 8)),
        resnet_stage_downsample=(
            (False, False, False),
            (True, False, False, False),
            (True, False, False, False, False, False),
        ),
        resnet_down=((8, 8), (8, 16), (16, 32)),
        bridge_connector=(32, 32),
        out_point_positions2D=((32, 8), (8, 2)),
        dilation_values=(
            (1,),
            (2,),
            (5,),
            (8, 3, 2),
            (12, 7, 4),
            (18, 12, 6),
        ),
        padding_mode="reflect",
        hidden_act="prelu",
    ):
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.bridge_in_channels = bridge_in_channels
        self.kernel_size = kernel_size
        self.stage_layer_num = stage_layer_num
        self.resnet_head = resnet_head
        self.resnet_stage_downsample = resnet_stage_downsample
        self.resnet_down = resnet_down
        self.bridge_connector = bridge_connector
        self.out_point_positions2D = out_point_positions2D
        self.dilation_values = dilation_values
        self.padding_mode = padding_mode
        self.num_hidden_layers = len(dilation_values)
        self.padding_mode = padding_mode
        self.hidden_act = hidden_act
        # For test_hidden_states_output: UVDoc outputs spatial hidden states (B, C, H, W)
        # with shape[-2:] = (8, 8) for image_size=128
        self.seq_length = 8
        self.hidden_size = 8

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self) -> UVDocConfig:
        return UVDocConfig(
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            hidden_act=self.hidden_act,
            bridge_in_channels=self.bridge_in_channels,
            stage_layer_num=list(self.stage_layer_num),
            resnet_head=[list(h) for h in self.resnet_head],
            resnet_stage_downsample=[list(down) for down in self.resnet_stage_downsample],
            resnet_down=[list(rd) for rd in self.resnet_down],
            bridge_connector=list(self.bridge_connector),
            out_point_positions2D=[list(op) for op in self.out_point_positions2D],
            dilation_values=[list(d) for d in self.dilation_values],
        )

    def create_and_check_uvdoc_document_rectification(self, config, pixel_values):
        model = UVDocModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, 2, 8, 8))


@require_torch
class UVDocModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (UVDocModel,) if is_torch_available() else ()

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
        self.config_tester.run_common_tests()

    def test_uvdoc_document_rectification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_uvdoc_document_rectification(*config_and_inputs)

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

    @unittest.skip(reason="UVDoc does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="UVDoc does not support training")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Large number of hidden layers but small spatial dimensions")
    def test_num_layers_is_small(self):
        pass


@require_torch
@require_vision
@slow
class UVDocModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "/workspace/model_weight_torch/UVDoc"
        self.model = UVDocModel.from_pretrained(model_path).to(torch_device)
        self.image_processor = UVDocImageProcessor() if is_vision_available() else None
        self.image = Image.open(
            requests.get(
                "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/doc_test.jpg", stream=True
            ).raw
        )

    def test_inference_document_rectification(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)
        bs = inputs["pixel_values"].shape[0]

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.image_processor.post_process_document_rectification(
            outputs.last_hidden_state, inputs["original_images"]
        )

        expected_shape_logits = torch.Size((bs, 2, 45, 31))
        expected_logits = torch.tensor(
            [
                [-0.7635, -0.7251, -0.6819],
                [-0.7643, -0.7250, -0.6814],
                [-0.7647, -0.7252, -0.6816],
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
