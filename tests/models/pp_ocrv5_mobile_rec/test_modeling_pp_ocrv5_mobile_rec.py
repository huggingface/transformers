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
"""Testing suite for the PPOCRVMobileRec model."""

import inspect
import unittest

import requests
from parameterized import parameterized

from transformers import (
    AutoImageProcessor,
    PPOCRV5MobileRecConfig,
    PPOCRV5MobileRecForTextRecognition,
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


class PPOCRV5MobileRecModelTester:
    def __init__(
        self,
        batch_size=3,
        image_size=[48, 320],
        num_channels=3,
        is_training=False,
        hidden_act="silu",
        hidden_size=10,
        mlp_ratio=2.0,
        depth=2,
        head_out_channels=18385,
        conv_kernel_size=[1, 3],
        qkv_bias=True,
        num_attention_heads=2,
        num_stages=5,
        attention_dropout=0.0,
    ):
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.depth = depth
        self.head_out_channels = head_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.qkv_bias = qkv_bias
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.num_stages = num_stages

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])
        config = self.get_config()

        return config, pixel_values

    def get_config(self) -> PPOCRV5MobileRecConfig:
        backbone_config = {
            "model_type": "pp_lcnet_v3",
            "scale": 0.95,
            "out_features": ["stage2", "stage3", "stage4", "stage5"],
            "out_indices": [2, 3, 4, 5],
            "divisor": 16,
            "block_configs": [
                [[3, 16, 32, 1, False]],
                [[3, 32, 32, 1, False], [3, 32, 32, 1, False]],
                [[3, 32, 32, [2, 1], False], [3, 32, 32, 1, False]],
                [
                    [3, 32, 32, [1, 2], False],
                    [5, 32, 32, 1, False],
                    [5, 32, 32, 1, False],
                    [5, 32, 32, 1, False],
                    [5, 32, 32, 1, False],
                ],
                [[5, 32, 32, [2, 1], True], [5, 32, 32, 1, True], [5, 32, 32, [2, 1], False], [5, 32, 32, 1, False]],
            ],
        }

        config = PPOCRV5MobileRecConfig(
            backbone_config=backbone_config,
            hidden_act=self.hidden_act,
            hidden_size=self.hidden_size,
            mlp_ratio=self.mlp_ratio,
            depth=self.depth,
            head_out_channels=self.head_out_channels,
            conv_kernel_size=self.conv_kernel_size,
            qkv_bias=self.qkv_bias,
            num_attention_heads=self.num_attention_heads,
            attention_dropout=self.attention_dropout,
        )

        return config


@require_torch
class PPOCRV5MobileRecModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PPOCRV5MobileRecForTextRecognition,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-feature-extraction": PPOCRV5MobileRecForTextRecognition} if is_torch_available() else {}
    )

    has_attentions = False
    test_resize_embeddings = False
    model_split_percents = [0.5, 0.8]

    def setUp(self):
        self.model_tester = PPOCRV5MobileRecModelTester()
        self.model_tester.parent = self
        self.config_tester = ConfigTester(
            self,
            config_class=PPOCRV5MobileRecConfig,
            has_text_modality=False,
            common_properties=[],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip("PPOCRV5MobileRec does not has no attribute `hf_device_map`")
    def test_model_parallelism(self):
        pass

    @unittest.skip("PPOCRV5MobileRec does not has no function `get_input_embeddings`")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="PPOCRV5MobileRec does not support attention")
    def test_retain_grad_hidden_states_attentions(self):
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

            head_hidden_states = outputs.head_hidden_states
            self.assertIsNotNone(head_hidden_states)
            self.assertEqual(len(head_hidden_states), self.model_tester.depth + 1)

            self.assertListEqual(
                list(head_hidden_states[0].shape[-2:]),
                [self.model_tester.hidden_size * self.model_tester.mlp_ratio * 2, self.model_tester.hidden_size],
            )

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
class PPOCRV5MobileRecModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors"
        self.model = PPOCRV5MobileRecForTextRecognition.from_pretrained(model_path).to(torch_device)
        self.image_processor = (
            AutoImageProcessor.from_pretrained(model_path, return_tensors="pt") if is_vision_available() else None
        )
        url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png"
        self.image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    def test_inference_text_recognition_head(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)
        outputs = self.model(**inputs)

        results = self.image_processor.post_process_text_recognition(outputs)
        expected_text = "绿洲仕格维花园公寓"
        expected_score = 0.9909055233001709

        self.assertEqual(results[0]["text"], expected_text)
        torch.testing.assert_close(results[0]["score"], expected_score, rtol=2e-2, atol=2e-2)
