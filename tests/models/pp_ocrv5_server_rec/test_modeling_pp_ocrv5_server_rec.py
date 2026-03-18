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
"""Testing suite for the PPOCRV5ServerRec model."""

import inspect
import unittest

import requests
from parameterized import parameterized

from transformers import (
    AutoImageProcessor,
    PPOCRV5ServerRecConfig,
    PPOCRV5ServerRecForTextRecognition,
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


class PPOCRV5ServerRecModelTester:
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

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])
        config = self.get_config()

        return config, pixel_values

    def get_config(self) -> PPOCRV5ServerRecConfig:
        backbone_config = {
            "model_type": "hgnet_v2",
            "arch": "L",
            "return_idx": [0, 1, 2, 3],
            "hidden_sizes": [16, 16, 16, 16],
            "stem_channels": [3, 16, 16],
            "stage_in_channels": [16, 16, 16, 16],
            "stage_mid_channels": [16, 16, 16, 16],
            "stage_out_channels": [16, 16, 16, 16],
            "freeze_stem_only": True,
            "freeze_at": 0,
            "freeze_norm": True,
            "lr_mult_list": [1.0, 1.0, 1.0, 1.0, 1.0],
            "out_features": ["stage1", "stage2", "stage3", "stage4"],
            "stage_downsample": [True, True, True, True],
            "stem_strides": [2, 1, 1, 1, 1],
            "stage_downsample_strides": [[2, 1], [1, 2], [2, 1], [2, 1]],
        }

        config = PPOCRV5ServerRecConfig(
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
class PPOCRV5ServerRecModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PPOCRV5ServerRecForTextRecognition,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-feature-extraction": PPOCRV5ServerRecForTextRecognition} if is_torch_available() else {}
    )

    has_attentions = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = PPOCRV5ServerRecModelTester()
        self.model_tester.parent = self
        self.config_tester = ConfigTester(
            self,
            config_class=PPOCRV5ServerRecConfig,
            has_text_modality=False,
            common_properties=[],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip("PPOCRV5ServerRec does not has no attribute `hf_device_map`")
    def test_cpu_offload(self):
        pass

    @unittest.skip("PPOCRV5ServerRec does not has no attribute `hf_device_map`")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip("PPOCRV5ServerRec does not has no attribute `hf_device_map`")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip("PPOCRV5ServerRec does not has no attribute `hf_device_map`")
    def test_model_parallelism(self):
        pass

    @unittest.skip("PPOCRV5ServerRec does not has no function `get_input_embeddings`")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerRec does not support attention")
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
        # PP-OCRV5 uses HGNetV2 backbone: hidden_states = (embedding, stage1, ..., stageN) = num_stages + 1
        config = self.model_tester.get_config()
        num_expected_hidden_states = len(config.backbone_config.depths) + 1

        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), num_expected_hidden_states)

            # First hidden state (embedding output) is 2x downsampled
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.image_size[0] // 2, self.model_tester.image_size[1] // 2],
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
class PPOCRV5ServerRecModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-OCRv5_server_rec_safetensors"
        self.model = PPOCRV5ServerRecForTextRecognition.from_pretrained(model_path).to(torch_device)
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
        expected_score = 0.9837473630905151

        self.assertEqual(results[0]["text"], expected_text)
        torch.testing.assert_close(results[0]["score"], expected_score, rtol=2e-2, atol=2e-2)
