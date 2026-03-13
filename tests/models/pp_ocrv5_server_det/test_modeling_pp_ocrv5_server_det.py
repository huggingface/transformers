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
"""Testing suite for the PP-OCRV5ServerDet model."""

import inspect
import unittest

import requests
from parameterized import parameterized

from transformers import (
    PPOCRV5ServerDetConfig,
    PPOCRV5ServerDetForObjectDetection,
    PPOCRV5ServerDetImageProcessorFast,
    PPOCRV5ServerDetModel,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_cv2,
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


class PPOCRV5ServerDetModelTester:
    def __init__(
        self, batch_size=3, image_size=128, num_channels=3, num_stages=5, is_training=False, scale=1.0, divisor=16
    ):
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.num_stages = num_stages
        self.scale = scale
        self.divisor = divisor

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self) -> PPOCRV5ServerDetConfig:
        # Minimal backbone config for fast tests (< 1M params for test_model_is_small)
        backbone_config = {
            "model_type": "hgnet_v2",
            "out_features": ["stage1", "stage2", "stage3", "stage4"],
            "out_indices": [1, 2, 3, 4],
            "depths": [1, 1, 1, 1],
            "hidden_sizes": [16, 32, 64, 128],
            "stage_in_channels": [16, 16, 32, 64],
            "stage_mid_channels": [16, 16, 32, 64],
            "stage_out_channels": [16, 32, 64, 128],
            "stage_num_blocks": [1, 1, 1, 1],
            "stage_downsample": [False, True, True, True],
            "stage_light_block": [False, False, True, True],
            "stage_kernel_size": [3, 3, 5, 5],
            "stage_numb_of_layers": [1, 1, 1, 1],
            "stem_channels": [3, 8, 16],
            "embedding_size": 16,
        }

        intraclass_block_config = {
            "reduce_channel": [1, 1, 0],
            "return_channel": [1, 1, 0],
            "vertical_long_to_small_conv_longratio": [[7, 1], [1, 1], [3, 0]],
            "vertical_long_to_small_conv_midratio": [[5, 1], [1, 1], [2, 0]],
            "vertical_long_to_small_conv_shortratio": [[3, 1], [1, 1], [1, 0]],
            "horizontal_small_to_long_conv_longratio": [[1, 7], [1, 1], [0, 3]],
            "horizontal_small_to_long_conv_midratio": [[1, 5], [1, 1], [0, 2]],
            "horizontal_small_to_long_conv_shortratio": [[1, 3], [1, 1], [0, 1]],
            "symmetric_conv_long_longratio": [[7, 7], [1, 1], [3, 3]],
            "symmetric_conv_long_midratio": [[5, 5], [1, 1], [2, 2]],
            "symmetric_conv_long_shortratio": [[3, 3], [1, 1], [1, 1]],
        }

        config = PPOCRV5ServerDetConfig(
            backbone_config=backbone_config,
            interpolate_mode="nearest",
            neck_out_channels=32,
            reduce_factor=2,
            intraclass_block_number=4,
            intraclass_block_config=intraclass_block_config,
            mode="large",
            scale_factor=2,
            scale_factor_list=[1, 2, 4, 8],
            hidden_act="relu",
            kernel_list=[3, 2, 2],
        )

        return config

    def create_and_check_pp_ocrv5_server_det_object_detection(self, config, pixel_values):
        model = PPOCRV5ServerDetForObjectDetection(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, 1, self.image_size, self.image_size))


@require_torch
class PPOCRV5ServerDetModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PPOCRV5ServerDetModel, PPOCRV5ServerDetForObjectDetection) if is_torch_available() else ()
    pipeline_model_mapping = {"object-detection": PPOCRV5ServerDetForObjectDetection} if is_torch_available() else {}

    test_resize_embeddings = False
    has_attentions = False

    def setUp(self):
        self.model_tester = PPOCRV5ServerDetModelTester(
            batch_size=3,
            is_training=False,
            image_size=128,
        )
        self.model_tester.parent = self
        self.config_tester = ConfigTester(
            self,
            config_class=PPOCRV5ServerDetConfig,
            has_text_modality=False,
            common_properties=[],
        )

    def test_config(self):
        # Skip create_and_test_config_with_num_labels: PP-OCRV5 has fixed single class (text)
        self.config_tester.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_from_and_save_pretrained_subfolder()
        self.config_tester.create_and_test_config_from_and_save_pretrained_composite()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()
        self.config_tester.create_and_test_config_from_pretrained_custom_kwargs()

    def test_pp_ocrv5_server_det_object_detection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_pp_ocrv5_server_det_object_detection(*config_and_inputs)

    @unittest.skip(reason="PPOCRV5ServerDet does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

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

            # First hidden state (embedding output) is 4x downsampled
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.image_size // 4, self.model_tester.image_size // 4],
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


@require_torch
@require_vision
@require_cv2
@slow
class PPOCRV5ServerDetModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-OCRV5_server_det_safetensors"
        self.model = PPOCRV5ServerDetForObjectDetection.from_pretrained(model_path).to(torch_device)
        self.image_processor = (
            PPOCRV5ServerDetImageProcessorFast.from_pretrained(model_path) if is_vision_available() else None
        )
        self.image = Image.open(
            requests.get(
                "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png", stream=True
            ).raw
        ).convert("RGB")

    def test_inference_object_detection_head(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)
        bs, c, h, w = inputs["pixel_values"].shape

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.image_processor.post_process_object_detection(outputs, target_sizes=inputs["target_sizes"])

        expected_shape_logits = torch.Size((bs, c // 3, h, w))

        expected_logits = torch.tensor(
            [
                [0.0004, 0.0003, 0.0002],
                [0.0003, 0.0002, 0.0002],
                [0.0006, 0.0003, 0.0003],
            ],
            device=torch_device,
        )

        self.assertEqual(outputs.last_hidden_state.shape, expected_shape_logits)
        torch.testing.assert_close(outputs.last_hidden_state[0, 0, :3, :3], expected_logits, rtol=2e-4, atol=2e-4)
        expected_shape_boxes = torch.Size((4, 4, 2))
        expected_boxes = torch.tensor(
            [
                [[76, 550], [399, 538], [400, 575], [77, 587]],
                [[14, 505], [517, 484], [519, 532], [16, 553]],
                [[193, 452], [401, 443], [403, 483], [195, 492]],
                [[32, 406], [488, 384], [491, 434], [34, 456]],
            ],
            dtype=torch.short,
            device=torch_device,
        )

        self.assertEqual(results[0]["boxes"].shape, expected_shape_boxes)
        torch.testing.assert_close(results[0]["boxes"], expected_boxes, rtol=2e-2, atol=2e-2)

        expected_scores = torch.tensor([0.9023, 0.8941, 0.8937, 0.8781], device=torch_device)
        self.assertEqual(results[0]["scores"].shape, (4,))
        torch.testing.assert_close(results[0]["scores"], expected_scores, rtol=2e-2, atol=2e-2)

        self.assertEqual(results[0]["labels"].shape, (4,))
        self.assertTrue((results[0]["labels"] == 0).all())  # Single class: text
