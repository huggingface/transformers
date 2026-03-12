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
"""Testing suite for the PP-OCRV5MobileDet model."""

import inspect
import unittest

import requests
from parameterized import parameterized

from transformers import (
    PPOCRV5MobileDetConfig,
    PPOCRV5MobileDetForObjectDetection,
    PPOCRV5MobileDetImageProcessorFast,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    is_flaky,
    require_cv2,
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


class PPOCRV5MobileDetModelTester:
    def __init__(
        self,
        batch_size=3,
        image_size=128,
        num_channels=3,
        num_stages=5,
        is_training=False,
        reduction=4,
        hidden_act="hardswish",
        layer_list_out_channels=[12, 24, 42, 360],
        neck_out_channels=96,
        kernel_list=[3, 2, 2],
        interpolate_mode="nearest",
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_stages = num_stages
        self.is_training = is_training
        self.reduction = reduction
        self.hidden_act = hidden_act
        self.layer_list_out_channels = layer_list_out_channels
        self.neck_out_channels = neck_out_channels
        self.kernel_list = kernel_list
        self.interpolate_mode = interpolate_mode

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self) -> PPOCRV5MobileDetConfig:
        backbone_config = {
            "model_type": "pp_lcnet_v3",
            "scale": 1,
            "out_features": ["stage2", "stage3", "stage4", "stage5"],
            "out_indices": [2, 3, 4, 5],
            "divisor": 16,
            "block_configs": [
                [[3, 16, 32, 1, False]],
                [[3, 32, 32, 2, False], [3, 32, 32, 1, False]],
                [[3, 32, 32, 2, False], [3, 32, 32, 1, False]],
                [
                    [3, 32, 32, 2, False],
                    [5, 32, 32, 1, False],
                    [5, 32, 32, 1, False],
                    [5, 32, 32, 1, False],
                    [5, 32, 32, 1, False],
                ],
                [[5, 32, 32, 2, True], [5, 32, 32, 1, True], [5, 32, 32, 1, False], [5, 32, 32, 1, False]],
            ],
        }
        config = PPOCRV5MobileDetConfig(
            backbone_config=backbone_config,
            reduction=self.reduction,
            hidden_act=self.hidden_act,
            layer_list_out_channels=self.layer_list_out_channels,
            neck_out_channels=self.neck_out_channels,
            kernel_list=self.kernel_list,
            interpolate_mode=self.interpolate_mode,
        )
        return config

    def create_and_check_pp_ocrv5_mobile_det_object_detection(self, config, pixel_values):
        model = PPOCRV5MobileDetForObjectDetection(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, 1, self.image_size, self.image_size))


@require_torch
class PPOCRV5MobileDetModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (PPOCRV5MobileDetForObjectDetection,) if is_torch_available() else ()
    pipeline_model_mapping = {"object-detection": PPOCRV5MobileDetForObjectDetection} if is_torch_available() else {}

    has_attentions = False
    test_inputs_embeds = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = PPOCRV5MobileDetModelTester(
            batch_size=3,
            is_training=False,
            image_size=128,
        )
        self.config_tester = ConfigTester(
            self, config_class=PPOCRV5MobileDetConfig, has_text_modality=False, common_properties=[]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @is_flaky()
    def test_batching_equivalence(self, atol=5e-2, rtol=5e-2):
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    @unittest.skip(reason="PPOCRV5MobileDet does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="PPOCRV5MobileDet does not support attention")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="PPOCRV5MobileDet does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="PPOCRV5MobileDet does not support.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    # PPOCRV5MobileDet have no seq_length
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
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

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


@require_cv2
@require_torch
@require_vision
@slow
class PPOCRV5MobileDetModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-OCRv5_mobile_det_safetensors"
        self.model = PPOCRV5MobileDetForObjectDetection.from_pretrained(model_path).to(torch_device)
        self.image_processor = (
            PPOCRV5MobileDetImageProcessorFast.from_pretrained(model_path) if is_vision_available() else None
        )
        url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png"
        self.image = Image.open(requests.get(url, stream=True).raw)

    def test_inference_object_detection_head(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)
        bs, c, h, w = inputs["pixel_values"].shape

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.image_processor.post_process_object_detection(outputs, target_sizes=inputs["target_sizes"])

        expected_shape_logits = torch.Size((bs, c // 3, h, w))

        expected_logits = torch.tensor(
            [
                [4.7810e-07, 5.0727e-07, 4.7810e-07],
                [5.7200e-07, 6.0746e-07, 5.7200e-07],
                [4.7810e-07, 5.0727e-07, 4.7810e-07],
            ]
        ).to(torch_device)

        self.assertEqual(outputs.logits.shape, expected_shape_logits)
        torch.testing.assert_close(outputs.logits[0, 0, :3, :3], expected_logits, rtol=2e-4, atol=2e-4)

        expected_shape_boxes = torch.Size((4, 4, 2))
        expected_boxes = torch.tensor(
            [
                [[75, 549], [450, 538], [452, 575], [77, 586]],
                [[11, 504], [517, 482], [519, 532], [13, 554]],
                [[188, 452], [401, 444], [402, 481], [189, 489]],
                [[37, 408], [486, 386], [489, 432], [39, 453]],
            ]
        ).to(torch_device)

        self.assertEqual(results[0]["boxes"].shape, expected_shape_boxes)
        torch.testing.assert_close(
            torch.from_numpy(results[0]["boxes"]).to(device=torch_device, dtype=torch.int64),
            expected_boxes,
            rtol=2e-4,
            atol=2e-4,
        )

        expected_scores = torch.tensor(
            [0.8365021048166233, 0.8168808277221563, 0.874713086968462, 0.8694220109429082]
        ).to(torch_device)
        self.assertEqual(len(results[0]["scores"]), 4)
        torch.testing.assert_close(
            torch.tensor(results[0]["scores"]).to(device=torch_device), expected_scores, rtol=2e-4, atol=2e-4
        )
