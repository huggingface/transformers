# coding = utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from parameterized import parameterized

from transformers import (
    PPOCRV5ServerDetConfig,
    PPOCRV5ServerDetForObjectDetection,
    PPOCRV5ServerDetImageProcessor,
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
        backbone_config = {
            "stage1": [48, 48, 128, 1, False, False, 3, 6, 2],
            "stage2": [128, 96, 512, 1, True, False, 3, 6, 2],
            "stage3": [512, 192, 1024, 3, True, True, 5, 6, 2],
            "stage4": [1024, 384, 2048, 1, True, True, 5, 6, 2],
        }
        intraclblock_config = {
            "reduce_channel": [1, 1, 0],
            "return_channel": [1, 1, 0],
            "v_layer_7x1": [[7, 1], [1, 1], [3, 0]],
            "v_layer_5x1": [[5, 1], [1, 1], [2, 0]],
            "v_layer_3x1": [[3, 1], [1, 1], [1, 0]],
            "q_layer_1x7": [[1, 7], [1, 1], [0, 3]],
            "q_layer_1x5": [[1, 5], [1, 1], [0, 2]],
            "q_layer_1x3": [[1, 3], [1, 1], [0, 1]],
            "c_layer_7x7": [[7, 7], [1, 1], [3, 3]],
            "c_layer_5x5": [[5, 5], [1, 1], [2, 2]],
            "c_layer_3x3": [[3, 3], [1, 1], [1, 1]],
        }
        self.backbone_config = backbone_config

        config = PPOCRV5ServerDetConfig(
            interpolate_mode="nearest",
            stem_channels=[3, 32, 48],
            backbone_config=backbone_config,
            use_lab=False,
            use_last_conv=True,
            class_expand=2048,
            dropout_prob=0.0,
            class_num=1000,
            out_indices=[0, 1, 2, 3],
            neck_out_channels=256,
            reduce_factor=2,
            intraclblock_config=intraclblock_config,
            head_in_channels=1024,
            k=50,
            mode="large",
            scale_factor=2,
            hidden_act="relu",
            kernel_list=[3, 2, 2],
            fix_nan=False,
        )

        return config

    def create_and_check_pp_ocrv5_server_det_object_detection(self, config, pixel_values):
        model = PPOCRV5ServerDetForObjectDetection(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, 1, self.image_size, self.image_size))


@require_torch
class PPOCRV5ServerDetModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (PPOCRV5ServerDetForObjectDetection,) if is_torch_available() else ()
    pipeline_model_mapping = {"object-detection": PPOCRV5ServerDetForObjectDetection} if is_torch_available() else {}

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
        self.config_tester.run_common_tests()

    # @unittest.skip(reason="PPOCRV5ServerDet does not use inputs_embeds")
    def test_pp_ocrv5_server_det_object_detection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_pp_ocrv5_server_det_object_detection(*config_and_inputs)

    @unittest.skip(reason="PPOCRV5ServerDet does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerDet does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerDet does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerDet does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerDet does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerDet does not support this test")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerDet does not support attention")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerDet does not support attention")
    def test_attention_outputs(self):
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

            self.assertEqual(hidden_states[0].shape[-1], self.model_tester.image_size)

            self.assertEqual(
                hidden_states[1].shape[1],
                self.model_tester.backbone_config["stage1"][0],
            )

            self.assertEqual(
                hidden_states[2].shape[1],
                self.model_tester.backbone_config["stage1"][1],
            )

            self.assertEqual(
                hidden_states[3].shape[1],
                self.model_tester.backbone_config["stage2"][0],
            )

            self.assertEqual(
                hidden_states[4].shape[1],
                self.model_tester.backbone_config["stage3"][0],
            )

            self.assertEqual(
                hidden_states[5].shape[1],
                self.model_tester.backbone_config["stage4"][0],
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
class PPOCRV5ServerDetModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "/workspace/model_weight_torch/PP-OCRv5_server_det"
        self.model = PPOCRV5ServerDetForObjectDetection.from_pretrained(model_path).to(torch_device)
        self.image_processor = (
            PPOCRV5ServerDetImageProcessor.from_pretrained(model_path) if is_vision_available() else None
        )
        path = "/workspace/PaddleX/paddlex/inference/models/text_detection/modeling/general_ocr_001.png"
        self.image = Image.open(path)

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
            ]
        ).to(torch_device)

        self.assertEqual(outputs.logits.shape, expected_shape_logits)
        torch.testing.assert_close(outputs.logits[0, 0, :3, :3], expected_logits, rtol=2e-4, atol=2e-4)

        expected_shape_boxes = torch.Size((4, 4, 2))
        expected_boxes = torch.tensor(
            [
                [[76, 546], [398, 534], [400, 575], [77, 587]],
                [[14, 499], [517, 478], [519, 532], [16, 553]],
                [[193, 447], [401, 438], [403, 483], [195, 492]],
                [[31, 400], [488, 378], [491, 434], [34, 456]],
            ],
        ).to(torch_device)

        self.assertEqual(results[0]["boxes"].shape, expected_shape_boxes)
        torch.testing.assert_close(
            torch.from_numpy(results[0]["boxes"]).to(device=torch_device, dtype=torch.int64),
            expected_boxes,
            rtol=2e-4,
            atol=2e-4,
        )

        expected_scores = torch.tensor(
            [0.9024514491711588, 0.8939725431302765, 0.8937050561554075, 0.8780838469131043]
        ).to(torch_device)
        self.assertEqual(len(results[0]["scores"]), 4)
        torch.testing.assert_close(
            torch.tensor(results[0]["scores"]).to(device=torch_device), expected_scores, rtol=2e-4, atol=2e-4
        )
