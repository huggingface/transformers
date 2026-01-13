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
"""Testing suite for the PP-OCRV5MobileDet model."""

import inspect
import unittest

from parameterized import parameterized

from transformers import (
    PPOCRV5MobileDetConfig,
    PPOCRV5MobileDetForObjectDetection,
    PPOCRV5MobileDetImageProcessor,
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


class PPOCRV5MobileDetModelTester:
    def __init__(
        self, batch_size=3, image_size=128, num_channels=3, num_stages=6, is_training=False, scale=1.0, divisor=16
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

    def get_config(self) -> PPOCRV5MobileDetConfig:
        backbone_config = {
            "blocks2": [[3, 16, 24, 1, False]],
            "blocks3": [[3, 24, 48, 2, False], [3, 48, 48, 1, False]],
            "blocks4": [[3, 48, 96, 2, False], [3, 96, 96, 1, False]],
            "blocks5": [
                [3, 96, 192, 2, False],
                [5, 192, 192, 1, False],
                [5, 192, 192, 1, False],
                [5, 192, 192, 1, False],
                [5, 192, 192, 1, False],
            ],
            "blocks6": [
                [5, 192, 384, 2, True],
                [5, 384, 384, 1, True],
                [5, 384, 384, 1, False],
                [5, 384, 384, 1, False],
            ],
            "layer_list_out_channels": [12, 18, 42, 360],
        }
        self.backbone_config = backbone_config

        config = PPOCRV5MobileDetConfig(
            backbone_config=backbone_config,
            scale=1.0,
            conv_kxk_num=4,
            reduction=4,
            divisor=16,
            num_channels=3,
            backbone_out_channels=512,
            hidden_act="hswish",
            neck_out_channels=96,
            shortcut=True,
            interpolate_mode="nearest",
            k=50,
            kernel_list=[3, 2, 2],
            fix_nan=False,
            output_hidden_states=False,
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

    def setUp(self):
        self.model_tester = PPOCRV5MobileDetModelTester(
            batch_size=3,
            is_training=False,
            image_size=128,
        )
        self.model_tester.parent = self
        self.config_tester = ConfigTester(
            self,
            config_class=PPOCRV5MobileDetConfig,
            has_text_modality=False,
            common_properties=["num_channels"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    # @unittest.skip(reason="PPOCRV5MobileDet does not use inputs_embeds")
    def test_pp_ocrv5_mobile_det_object_detection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_pp_ocrv5_mobile_det_object_detection(*config_and_inputs)

    @unittest.skip(reason="PPOCRV5MobileDet does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="PPOCRV5MobileDet does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="PPOCRV5MobileDet does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="PPOCRV5MobileDet does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="PPOCRV5MobileDet does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="PPOCRV5MobileDet does not support this test")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="PPOCRV5MobileDet does not support attention")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="PPOCRV5MobileDet does not support attention")
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
        def make_divisible(v, divisor: int = 16, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_stages = self.model_tester.num_stages
            scale = self.model_tester.scale
            divisor = self.model_tester.divisor
            self.assertEqual(len(hidden_states), expected_num_stages + 1)

            self.assertEqual(hidden_states[0].shape[-1], self.model_tester.image_size)

            self.assertEqual(
                hidden_states[1].shape[1],
                make_divisible(self.model_tester.backbone_config["blocks2"][0][1] * scale, divisor),
            )

            self.assertEqual(
                hidden_states[2].shape[1],
                make_divisible(self.model_tester.backbone_config["blocks2"][0][2] * scale, divisor),
            )

            self.assertEqual(
                hidden_states[3].shape[1],
                make_divisible(self.model_tester.backbone_config["blocks3"][0][2] * scale, divisor),
            )

            self.assertEqual(
                hidden_states[4].shape[1],
                make_divisible(self.model_tester.backbone_config["blocks4"][0][2] * scale, divisor),
            )

            self.assertEqual(
                hidden_states[5].shape[1],
                make_divisible(self.model_tester.backbone_config["blocks5"][0][2] * scale, divisor),
            )

            self.assertEqual(
                hidden_states[6].shape[1],
                make_divisible(self.model_tester.backbone_config["blocks6"][0][2] * scale, divisor),
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
# @slow
class PPOCRV5MobileDetModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "/workspace/model_weight_torch/PP-OCRv5_mobile_det"
        self.model = PPOCRV5MobileDetForObjectDetection.from_pretrained(model_path).to(torch_device)
        self.image_processor = (
            PPOCRV5MobileDetImageProcessor.from_pretrained(model_path) if is_vision_available() else None
        )
        path = "/workspace/PaddleX/paddlex/inference/models/text_detection/modeling/general_ocr_001.png"
        self.image = Image.open(path)

    def test_inference_object_detection_head(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)
        bs, c, h, w = inputs["pixel_values"].shape

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.image_processor.post_process_object_detection(
            outputs.logits, ori_size_list=inputs["target_sizes"]
        )

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
