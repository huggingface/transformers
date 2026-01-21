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
"""Testing suite for the PP-LCNet model."""

import inspect
import unittest

from parameterized import parameterized

from transformers import (
    UVDocConfig,
    UVDocForDocumentRectification,
    UVDocImageProcessor,
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
            "bridge_1": 1,
            "bridge_2": 2,
            "bridge_3": 5,
            "bridge_4": [8, 3, 2],
            "bridge_5": [12, 7, 4],
            "bridge_6": [18, 12, 6],
        }

        self.dilation_values = dilation_values

        config = UVDocConfig(
            num_filter=32,
            in_channels=3,
            kernel_size=5,
            stride=[1, 2, 2, 2],
            map_num=[1, 2, 4, 8, 16],
            block_nums=[3, 4, 6, 3],
            dilation_values=dilation_values,
            padding_mode="reflect",
            upsample_size=[712, 488],
            upsample_mode="bilinear",
        )

        return config

    def create_and_check_pp_lcnet_image_classification(self, config, pixel_values):
        model = UVDocForDocumentRectification(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values)

        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_channels, self.image_size, self.image_size)
        )


@require_torch
class UVDocModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (UVDocForDocumentRectification,) if is_torch_available() else ()

    has_attentions = False

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

    def test_pp_lcnet_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_pp_lcnet_image_classification(*config_and_inputs)

    @unittest.skip(reason="UVDoc does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="UVDoc does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="UVDoc does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="UVDoc does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="UVDoc does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="UVDoc does not support this test")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="UVDoc does not support attention")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="UVDoc does not support attention")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="UVDoc does not support train")
    def test_problem_types(self):
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

            self.assertEqual(hidden_states[0].shape[1], self.model_tester.image_size)
            expected_hidden_states_output_shape = [3, 128, 45, 31]
            self.assertListEqual(list(hidden_states[0].shape), expected_hidden_states_output_shape)
            self.assertListEqual(list(hidden_states[1].shape), expected_hidden_states_output_shape)
            self.assertListEqual(list(hidden_states[2].shape), expected_hidden_states_output_shape)
            self.assertListEqual(list(hidden_states[3].shape), expected_hidden_states_output_shape)
            self.assertListEqual(list(hidden_states[4].shape), expected_hidden_states_output_shape)
            self.assertListEqual(list(hidden_states[5].shape), expected_hidden_states_output_shape)
            self.assertListEqual(list(hidden_states[6].shape), expected_hidden_states_output_shape)

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
class UVDocModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "/workspace/model_weight_torch/PP-LCNet_x1_0_doc_ori"

        self.model = UVDocForDocumentRectification.from_pretrained(model_path).to(torch_device)
        self.image_processor = UVDocImageProcessor.from_pretrained(model_path) if is_vision_available() else None
        path = "/workspace/PaddleX/paddlex/inference/models/image_unwarping/modeling/doc_test.jpg"
        self.image = Image.open(path)

    def test_inference_image_classification_head(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)
        bs, c, h, w = inputs["pixel_values"].shape

        with torch.no_grad():
            outputs = self.model(**inputs)

        expected_shape_logits = torch.Size((bs, 4))

        expected_logits = torch.tensor([[0.0511, 0.0259, 0.8973, 0.0257]]).to(torch_device)

        self.assertEqual(outputs.logits.shape, expected_shape_logits)
        torch.testing.assert_close(outputs.logits, expected_logits, rtol=2e-4, atol=2e-4)

        expected_labels = torch.tensor([2]).to(torch_device)
        predicted_label = outputs.logits.argmax(-1).item()

        self.assertEqual(predicted_label, expected_labels)
