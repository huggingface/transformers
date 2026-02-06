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
"""Testing suite for the PPOCRV5ServerRec model."""

import inspect
import unittest

from parameterized import parameterized

from transformers import (
    PPOCRV5ServerRecConfig,
    PPOCRV5ServerRecForTextRecognition,
    PPOCRV5ServerRecImageProcessor,
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
    ):
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])
        config = self.get_config()

        return config, pixel_values

    def get_config(self) -> PPOCRV5ServerRecConfig:
        config = PPOCRV5ServerRecConfig(
            stage_config={
                "stage1": [48, 48, 128, 1, True, False, 3, 6, [2, 1]],
                "stage2": [128, 96, 512, 1, True, False, 3, 6, [1, 2]],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6, [2, 1]],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6, [2, 1]],
            },
            stem_channels=[3, 32, 48],
            text_rec=True,
            det=False,
            use_lab=False,
            use_last_conv=True,
            class_expand=2048,
            dropout_prob=0.0,
            class_num=1000,
            lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0],
            out_indices=None,
            head_list=[
                {
                    "CTCHead": {
                        "Neck": {
                            "name": "svtr",
                            "dims": 120,
                            "depth": 2,
                            "hidden_dims": 120,
                            "kernel_size": [1, 3],
                            "use_guide": True,
                        },
                        "Head": {"fc_decay": 0.00001},
                    }
                },
                {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
            ],
            decode_list={"CTCLabelDecode": 18385, "SARLabelDecode": 18387, "NRTRLabelDecode": 18388},
        )

        return config


@require_torch
class PPOCRV5ServerRecModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PPOCRV5ServerRecForTextRecognition,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-feature-extraction": PPOCRV5ServerRecForTextRecognition} if is_torch_available() else {}
    )

    has_attentions = False

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

    @unittest.skip(reason="PPOCRV5ServerRec does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerRec does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerRec does not use test_init_weights_can_init_buffers")
    def test_init_weights_can_init_buffers(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerRec does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerRec does not support init all missing weights")
    def test_can_init_all_missing_weights(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerRec does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerRec does not support batching inference")
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerRec does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerRec does not support this test")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerRec does not support attention")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerRec does not support attention")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="PPOCRV5ServerRec does not support train")
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

    @unittest.skip(reason="PPOCRV5ServerRec does not support hidden_states")
    def test_hidden_states_output(self):
        pass


@require_torch
@require_vision
@slow
class PPOCRV5ServerRecModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "./pp_ocrv5_server_rec_model"

        self.model = PPOCRV5ServerRecForTextRecognition.from_pretrained(model_path).to(torch_device)
        self.image_processor = (
            PPOCRV5ServerRecImageProcessor.from_pretrained(model_path, return_tensors="pt")
            if is_vision_available()
            else None
        )
        path = "./general_ocr_rec_001.png"
        self.image = Image.open(path).convert("RGB")

    def test_inference_text_recognition_head(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt")["pixel_values"]
        inputs = inputs.to(torch_device)

        with torch.no_grad():
            outputs = self.model(inputs)

        pred_text = self.image_processor.post_process_text_recognition(outputs)[0][0]
        expected_text = "绿洲仕格维花园公寓"

        self.assertEqual(pred_text, expected_text)
