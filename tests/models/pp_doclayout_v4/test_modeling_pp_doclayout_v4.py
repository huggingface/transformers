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
"""Testing suite for the PP-DocLayoutV4 model."""

import math
import unittest

from transformers import (
    PPDocLayoutV4Config,
    PPDocLayoutV4ForObjectDetection,
    PPDocLayoutV4ImageProcessor,
    is_torch_available,
    is_vision_available,
)
from transformers.image_utils import load_image
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_vision,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch


class PPDocLayoutV4ModelTester:
    def __init__(self, parent, batch_size=3, image_size=128, num_labels=25):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = 3
        self.image_size = image_size
        self.num_labels = num_labels
        self.is_training = False

        self.encoder_hidden_dim = 32
        self.encoder_in_channels = [32, 32, 32]
        self.feat_strides = [8, 16, 32]
        self.encoder_layers = 1
        self.encoder_attention_heads = 2
        self.d_model = 32
        self.num_queries = 30
        self.decoder_layers = 2
        self.decoder_attention_heads = 2

        self.encoder_seq_length = math.ceil(self.image_size / self.feat_strides[-1]) ** 2

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        return self.get_config(), pixel_values

    def get_config(self):
        backbone_config = {
            "model_type": "hgnet_v2",
            "hidden_sizes": [32, 32, 32, 32],
            "stem_channels": [3, 32, 32],
            "stage_in_channels": [32, 32, 32, 32],
            "stage_mid_channels": [32, 32, 32, 32],
            "stage_out_channels": [32, 32, 32, 32],
            # PP-DocLayoutV4 consumes the last three stages only.
            "return_idx": [1, 2, 3],
            "out_features": ["stage2", "stage3", "stage4"],
        }
        return PPDocLayoutV4Config(
            backbone_config=backbone_config,
            num_labels=self.num_labels,
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_in_channels=self.encoder_in_channels,
            feat_strides=self.feat_strides,
            encoder_layers=self.encoder_layers,
            encoder_ffn_dim=64,
            encoder_attention_heads=self.encoder_attention_heads,
            d_model=self.d_model,
            num_queries=self.num_queries,
            decoder_in_channels=[32, 32, 32],
            decoder_ffn_dim=8,
            decoder_layers=self.decoder_layers,
            decoder_attention_heads=self.decoder_attention_heads,
            # Denoising is a training-only path, and PP-DocLayoutV4 does not support training.
            num_denoising=0,
            global_pointer_head_size=8,
            gp_dropout_value=0.0,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        return config, {"pixel_values": pixel_values}


@require_torch
class PPDocLayoutV4ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PPDocLayoutV4ForObjectDetection,) if is_torch_available() else ()
    pipeline_model_mapping = {"object-detection": PPDocLayoutV4ForObjectDetection} if is_torch_available() else {}
    is_encoder_decoder = True

    test_missing_keys = False
    test_torch_exportable = True
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = PPDocLayoutV4ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=PPDocLayoutV4Config,
            has_text_modality=False,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="PPDocLayoutV4 does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="PPDocLayoutV4 does not support training")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    # PP-DocLayoutV4 has no `num_hidden_layers`: the encoder depth follows `encoder_in_channels` and the encoder
    # hidden states are feature maps, not sequences, so the common shape checks do not apply.
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config).to(torch_device).eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            feature_size = self.model_tester.image_size // self.model_tester.feat_strides[-1]
            encoder_hidden_states = outputs.encoder_hidden_states
            self.assertEqual(len(encoder_hidden_states), len(self.model_tester.encoder_in_channels) - 1)
            self.assertListEqual(list(encoder_hidden_states[1].shape[-2:]), [feature_size, feature_size])

            decoder_hidden_states = outputs.decoder_hidden_states
            self.assertEqual(len(decoder_hidden_states), self.model_tester.decoder_layers + 1)
            self.assertListEqual(
                list(decoder_hidden_states[0].shape[-2:]),
                [self.model_tester.num_queries, self.model_tester.d_model],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    # Same reason, plus the model returns more outputs than the common `correct_outlen` accounts for.
    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict["output_attentions"] = True

        for model_class in self.all_model_classes:
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(len(outputs.encoder_attentions), self.model_tester.encoder_layers)
            self.assertListEqual(
                list(outputs.encoder_attentions[0].shape[-3:]),
                [
                    self.model_tester.encoder_attention_heads,
                    self.model_tester.encoder_seq_length,
                    self.model_tester.encoder_seq_length,
                ],
            )

            self.assertEqual(len(outputs.decoder_attentions), self.model_tester.decoder_layers)
            self.assertListEqual(
                list(outputs.decoder_attentions[0].shape[-3:]),
                [
                    self.model_tester.decoder_attention_heads,
                    self.model_tester.num_queries,
                    self.model_tester.num_queries,
                ],
            )


@require_torch
@require_vision
@slow
class PPDocLayoutV4ModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-DocLayoutV4_safetensors"
        self.model = PPDocLayoutV4ForObjectDetection.from_pretrained(model_path).to(torch_device)
        self.image_processor = (
            PPDocLayoutV4ImageProcessor.from_pretrained(model_path) if is_vision_available() else None
        )
        img_url = url_to_local_path(
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg"
        )
        self.image = load_image(img_url)

    def test_inference_object_detection_head(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        expected_shape_logits = torch.Size((1, 300, self.model.config.num_labels))
        logits_expectations = Expectations(
            {
                (None, None): [
                    [-3.5623, -4.5347, -5.0415],
                    [-3.7752, -3.6966, -4.4212],
                    [-4.4829, -4.3740, -4.5478],
                ]
            }
        )
        expected_logits = torch.tensor(logits_expectations.get_expectation()).to(torch_device)
        self.assertEqual(outputs.logits.shape, expected_shape_logits)
        torch.testing.assert_close(outputs.logits[0, :3, :3], expected_logits, rtol=2e-4, atol=2e-2)

        expected_shape_boxes = torch.Size((1, 300, self.model.config.num_coords))
        boxes_expectations = Expectations(
            {
                (None, None): [
                    [0.3719, 0.1785, 0.3313],
                    [0.7257, 0.4412, 0.3301],
                    [0.7253, 0.2664, 0.3306],
                ]
            }
        )
        expected_boxes = torch.tensor(boxes_expectations.get_expectation()).to(torch_device)
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        torch.testing.assert_close(outputs.pred_boxes[0, :3, :3], expected_boxes, rtol=2e-4, atol=2e-2)

        expected_shape_order_logits = torch.Size((1, 300, 300))
        self.assertEqual(outputs.relative_order_logits.shape, expected_shape_order_logits)
        self.assertEqual(outputs.successor_order_logits.shape, expected_shape_order_logits)
        relative_order_expectations = Expectations(
            {
                (None, None): [
                    [0.0000, 42.6447, 46.4521],
                    [-42.6447, 0.0000, -28.6615],
                    [-46.4521, 28.6615, 0.0000],
                ]
            }
        )
        expected_relative_order_logits = torch.tensor(relative_order_expectations.get_expectation()).to(torch_device)
        torch.testing.assert_close(
            outputs.relative_order_logits[0, :3, :3], expected_relative_order_logits, rtol=2e-2, atol=2e-2
        )

        # verify postprocessing
        results = self.image_processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=[self.image.size[::-1]]
        )[0]

        scores_expectations = Expectations(
            {
                (None, None): [
                    0.9885,
                    0.9781,
                    0.9938,
                    0.9900,
                    0.9871,
                    0.9833,
                    0.9771,
                    0.9010,
                    0.9529,
                    0.6550,
                    0.7850,
                    0.9787,
                    0.9286,
                ]
            }
        )
        expected_scores = torch.tensor(scores_expectations.get_expectation()).to(torch_device)
        torch.testing.assert_close(results["scores"], expected_scores, rtol=2e-2, atol=2e-2)

        expected_labels = [22, 17, 22, 22, 22, 22, 22, 22, 22, 22, 10, 16, 8]
        self.assertSequenceEqual(results["labels"].tolist(), expected_labels)

        # Results come back sorted by reading order, which the model resolves into a single chain here.
        self.assertSequenceEqual(results["order_seq"].tolist(), list(range(13)))

        slice_boxes_expectations = Expectations(
            {
                (None, None): [
                    [336.0739, 182.0364, 894.1705, 652.6191],
                    [336.4460, 681.8829, 868.7751, 796.9087],
                    [334.0145, 840.8432, 889.1123, 1452.2927],
                    [920.6475, 183.6178, 1476.7504, 462.7547],
                ]
            }
        )
        expected_slice_boxes = torch.tensor(slice_boxes_expectations.get_expectation()).to(torch_device)
        torch.testing.assert_close(results["boxes"][:4], expected_slice_boxes, rtol=2e-2, atol=2e-2)

        # Unlike PP-DocLayoutV3 the polygon is always the four regressed corners, in TL, TR, BR, BL order.
        polygon_points_expectations = Expectations(
            {
                (None, None): [
                    [336.0739, 182.0364],
                    [893.9496, 182.2223],
                    [894.1705, 652.5367],
                    [336.4081, 652.6191],
                ]
            }
        )
        expected_polygon_points = torch.tensor(polygon_points_expectations.get_expectation()).to(torch_device)
        self.assertEqual(results["polygon_points"].shape, torch.Size((13, 4, 2)))
        torch.testing.assert_close(results["polygon_points"][0], expected_polygon_points, rtol=2e-2, atol=2e-2)
