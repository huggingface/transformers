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
"""Testing suite for the PP-DocLayoutV3 model."""

import inspect
import math
import unittest

import pytest
import requests
from parameterized import parameterized

from transformers import (
    PPDocLayoutV3Config,
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ImageProcessorFast,
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


class PPDocLayoutV3ModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        is_training=False,
        n_targets=3,
        num_labels=25,
        initializer_range=0.01,
        layer_norm_eps=1e-5,
        batch_norm_eps=1e-5,
        # backbone
        backbone_config=None,
        # encoder HybridEncoder
        encoder_hidden_dim=32,
        encoder_in_channels=[32, 32, 32],
        feat_strides=[8, 16, 32],
        encoder_layers=1,
        encoder_ffn_dim=64,
        encoder_attention_heads=2,
        dropout=0.0,
        activation_dropout=0.0,
        encode_proj_layers=[2],
        positional_encoding_temperature=10000,
        encoder_activation_function="gelu",
        activation_function="silu",
        eval_size=None,
        normalize_before=False,
        mask_feature_channels=[32, 32],
        x4_feat_dim=32,
        # decoder PPDocLayoutV3Transformer
        d_model=32,
        num_queries=30,
        decoder_in_channels=[32, 32, 32],
        decoder_ffn_dim=8,
        num_feature_levels=3,
        decoder_n_points=4,
        decoder_layers=2,
        decoder_attention_heads=2,
        decoder_activation_function="relu",
        attention_dropout=0.0,
        num_denoising=0,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_initial_query=False,
        anchor_image_size=None,
        image_size=128,
        disable_custom_kernels=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = 3
        self.is_training = is_training
        self.n_targets = n_targets
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps
        self.backbone_config = backbone_config
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_in_channels = encoder_in_channels
        self.feat_strides = feat_strides
        self.num_labels = num_labels
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.encode_proj_layers = encode_proj_layers
        self.positional_encoding_temperature = positional_encoding_temperature
        self.encoder_activation_function = encoder_activation_function
        self.activation_function = activation_function
        self.eval_size = eval_size
        self.normalize_before = normalize_before
        self.mask_feature_channels = mask_feature_channels
        self.x4_feat_dim = x4_feat_dim
        self.d_model = d_model
        self.num_queries = num_queries
        self.decoder_in_channels = decoder_in_channels
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_feature_levels = num_feature_levels
        self.decoder_n_points = decoder_n_points
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_activation_function = decoder_activation_function
        self.attention_dropout = attention_dropout
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learn_initial_query = learn_initial_query
        self.anchor_image_size = anchor_image_size
        self.image_size = image_size
        self.disable_custom_kernels = disable_custom_kernels

        self.encoder_seq_length = math.ceil(self.image_size / 32) * math.ceil(self.image_size / 32)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        hidden_sizes = [10, 20, 30, 40]
        backbone_config = {
            "model_type": "hgnet_v2",
            "arch": "L",
            "return_idx": [0, 1, 2, 3],
            "hidden_sizes": [32, 32, 32, 32],
            "stem_channels": [3, 32, 32],
            "stage_in_channels": [32, 32, 32, 32],
            "stage_mid_channels": [32, 32, 32, 32],
            "stage_out_channels": [32, 32, 32, 32],
            "freeze_stem_only": True,
            "freeze_at": 0,
            "freeze_norm": True,
            "lr_mult_list": [0, 0.05, 0.05, 0.05, 0.05],
            "out_features": ["stage1", "stage2", "stage3", "stage4"],
        }
        return PPDocLayoutV3Config(
            backbone_config=backbone_config,
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_in_channels=hidden_sizes[1:],
            feat_strides=self.feat_strides,
            encoder_layers=self.encoder_layers,
            encoder_ffn_dim=self.encoder_ffn_dim,
            encoder_attention_heads=self.encoder_attention_heads,
            dropout=self.dropout,
            activation_dropout=self.activation_dropout,
            encode_proj_layers=self.encode_proj_layers,
            positional_encoding_temperature=self.positional_encoding_temperature,
            encoder_activation_function=self.encoder_activation_function,
            activation_function=self.activation_function,
            eval_size=self.eval_size,
            normalize_before=self.normalize_before,
            mask_feature_channels=self.mask_feature_channels,
            x4_feat_dim=self.x4_feat_dim,
            d_model=self.d_model,
            num_queries=self.num_queries,
            decoder_in_channels=self.decoder_in_channels,
            decoder_ffn_dim=self.decoder_ffn_dim,
            num_feature_levels=self.num_feature_levels,
            decoder_n_points=self.decoder_n_points,
            decoder_layers=self.decoder_layers,
            decoder_attention_heads=self.decoder_attention_heads,
            decoder_activation_function=self.decoder_activation_function,
            attention_dropout=self.attention_dropout,
            num_denoising=self.num_denoising,
            label_noise_ratio=self.label_noise_ratio,
            box_noise_scale=self.box_noise_scale,
            learn_initial_query=self.learn_initial_query,
            anchor_image_size=self.anchor_image_size,
            image_size=self.image_size,
            disable_custom_kernels=self.disable_custom_kernels,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class PPDocLayoutV3ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (PPDocLayoutV3ForObjectDetection,) if is_torch_available() else ()
    pipeline_model_mapping = {"object-detection": PPDocLayoutV3ForObjectDetection} if is_torch_available() else {}
    is_encoder_decoder = True

    test_missing_keys = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = PPDocLayoutV3ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=PPDocLayoutV3Config,
            has_text_modality=False,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="PPDocLayoutV3 has tied weights.")
    def test_load_save_without_tied_weights(self):
        pass

    @unittest.skip(reason="PPDocLayoutV3 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="PPDocLayoutV3 does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="PPDocLayoutV3 does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="PPDocLayoutV3 does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="PPDocLayoutV3 does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="PPDocLayoutV3 does not support training")
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
                inputs_dict[key] = tensor.to(dtype)
            with torch.no_grad():
                _ = model(**self._prepare_for_class(inputs_dict, model_class))

    # We have not `num_hidden_layers`, use `encoder_in_channels` instead
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", len(self.model_tester.encoder_in_channels) - 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertListEqual(
                list(hidden_states[1].shape[-2:]),
                [
                    self.model_tester.image_size // self.model_tester.feat_strides[-1],
                    self.model_tester.image_size // self.model_tester.feat_strides[-1],
                ],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                expected_num_layers = getattr(
                    self.model_tester, "expected_num_hidden_layers", self.model_tester.decoder_layers + 1
                )

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [self.model_tester.num_queries, self.model_tester.d_model],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions
            self.assertEqual(len(attentions), self.model_tester.encoder_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions
            self.assertEqual(len(attentions), self.model_tester.encoder_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [
                    self.model_tester.encoder_attention_heads,
                    self.model_tester.encoder_seq_length,
                    self.model_tester.encoder_seq_length,
                ],
            )
            out_len = len(outputs)

            correct_outlen = 14

            # loss is at first position
            if "labels" in inputs_dict:
                correct_outlen += 1  # loss is added to beginning
            # Object Detection model returns pred_logits and pred_boxes
            if model_class.__name__ == "PPDocLayoutV3ForObjectDetection":
                correct_outlen += 3

            self.assertEqual(out_len, correct_outlen)

            # decoder attentions
            decoder_attentions = outputs.decoder_attentions
            self.assertIsInstance(decoder_attentions, (list, tuple))
            self.assertEqual(len(decoder_attentions), self.model_tester.decoder_layers)
            self.assertListEqual(
                list(decoder_attentions[0].shape[-3:]),
                [
                    self.model_tester.decoder_attention_heads,
                    self.model_tester.num_queries,
                    self.model_tester.num_queries,
                ],
            )

            # cross attentions
            cross_attentions = outputs.cross_attentions
            self.assertIsInstance(cross_attentions, (list, tuple))
            self.assertEqual(len(cross_attentions), self.model_tester.decoder_layers)
            self.assertListEqual(
                list(cross_attentions[0].shape[-3:]),
                [
                    self.model_tester.decoder_attention_heads,
                    self.model_tester.num_feature_levels,
                    self.model_tester.decoder_n_points,
                ],
            )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            else:
                added_hidden_states = 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions

            self.assertEqual(len(self_attentions), self.model_tester.encoder_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [
                    self.model_tester.encoder_attention_heads,
                    self.model_tester.encoder_seq_length,
                    self.model_tester.encoder_seq_length,
                ],
            )


# TODO:
# Later, we will determine these values based on the latest weights.
@require_torch
@require_vision
@slow
@pytest.mark.xfail(reason="Weigths will determine the values of these tests")
class PPDocLayoutV3ModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        model_path = "PaddlePaddle/PP-DocLayoutV3_safetensors"
        self.model = PPDocLayoutV3ForObjectDetection.from_pretrained(model_path).to(torch_device)
        self.image_processor = (
            PPDocLayoutV3ImageProcessorFast.from_pretrained(model_path) if is_vision_available() else None
        )
        url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_demo.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw)

    def test_inference_object_detection_head(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        expected_shape_logits = torch.Size((1, 300, self.model.config.num_labels))
        expected_logits = torch.tensor(
            [[-5.6224, -6.5667, -4.9352], [-5.7931, -5.5543, -5.6476], [-4.5742, -5.0603, -7.2864]]
        ).to(torch_device)
        self.assertEqual(outputs.logits.shape, expected_shape_logits)
        torch.testing.assert_close(outputs.logits[0, :3, :3], expected_logits, rtol=2e-4, atol=2e-4)

        expected_shape_boxes = torch.Size((1, 300, 4))
        expected_boxes = torch.tensor(
            [[0.4000, 0.9702, 0.3897], [0.3642, 0.3164, 0.3212], [0.3716, 0.1786, 0.3386]]
        ).to(torch_device)
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        torch.testing.assert_close(outputs.pred_boxes[0, :3, :3], expected_boxes, rtol=2e-4, atol=2e-4)

        expected_shape_order_logits = torch.Size((1, 300, 300))
        expected_order_logits = torch.tensor(
            [
                [-10000.0000, -937.0416, -1045.3816],
                [-10000.0000, -10000.0000, -343.8752],
                [-10000.0000, -10000.0000, -10000.0000],
            ]
        ).to(torch_device)
        self.assertEqual(outputs.order_logits.shape, expected_shape_order_logits)
        torch.testing.assert_close(outputs.order_logits[0, :3, :3], expected_order_logits, rtol=2e-4, atol=2e-4)

        # verify postprocessing
        results = self.image_processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=[self.image.size[::-1]]
        )[0]

        expected_scores = torch.tensor(
            [0.9834, 0.9485, 0.9837, 0.9728, 0.9741, 0.9770, 0.9508, 0.9390, 0.9482, 0.8391, 0.9358, 0.8249, 0.9095]
        ).to(torch_device)
        torch.testing.assert_close(results["scores"], expected_scores, rtol=2e-4, atol=2e-4)

        expected_labels = [22, 17, 22, 22, 22, 22, 22, 10, 10, 22, 10, 16, 8]
        self.assertSequenceEqual(results["labels"].tolist(), expected_labels)

        expected_slice_boxes = torch.tensor(
            [
                [334.5682, 182.9777, 894.6927, 652.4594],
                [336.7216, 683.4235, 867.9361, 796.9210],
                [335.2677, 841.1227, 891.1608, 1453.0148],
                [919.8677, 183.4835, 1475.8800, 463.4977],
            ]
        ).to(torch_device)
        torch.testing.assert_close(results["boxes"][:4], expected_slice_boxes, rtol=2e-4, atol=2e-4)
