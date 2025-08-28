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
"""Testing suite for the PyTorch D-FINE model."""

import inspect
import math
import tempfile
import unittest

from parameterized import parameterized

from transformers import (
    DFineConfig,
    HGNetV2Config,
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
from transformers.utils import cached_property


if is_torch_available():
    import torch

    from transformers import DFineForObjectDetection, DFineModel

if is_vision_available():
    from PIL import Image

from transformers import RTDetrImageProcessor

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


CHECKPOINT = "ustc-community/dfine-small-coco"


class DFineModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        is_training=True,
        use_labels=True,
        n_targets=3,
        num_labels=10,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        batch_norm_eps=1e-5,
        # backbone
        backbone_config=None,
        # encoder HybridEncoder
        encoder_hidden_dim=32,
        encoder_in_channels=[128, 256, 512],
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
        # decoder DFineTransformer
        d_model=32,
        num_queries=30,
        decoder_in_channels=[32, 32, 32],
        decoder_ffn_dim=64,
        num_feature_levels=3,
        decoder_n_points=[3, 6, 3],
        decoder_n_levels=3,
        decoder_layers=2,
        decoder_attention_heads=2,
        decoder_activation_function="relu",
        attention_dropout=0.0,
        num_denoising=0,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learn_initial_query=False,
        anchor_image_size=None,
        image_size=64,
        disable_custom_kernels=True,
        with_box_refine=True,
        decoder_offset_scale=0.5,
        eval_idx=-1,
        layer_scale=1,
        reg_max=32,
        reg_scale=4.0,
        depth_mult=0.34,
        hidden_expansion=0.5,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = 3
        self.is_training = is_training
        self.use_labels = use_labels
        self.n_targets = n_targets
        self.num_labels = num_labels
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps
        self.backbone_config = backbone_config
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_in_channels = encoder_in_channels
        self.feat_strides = feat_strides
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
        self.d_model = d_model
        self.num_queries = num_queries
        self.decoder_in_channels = decoder_in_channels
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_feature_levels = num_feature_levels
        self.decoder_n_points = decoder_n_points
        self.decoder_n_levels = decoder_n_levels
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_activation_function = decoder_activation_function
        self.attention_dropout = attention_dropout
        self.decoder_offset_scale = decoder_offset_scale
        self.eval_idx = eval_idx
        self.layer_scale = layer_scale
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        self.depth_mult = depth_mult
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learn_initial_query = learn_initial_query
        self.anchor_image_size = anchor_image_size
        self.image_size = image_size
        self.disable_custom_kernels = disable_custom_kernels
        self.with_box_refine = with_box_refine
        self.hidden_expansion = hidden_expansion

        self.encoder_seq_length = math.ceil(self.image_size / 32) * math.ceil(self.image_size / 32)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        pixel_mask = torch.ones([self.batch_size, self.image_size, self.image_size], device=torch_device)

        labels = None
        if self.use_labels:
            # labels is a list of Dict (each Dict being the labels for a given example in the batch)
            labels = []
            for i in range(self.batch_size):
                target = {}
                target["class_labels"] = torch.randint(
                    high=self.num_labels, size=(self.n_targets,), device=torch_device
                )
                target["boxes"] = torch.rand(self.n_targets, 4, device=torch_device)
                labels.append(target)

        config = self.get_config()
        config.num_labels = self.num_labels
        return config, pixel_values, pixel_mask, labels

    def get_config(self):
        hidden_sizes = [64, 128, 256, 512]
        backbone_config = HGNetV2Config(
            stage_in_channels=[16, 64, 128, 256],
            stage_mid_channels=[16, 32, 64, 128],
            stage_out_channels=[64, 128, 256, 512],
            stage_num_blocks=[1, 1, 2, 1],
            stage_downsample=[False, True, True, True],
            stage_light_block=[False, False, True, True],
            stage_kernel_size=[3, 3, 5, 5],
            stage_numb_of_layers=[3, 3, 3, 3],
            embeddings_size=10,
            hidden_sizes=hidden_sizes,
            depths=[1, 1, 2, 1],
            out_features=["stage2", "stage3", "stage4"],
            out_indices=[2, 3, 4],
            stem_channels=[3, 16, 16],
            use_lab=True,
        )
        return DFineConfig.from_backbone_configs(
            backbone_config=backbone_config,
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_in_channels=self.encoder_in_channels,
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
            d_model=self.d_model,
            num_queries=self.num_queries,
            decoder_in_channels=self.decoder_in_channels,
            decoder_ffn_dim=self.decoder_ffn_dim,
            num_feature_levels=self.num_feature_levels,
            decoder_n_points=self.decoder_n_points,
            decoder_n_levels=self.decoder_n_levels,
            decoder_layers=self.decoder_layers,
            decoder_attention_heads=self.decoder_attention_heads,
            decoder_activation_function=self.decoder_activation_function,
            decoder_offset_scale=self.decoder_offset_scale,
            eval_idx=self.eval_idx,
            layer_scale=self.layer_scale,
            reg_max=self.reg_max,
            reg_scale=self.reg_scale,
            depth_mult=self.depth_mult,
            attention_dropout=self.attention_dropout,
            num_denoising=self.num_denoising,
            label_noise_ratio=self.label_noise_ratio,
            box_noise_scale=self.box_noise_scale,
            learn_initial_query=self.learn_initial_query,
            anchor_image_size=self.anchor_image_size,
            image_size=self.image_size,
            disable_custom_kernels=self.disable_custom_kernels,
            with_box_refine=self.with_box_refine,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, labels = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def create_and_check_d_fine_model(self, config, pixel_values, pixel_mask, labels):
        model = DFineModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.num_queries, self.d_model))

    def create_and_check_d_fine_object_detection_head_model(self, config, pixel_values, pixel_mask, labels):
        model = DFineForObjectDetection(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_queries, self.num_labels))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_queries, self.num_labels))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))


@require_torch
class DFineModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (DFineModel, DFineForObjectDetection) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-feature-extraction": DFineModel, "object-detection": DFineForObjectDetection}
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    test_torch_exportable = True

    # special case for head models
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "DFineForObjectDetection":
                labels = []
                for i in range(self.model_tester.batch_size):
                    target = {}
                    target["class_labels"] = torch.ones(
                        size=(self.model_tester.n_targets,), device=torch_device, dtype=torch.long
                    )
                    target["boxes"] = torch.ones(
                        self.model_tester.n_targets, 4, device=torch_device, dtype=torch.float
                    )
                    labels.append(target)
                inputs_dict["labels"] = labels

        return inputs_dict

    def setUp(self):
        self.model_tester = DFineModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=DFineConfig,
            has_text_modality=False,
            common_properties=["hidden_size", "num_attention_heads"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_d_fine_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_d_fine_model(*config_and_inputs)

    def test_d_fine_object_detection_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_d_fine_object_detection_head_model(*config_and_inputs)

    @unittest.skip(reason="DFine doesn't work well with `nn.DataParallel")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="DFine does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="DFine does not use test_inputs_embeds_matches_input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="DFine does not support input and output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="DFine does not support input and output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="DFine does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="Not relevant for the model")
    def test_can_init_all_missing_weights(self):
        pass

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

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

            correct_outlen = 15

            # loss is at first position
            if "labels" in inputs_dict:
                correct_outlen += 1  # loss is added to beginning
            # Object Detection model returns pred_logits and pred_boxes
            if model_class.__name__ == "DFineForObjectDetection":
                correct_outlen += 2

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
                    self.model_tester.num_queries,
                    self.model_tester.decoder_attention_heads,
                    self.model_tester.decoder_n_levels * self.model_tester.decoder_n_points
                    if isinstance(self.model_tester.decoder_n_points, int)
                    else sum(self.model_tester.decoder_n_points),
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
                # DFine should maintin encoder_hidden_states output
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

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        # we take the first output since last_hidden_state is the first item
        output = outputs[0]

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_attentions = outputs.encoder_attentions[0]
        encoder_hidden_states.retain_grad()
        encoder_attentions.retain_grad()

        decoder_attentions = outputs.decoder_attentions[0]
        decoder_attentions.retain_grad()

        cross_attentions = outputs.cross_attentions[0]
        cross_attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(encoder_attentions.grad)
        self.assertIsNotNone(decoder_attentions.grad)
        self.assertIsNotNone(cross_attentions.grad)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_different_timm_backbone(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # let's pick a random timm backbone
        config.encoder_in_channels = [24, 40, 432]
        config.backbone = "tf_mobilenetv3_small_075"
        config.backbone_config = None
        config.use_timm_backbone = True
        config.backbone_kwargs = {"out_indices": [2, 3, 4]}

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if model_class.__name__ == "DFineForObjectDetection":
                expected_shape = (
                    self.model_tester.batch_size,
                    self.model_tester.num_queries,
                    self.model_tester.num_labels,
                )
                self.assertEqual(outputs.logits.shape, expected_shape)
                # Confirm out_indices was propagated to backbone
                self.assertEqual(len(model.model.backbone.intermediate_channel_sizes), 3)
            else:
                # Confirm out_indices was propagated to backbone
                self.assertEqual(len(model.backbone.intermediate_channel_sizes), 3)

            self.assertTrue(outputs)

    def test_hf_backbone(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Load a pretrained HF checkpoint as backbone
        config.backbone = "microsoft/resnet-18"
        config.backbone_config = None
        config.use_timm_backbone = False
        config.use_pretrained_backbone = True
        config.backbone_kwargs = {"out_indices": [2, 3, 4]}

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if model_class.__name__ == "DFineForObjectDetection":
                expected_shape = (
                    self.model_tester.batch_size,
                    self.model_tester.num_queries,
                    self.model_tester.num_labels,
                )
                self.assertEqual(outputs.logits.shape, expected_shape)
                # Confirm out_indices was propagated to backbone
                self.assertEqual(len(model.model.backbone.intermediate_channel_sizes), 3)
            else:
                # Confirm out_indices was propagated to backbone
                self.assertEqual(len(model.backbone.intermediate_channel_sizes), 3)

            self.assertTrue(outputs)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        configs_no_init.initializer_bias_prior_prob = 0.2
        bias_value = -1.3863  # log_e ((1 - 0.2) / 0.2)

        failed_cases = []

        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            # Skip the check for the backbone
            for name, module in model.named_modules():
                if module.__class__.__name__ == "DFineConvEncoder":
                    backbone_params = [f"{name}.{key}" for key in module.state_dict()]
                    break

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if ("class_embed" in name and "bias" in name) or "enc_score_head.bias" in name:
                        bias_tensor = torch.full_like(param.data, bias_value)
                        try:
                            torch.testing.assert_close(param.data, bias_tensor, atol=1e-4, rtol=1e-4)
                        except AssertionError:
                            failed_cases.append(
                                f"Parameter {name} of model {model_class} seems not properly initialized. "
                                f"Biases should be initialized to {bias_value}, got {param.data}"
                            )
                    elif (
                        "level_embed" in name
                        or "sampling_offsets.bias" in name
                        or "value_proj" in name
                        or "output_proj" in name
                        or "reference_points" in name
                        or "enc_score_head.weight" in name
                        or ("class_embed" in name and "weight" in name)
                        or name in backbone_params
                    ):
                        continue
                    else:
                        mean = param.data.mean()
                        round_mean = (mean * 1e9).round() / 1e9
                        round_mean = round_mean.item()
                        if round_mean not in [0.0, 1.0]:
                            failed_cases.append(
                                f"Parameter {name} of model {model_class} seems not properly initialized. "
                                f"Mean is {round_mean}, but should be in [0, 1]"
                            )

        message = "\n" + "\n".join(failed_cases)
        self.assertTrue(not failed_cases, message)

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

    @parameterized.expand(["float32", "float16", "bfloat16"])
    @require_torch_accelerator
    @slow
    def test_inference_equivalence_for_static_and_dynamic_anchors(self, dtype_str):
        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype_str]

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        h, w = inputs_dict["pixel_values"].shape[-2:]

        # convert inputs to the desired dtype
        for key, tensor in inputs_dict.items():
            if tensor.dtype == torch.float32:
                inputs_dict[key] = tensor.to(dtype)

        for model_class in self.all_model_classes:
            with tempfile.TemporaryDirectory() as tmpdirname:
                model_class(config).save_pretrained(tmpdirname)
                model_static = model_class.from_pretrained(
                    tmpdirname, anchor_image_size=[h, w], device_map=torch_device, dtype=dtype
                ).eval()
                model_dynamic = model_class.from_pretrained(
                    tmpdirname, anchor_image_size=None, device_map=torch_device, dtype=dtype
                ).eval()

            self.assertIsNotNone(model_static.config.anchor_image_size)
            self.assertIsNone(model_dynamic.config.anchor_image_size)

            with torch.no_grad():
                outputs_static = model_static(**self._prepare_for_class(inputs_dict, model_class))
                outputs_dynamic = model_dynamic(**self._prepare_for_class(inputs_dict, model_class))

            torch.testing.assert_close(
                outputs_static.last_hidden_state, outputs_dynamic.last_hidden_state, rtol=1e-4, atol=1e-4
            )


TOLERANCE = 1e-4


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
@slow
class DFineModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return RTDetrImageProcessor.from_pretrained(CHECKPOINT) if is_vision_available() else None

    def test_inference_object_detection_head(self):
        model = DFineForObjectDetection.from_pretrained(CHECKPOINT).to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_shape_logits = torch.Size((1, 300, model.config.num_labels))
        self.assertEqual(outputs.logits.shape, expected_shape_logits)

        expected_logits = torch.tensor(
            [
                [-3.8098, -4.7725, -5.9945],
                [-5.2975, -9.4991, -6.1654],
                [-5.3502, -3.9532, -6.3631],
            ]
        ).to(torch_device)
        expected_boxes = torch.tensor(
            [
                [0.7678, 0.4148, 0.4644],
                [0.1691, 0.1987, 0.2124],
                [0.2582, 0.5482, 0.4751],
            ]
        ).to(torch_device)

        torch.testing.assert_close(outputs.logits[0, :3, :3], expected_logits, atol=2e-4, rtol=2e-4)

        expected_shape_boxes = torch.Size((1, 300, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        torch.testing.assert_close(outputs.pred_boxes[0, :3, :3], expected_boxes, atol=2e-4, rtol=2e-4)

        # verify postprocessing
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=[image.size[::-1]]
        )[0]

        expected_scores = torch.tensor([0.9642, 0.9542, 0.9536, 0.8548], device=torch_device)
        expected_labels = [15, 65, 15, 57]
        expected_slice_boxes = torch.tensor(
            [
                [1.3186e01, 5.4130e01, 3.1727e02, 4.7212e02],
                [4.0275e01, 7.2975e01, 1.7620e02, 1.1777e02],
                [3.4276e02, 2.3428e01, 6.3998e02, 3.7477e02],
                [5.8418e-01, 1.1794e00, 6.3933e02, 4.7486e02],
            ],
            device=torch_device,
        )
        torch.testing.assert_close(results["scores"][:4], expected_scores, atol=1e-3, rtol=1e-4)
        self.assertSequenceEqual(results["labels"][:4].tolist(), expected_labels)
        torch.testing.assert_close(results["boxes"][:4], expected_slice_boxes[:4], atol=1e-3, rtol=1e-4)
