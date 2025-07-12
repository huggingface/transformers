# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch SAM-HQ model."""

import tempfile
import unittest

import requests

from transformers import (
    SamHQConfig,
    SamHQMaskDecoderConfig,
    SamHQPromptEncoderConfig,
    SamHQVisionConfig,
    SamHQVisionModel,
    pipeline,
)
from transformers.testing_utils import Expectations, cleanup, require_torch, require_torch_sdpa, slow, torch_device
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import SamHQModel, SamHQProcessor


if is_vision_available():
    from PIL import Image


class SamHQVisionModelTester:
    def __init__(
        self,
        parent,
        hidden_size=36,
        intermediate_size=72,
        projection_dim=62,
        output_channels=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=24,
        patch_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-06,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        qkv_bias=True,
        mlp_ratio=4.0,
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=False,
        window_size=14,
        global_attn_indexes=[2, 5, 8, 11],
        num_pos_feats=16,
        mlp_dim=None,
        batch_size=2,
    ):
        self.parent = parent
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.output_channels = output_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.rel_pos_zero_init = rel_pos_zero_init
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.num_pos_feats = num_pos_feats
        self.mlp_dim = mlp_dim
        self.batch_size = batch_size

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def get_config(self):
        return SamHQVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
            initializer_factor=self.initializer_factor,
            output_channels=self.output_channels,
            qkv_bias=self.qkv_bias,
            mlp_ratio=self.mlp_ratio,
            use_abs_pos=self.use_abs_pos,
            use_rel_pos=self.use_rel_pos,
            rel_pos_zero_init=self.rel_pos_zero_init,
            window_size=self.window_size,
            global_attn_indexes=self.global_attn_indexes,
            num_pos_feats=self.num_pos_feats,
            mlp_dim=self.mlp_dim,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def create_and_check_model(self, config, pixel_values):
        model = SamHQVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        output_size = self.image_size // self.patch_size
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_channels, output_size, output_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class SamHQVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SAM's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (SamHQVisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = SamHQVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SamHQVisionConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SAM's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        expected_attention_shape = (
            self.model_tester.batch_size * self.model_tester.num_attention_heads,
            196,
            196,
        )

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

            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-4:]),
                list(expected_attention_shape),
            )

    @unittest.skip(reason="SamVisionModel does not support training")
    def test_training(self):
        pass

    @unittest.skip(reason="SamVisionModel does not support training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="SamVisionModel does not support training")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Hidden_states is tested in create_and_check_model tests")
    def test_hidden_states_output(self):
        pass

    @require_torch_sdpa
    def test_sdpa_can_compile_dynamic(self):
        self.skipTest(reason="SAM model can't be compiled dynamic yet")


class SamHQPromptEncoderTester:
    def __init__(
        self,
        hidden_size=32,
        input_image_size=24,
        patch_size=2,
        mask_input_channels=4,
        num_point_embeddings=4,
        hidden_act="gelu",
    ):
        self.hidden_size = hidden_size
        self.input_image_size = input_image_size
        self.patch_size = patch_size
        self.mask_input_channels = mask_input_channels
        self.num_point_embeddings = num_point_embeddings
        self.hidden_act = hidden_act

    def get_config(self):
        return SamHQPromptEncoderConfig(
            image_size=self.input_image_size,
            patch_size=self.patch_size,
            mask_input_channels=self.mask_input_channels,
            hidden_size=self.hidden_size,
            num_point_embeddings=self.num_point_embeddings,
            hidden_act=self.hidden_act,
        )

    def prepare_config_and_inputs(self):
        dummy_points = floats_tensor([self.batch_size, 3, 2])
        config = self.get_config()

        return config, dummy_points


class SamHQMaskDecoderTester:
    def __init__(
        self,
        hidden_size=32,
        hidden_act="relu",
        mlp_dim=64,
        num_hidden_layers=12,
        num_attention_heads=4,
        attention_downsample_rate=2,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=32,
        layer_norm_eps=1e-6,
        vit_dim=36,
    ):
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_dim = mlp_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_downsample_rate = attention_downsample_rate
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.layer_norm_eps = layer_norm_eps
        self.vit_dim = vit_dim

    def get_config(self):
        return SamHQMaskDecoderConfig(
            hidden_size=self.hidden_size,
            hidden_act=self.hidden_act,
            mlp_dim=self.mlp_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            attention_downsample_rate=self.attention_downsample_rate,
            num_multimask_outputs=self.num_multimask_outputs,
            iou_head_depth=self.iou_head_depth,
            iou_head_hidden_dim=self.iou_head_hidden_dim,
            layer_norm_eps=self.layer_norm_eps,
            vit_dim=self.vit_dim,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        dummy_inputs = {
            "image_embedding": floats_tensor([self.batch_size, self.hidden_size]),
        }
        return config, dummy_inputs


class SamHQModelTester:
    def __init__(
        self,
        parent,
        hidden_size=36,
        intermediate_size=72,
        projection_dim=62,
        output_channels=32,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_channels=3,
        image_size=24,
        patch_size=2,
        hidden_act="gelu",
        layer_norm_eps=1e-06,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        qkv_bias=True,
        mlp_ratio=4.0,
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=False,
        window_size=14,
        global_attn_indexes=[2, 5, 8, 11],
        num_pos_feats=16,
        mlp_dim=None,
        batch_size=2,
    ):
        self.parent = parent
        self.image_size = image_size
        self.patch_size = patch_size
        self.output_channels = output_channels
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.rel_pos_zero_init = rel_pos_zero_init
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.num_pos_feats = num_pos_feats
        self.mlp_dim = mlp_dim
        self.batch_size = batch_size

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

        self.prompt_encoder_tester = SamHQPromptEncoderTester()
        self.mask_decoder_tester = SamHQMaskDecoderTester()

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        vision_config = SamHQVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
            initializer_factor=self.initializer_factor,
            output_channels=self.output_channels,
            qkv_bias=self.qkv_bias,
            mlp_ratio=self.mlp_ratio,
            use_abs_pos=self.use_abs_pos,
            use_rel_pos=self.use_rel_pos,
            rel_pos_zero_init=self.rel_pos_zero_init,
            window_size=self.window_size,
            global_attn_indexes=self.global_attn_indexes,
            num_pos_feats=self.num_pos_feats,
            mlp_dim=self.mlp_dim,
        )

        prompt_encoder_config = self.prompt_encoder_tester.get_config()

        mask_decoder_config = self.mask_decoder_tester.get_config()

        return SamHQConfig(
            vision_config=vision_config,
            prompt_encoder_config=prompt_encoder_config,
            mask_decoder_config=mask_decoder_config,
        )

    def create_and_check_model(self, config, pixel_values):
        model = SamHQModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            # Explicitly pass multimask_output=True
            result = model(pixel_values, multimask_output=True)

        self.parent.assertEqual(result.iou_scores.shape, (self.batch_size, 1, 3))
        self.parent.assertEqual(result.pred_masks.shape[:3], (self.batch_size, 1, 3))

    def create_and_check_get_image_features(self, config, pixel_values):
        model = SamHQModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            image_embeddings = model.get_image_embeddings(pixel_values)
        self.parent.assertEqual(image_embeddings[0][0].shape, (self.output_channels, 12, 12))

    def create_and_check_get_image_and_intermediate_embeddings(self, config, pixel_values):
        model = SamHQModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            image_embeddings, intermediate_embeddings = model.get_image_embeddings(pixel_values)

        self.parent.assertEqual(image_embeddings[0].shape, (self.output_channels, 12, 12))
        self.parent.assertEqual(intermediate_embeddings[0][0].shape, (12, 12, self.hidden_size))

    def create_and_check_get_image_intermediate_embeddings(self, config, pixel_values):
        model = SamHQModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            image_embeddings, intermediate_embeddings = model.get_image_embeddings(pixel_values)

        self.parent.assertIsInstance(intermediate_embeddings, list)
        self.parent.assertTrue(len(intermediate_embeddings) > 0)
        for embedding in intermediate_embeddings:
            self.parent.assertEqual(embedding.shape, (self.batch_size, 12, 12, self.hidden_size))

    def create_and_check_get_image_hidden_states(self, config, pixel_values):
        model = SamHQModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model.vision_encoder(
                pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )

        # after computing the convolutional features
        expected_hidden_states_shape = (self.batch_size, 12, 12, 36)
        self.parent.assertEqual(len(result.hidden_states), self.num_hidden_layers + 1)
        self.parent.assertEqual(result[-1][0].shape, expected_hidden_states_shape)

        with torch.no_grad():
            result = model.vision_encoder(
                pixel_values,
                output_hidden_states=True,
            )
        # after computing the convolutional features
        expected_hidden_states_shape = (self.batch_size, 12, 12, 36)
        self.parent.assertEqual(len(result.hidden_states), self.num_hidden_layers + 1)
        self.parent.assertEqual(result[1][0].shape, expected_hidden_states_shape)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class SamHQModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SAM-HQ's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (SamHQModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": SamHQModel, "mask-generation": SamHQModel} if is_torch_available() else {}
    )
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False
    test_cpu_offload = False
    test_disk_offload_bin = False
    test_disk_offload_safetensors = False

    # TODO: Fix me @Arthur: `run_batch_test` in `tests/test_pipeline_mixin.py` not working
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    def setUp(self):
        self.model_tester = SamHQModelTester(self)
        common_properties = ["initializer_range"]
        self.config_tester = ConfigTester(
            self, config_class=SamHQConfig, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SAM-HQ's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Compile not yet supported in SamHQ models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_get_image_features(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_get_image_features(*config_and_inputs)

    def test_get_image_and_intermediate_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_get_image_and_intermediate_embeddings(*config_and_inputs)

    def test_get_image_intermediate_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_get_image_intermediate_embeddings(*config_and_inputs)

    def test_image_hidden_states(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_get_image_hidden_states(*config_and_inputs)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        expected_vision_attention_shape = (
            self.model_tester.batch_size * self.model_tester.num_attention_heads,
            196,
            196,
        )
        expected_mask_decoder_attention_shape = (self.model_tester.batch_size, 1, 144, 32)

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

            vision_attentions = outputs.vision_attentions
            self.assertEqual(len(vision_attentions), self.model_tester.num_hidden_layers)

            mask_decoder_attentions = outputs.mask_decoder_attentions
            self.assertEqual(len(mask_decoder_attentions), self.model_tester.mask_decoder_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            config.vision_config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            vision_attentions = outputs.vision_attentions
            self.assertEqual(len(vision_attentions), self.model_tester.num_hidden_layers)

            mask_decoder_attentions = outputs.mask_decoder_attentions
            self.assertEqual(len(mask_decoder_attentions), self.model_tester.mask_decoder_tester.num_hidden_layers)

            self.assertListEqual(
                list(vision_attentions[0].shape[-4:]),
                list(expected_vision_attention_shape),
            )

            self.assertListEqual(
                list(mask_decoder_attentions[0].shape[-4:]),
                list(expected_mask_decoder_attention_shape),
            )

    @unittest.skip(reason="SamHQModel does not support training")
    def test_training(self):
        pass

    @unittest.skip(reason="SamHQModel does not support training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="SamHQModel does not support training")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Hidden_states is tested in create_and_check_model tests")
    def test_hidden_states_output(self):
        pass

    def check_pt_tf_outputs(self, tf_outputs, pt_outputs, model_class, tol=5e-5, name="outputs", attributes=None):
        # Use a slightly higher default tol to make the tests non-flaky
        super().check_pt_tf_outputs(tf_outputs, pt_outputs, model_class, tol=tol, name=name, attributes=attributes)

    @slow
    def test_model_from_pretrained(self):
        model_name = "syscv-community/sam-hq-vit-base"
        model = SamHQModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @require_torch_sdpa
    def test_sdpa_can_compile_dynamic(self):
        self.skipTest(reason="SamHQModel can't be compiled dynamic yet")

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        """
        Tests if composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are calles "SDPAAttention".
        In contrast to the above test, this one checks if the "config._attn_implamentation" is a dict after the model
        is loaded, because we manually replicate requested attn implementation on each sub-config when loading.
        See https://github.com/huggingface/transformers/pull/32238 for more info

        The test tries to cover most general cases of composite models, VLMs with vision and text configs. Any model
        that has a different set of sub-configs has to overwrite this test.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname, attn_implementation="sdpa")
                model_sdpa = model_sdpa.eval().to(torch_device)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)

                # Root model determines SDPA support
                attn_impl = "sdpa" if model._supports_sdpa else "eager"

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(model_sdpa.vision_encoder.config._attn_implementation == attn_impl)
                self.assertTrue(model_sdpa.mask_decoder.config._attn_implementation == attn_impl)

                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.vision_encoder.config._attn_implementation == "eager")
                self.assertTrue(model_eager.mask_decoder.config._attn_implementation == "eager")

                # Verify SDPA/eager layer presence
                has_sdpa = False
                for name, submodule in model_sdpa.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        has_sdpa = True
                        break

                if not has_sdpa and attn_impl == "sdpa":
                    raise ValueError("The SDPA model should have SDPA attention layers")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")


def prepare_image():
    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


def prepare_dog_img():
    img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dog-sam.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


@slow
class SamHQModelIntegrationTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        cleanup(torch_device, gc_collect=True)

    def test_inference_mask_generation_no_point(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()
        inputs = processor(images=raw_image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.iou_scores

        masks = outputs.pred_masks[0, 0, 0, 0, :3]
        self.assertTrue(torch.allclose(scores[0][0][-1], torch.tensor(0.4482), atol=2e-4))

        expectations = Expectations(
            {
                (None, None): [-13.1695, -14.6201, -14.8989],
                ("cuda", 8): [-13.1668, -14.6182, -14.8970],
            }
        )
        EXPECTED_MASKS = torch.tensor(expectations.get_expectation()).to(torch_device)
        torch.testing.assert_close(masks, EXPECTED_MASKS, atol=2e-3, rtol=2e-3)

    def test_inference_mask_generation_one_point_one_bb(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()
        input_boxes = [[[650, 900, 1000, 1250]]]
        input_points = [[[820, 1080]]]

        inputs = processor(
            images=raw_image, input_boxes=input_boxes, input_points=input_points, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.iou_scores.squeeze()
        masks = outputs.pred_masks[0, 0, 0, 0, :3]
        self.assertTrue(torch.allclose(scores[-1], torch.tensor(0.9700), atol=2e-4))
        self.assertTrue(
            torch.allclose(masks, torch.tensor([-29.9144, -30.0546, -30.9526]).to(torch_device), atol=3e-2)
        )

    def test_inference_mask_generation_batched_points_batched_images(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()
        input_points = [
            [[[820, 1080]], [[820, 1080]], [[820, 1080]], [[820, 1080]]],
            [[[510, 1080]], [[820, 1080]], [[820, 1080]], [[820, 1080]]],
        ]

        inputs = processor(images=[raw_image, raw_image], input_points=input_points, return_tensors="pt").to(
            torch_device
        )

        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.iou_scores.squeeze()
        masks = outputs.pred_masks[0, 0, 0, 0, :3]

        expectations = Expectations(
            {
                (None, None): [
                    [
                        [0.9195, 0.8316, 0.6614],
                        [0.9195, 0.8316, 0.6614],
                        [0.9195, 0.8316, 0.6614],
                        [0.9195, 0.8316, 0.6614],
                    ],
                    [
                        [0.7598, 0.7388, 0.3110],
                        [0.9195, 0.8317, 0.6614],
                        [0.9195, 0.8317, 0.6614],
                        [0.9195, 0.8317, 0.6614],
                    ],
                ],
                ("cuda", 8): [
                    [
                        [0.9195, 0.8316, 0.6614],
                        [0.9195, 0.8316, 0.6614],
                        [0.9195, 0.8316, 0.6614],
                        [0.9195, 0.8316, 0.6614],
                    ],
                    [
                        [0.7597, 0.7387, 0.3110],
                        [0.9195, 0.8316, 0.6614],
                        [0.9195, 0.8316, 0.6614],
                        [0.9195, 0.8316, 0.6614],
                    ],
                ],
            }
        )
        EXPECTED_SCORES = torch.tensor(expectations.get_expectation()).to(torch_device)

        expectations = Expectations(
            {
                (None, None): [-40.2445, -37.4300, -38.1577],
                ("cuda", 8): [-40.2351, -37.4334, -38.1526],
            }
        )
        EXPECTED_MASKS = torch.tensor(expectations.get_expectation()).to(torch_device)

        torch.testing.assert_close(scores, EXPECTED_SCORES, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(masks, EXPECTED_MASKS, atol=9e-3, rtol=9e-3)

    def test_inference_mask_generation_one_point_one_bb_zero(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()
        input_boxes = [[[620, 900, 1000, 1255]]]
        input_points = [[[820, 1080]]]
        labels = [[0]]

        inputs = processor(
            images=raw_image,
            input_boxes=input_boxes,
            input_points=input_points,
            input_labels=labels,
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.iou_scores.squeeze()

        self.assertTrue(torch.allclose(scores[-1], torch.tensor(0.8680), atol=1e-3))

    def test_inference_mask_generation_with_labels(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")
        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()
        input_points = [[[400, 650]]]
        input_labels = [[1]]

        inputs = processor(
            images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        scores = outputs.iou_scores.squeeze()
        self.assertTrue(torch.allclose(scores[-1], torch.tensor(0.9137), atol=1e-4))

    def test_inference_mask_generation_without_labels(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")
        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()
        input_points = [[[400, 650]]]

        inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        scores = outputs.iou_scores.squeeze()
        self.assertTrue(torch.allclose(scores[-1], torch.tensor(0.9137), atol=1e-3))

    def test_inference_mask_generation_two_points_with_labels(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")
        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()
        input_points = [[[400, 650], [800, 650]]]
        input_labels = [[1, 1]]

        inputs = processor(
            images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        scores = outputs.iou_scores.squeeze()
        self.assertTrue(torch.allclose(scores[-1], torch.tensor(0.8859), atol=1e-3))

    def test_inference_mask_generation_two_points_without_labels(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")
        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()
        input_points = [[[400, 650], [800, 650]]]

        inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        scores = outputs.iou_scores.squeeze()
        self.assertTrue(torch.allclose(scores[-1], torch.tensor(0.8859), atol=1e-3))

    def test_inference_mask_generation_two_points_batched(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()

        input_points = [[[400, 650], [800, 650]], [[400, 650]]]
        input_labels = [[1, 1], [1]]

        inputs = processor(
            images=[raw_image, raw_image],
            input_points=input_points,
            input_labels=input_labels,
            images_kwargs={"point_pad_value": -10},
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.iou_scores.squeeze()
        self.assertTrue(torch.allclose(scores[0][-1], torch.tensor(0.4482), atol=1e-4))
        self.assertTrue(torch.allclose(scores[1][-1], torch.tensor(0.4482), atol=1e-4))

    def test_inference_mask_generation_one_box(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()

        input_boxes = [[[75, 275, 1725, 850]]]

        inputs = processor(images=raw_image, input_boxes=input_boxes, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.iou_scores.squeeze()
        self.assertTrue(torch.allclose(scores[-1], torch.tensor(0.6265), atol=1e-4))

    def test_inference_mask_generation_batched_image_one_point(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()
        raw_dog_image = prepare_dog_img()

        input_points = [[[820, 1080]], [[220, 470]]]

        inputs = processor(images=[raw_image, raw_dog_image], input_points=input_points, return_tensors="pt").to(
            torch_device
        )

        with torch.no_grad():
            outputs = model(**inputs)
        scores_batched = outputs.iou_scores.squeeze()

        input_points = [[[220, 470]]]

        inputs = processor(images=raw_dog_image, input_points=input_points, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)
        scores_single = outputs.iou_scores.squeeze()
        self.assertTrue(torch.allclose(scores_batched[1, :], scores_single, atol=1e-4))

    def test_inference_mask_generation_two_points_point_batch(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()

        input_points = torch.Tensor([[[400, 650]], [[220, 470]]]).cpu()  # fmt: skip

        input_points = input_points.unsqueeze(0)

        inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        iou_scores = outputs.iou_scores.cpu()
        self.assertTrue(iou_scores.shape == (1, 2, 3))
        torch.testing.assert_close(
            iou_scores, torch.tensor([[[0.9889, 0.9508, 0.9137], [0.8070, 0.7934, 0.7932]]]), atol=1e-3, rtol=1e-3
        )

    def test_inference_mask_generation_three_boxes_point_batch(self):
        model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
        processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

        model.to(torch_device)
        model.eval()

        raw_image = prepare_image()

        # fmt: off
        input_boxes = torch.Tensor([[[620, 900, 1000, 1255]], [[75, 275, 1725, 850]],  [[75, 275, 1725, 850]]]).cpu()
        EXPECTED_IOU = torch.tensor([[[0.9850, 0.9730, 0.9726],
         [0.8891, 0.8017, 0.6265],
         [0.8891, 0.8017, 0.6265]]])
        # fmt: on
        input_boxes = input_boxes.unsqueeze(0)

        inputs = processor(raw_image, input_boxes=input_boxes, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        iou_scores = outputs.iou_scores.cpu()
        self.assertTrue(iou_scores.shape == (1, 3, 3))
        torch.testing.assert_close(iou_scores, EXPECTED_IOU, atol=1e-4, rtol=1e-4)

    def test_dummy_pipeline_generation(self):
        generator = pipeline("mask-generation", model="syscv-community/sam-hq-vit-base", device=torch_device)
        raw_image = prepare_image()

        _ = generator(raw_image, points_per_batch=64)
