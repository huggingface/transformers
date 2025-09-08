# coding=utf-8
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
"""Testing suite for the PyTorch SAM2 model."""

import gc
import tempfile
import unittest

import requests

from transformers import (
    Sam2Config,
    Sam2HieraDetConfig,
    Sam2MaskDecoderConfig,
    Sam2Processor,
    Sam2PromptEncoderConfig,
    Sam2VisionConfig,
    pipeline,
)
from transformers.testing_utils import (
    backend_empty_cache,
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available
from transformers.video_utils import load_video

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import Sam2Model, Sam2Processor, Sam2VisionModel


if is_vision_available():
    from PIL import Image


class Sam2VisionModelTester:
    def __init__(
        self,
        parent,
        hidden_size=12,
        embed_dim_per_stage=[12, 24, 48, 96],
        num_attention_heads_per_stage=[1, 2, 4, 8],
        num_channels=3,
        image_size=128,
        patch_kernel_size=7,
        patch_stride=4,
        patch_padding=3,
        batch_size=2,
        blocks_per_stage=[1, 2, 7, 2],
        backbone_channel_list=[96, 48, 24, 12],
        backbone_feature_sizes=[[32, 32], [16, 16], [8, 8]],
        fpn_hidden_size=32,
        is_training=False,
    ):
        self.parent = parent
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.batch_size = batch_size
        self.is_training = is_training
        self.blocks_per_stage = blocks_per_stage
        self.embed_dim_per_stage = embed_dim_per_stage
        self.num_attention_heads_per_stage = num_attention_heads_per_stage
        self.backbone_channel_list = backbone_channel_list
        self.backbone_feature_sizes = backbone_feature_sizes
        self.fpn_hidden_size = fpn_hidden_size

    def get_config(self):
        backbone_config = Sam2HieraDetConfig(
            hidden_size=self.hidden_size,
            num_channels=self.num_channels,
            image_size=self.image_size,
            patch_stride=self.patch_stride,
            patch_kernel_size=self.patch_kernel_size,
            patch_padding=self.patch_padding,
            blocks_per_stage=self.blocks_per_stage,
            embed_dim_per_stage=self.embed_dim_per_stage,
            num_attention_heads_per_stage=self.num_attention_heads_per_stage,
        )
        return Sam2VisionConfig(
            backbone_config=backbone_config,
            backbone_channel_list=self.backbone_channel_list,
            backbone_feature_sizes=self.backbone_feature_sizes,
            fpn_hidden_size=self.fpn_hidden_size,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def create_and_check_model(self, config, pixel_values):
        model = Sam2VisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        output_size = self.image_size // self.patch_stride // (2 * len(self.blocks_per_stage))
        output_channels = self.hidden_size * 2 * len(self.blocks_per_stage)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, output_size, output_size, output_channels)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class Sam2VisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SAM's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (Sam2VisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = Sam2VisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Sam2VisionConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

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

    # Overriding as attention shape depends on window_size
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
            attentions = outputs.attentions
            expected_num_attentions = sum(self.model_tester.blocks_per_stage)
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            window_size = config.backbone_config.window_size_per_stage[0]
            out_dim = config.backbone_config.hidden_size
            patch_stride = config.backbone_config.patch_stride
            num_windows = (
                self.model_tester.batch_size * (config.backbone_config.image_size // (window_size * patch_stride)) ** 2
            )
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)
            self.assertListEqual(
                list(attentions[0].shape[-4:]),
                [num_windows, window_size, window_size, out_dim],
            )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)
            self.assertListEqual(
                list(attentions[0].shape[-4:]),
                [num_windows, window_size, window_size, out_dim],
            )

    # Overriding as attention shape depends on window_size
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class, image_size):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = sum(self.model_tester.blocks_per_stage) + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertListEqual(
                list(hidden_states[0].shape[-4:]),
                [
                    self.model_tester.batch_size,
                    self.model_tester.image_size // self.model_tester.patch_stride,
                    self.model_tester.image_size // self.model_tester.patch_stride,
                    self.model_tester.hidden_size,
                ],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        image_size = self.model_tester.image_size

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class, image_size)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class, image_size)

    # Override as diffence slightly higher than the threshold
    def test_batching_equivalence(self, atol=5e-4, rtol=5e-4):
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    def test_sdpa_can_compile_dynamic(self):
        self.skipTest(reason="SAM model can't be compiled dynamic yet")


class Sam2PromptEncoderTester:
    def __init__(
        self,
        hidden_size=32,
        input_image_size=128,
        patch_size=16,
        mask_input_channels=8,
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
        return Sam2PromptEncoderConfig(
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


class Sam2MaskDecoderTester:
    def __init__(
        self,
        hidden_size=32,
        hidden_act="relu",
        mlp_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        attention_downsample_rate=2,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=32,
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

    def get_config(self):
        return Sam2MaskDecoderConfig(
            hidden_size=self.hidden_size,
            hidden_act=self.hidden_act,
            mlp_dim=self.mlp_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            attention_downsample_rate=self.attention_downsample_rate,
            num_multimask_outputs=self.num_multimask_outputs,
            iou_head_depth=self.iou_head_depth,
            iou_head_hidden_dim=self.iou_head_hidden_dim,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        dummy_inputs = {
            "image_embedding": floats_tensor([self.batch_size, self.hidden_size]),
        }

        return config, dummy_inputs


class Sam2ModelTester:
    def __init__(
        self,
        parent,
        num_channels=3,
        image_size=128,
        hidden_size=12,
        patch_kernel_size=7,
        patch_stride=4,
        patch_padding=3,
        blocks_per_stage=[1, 2, 7, 2],
        embed_dim_per_stage=[12, 24, 48, 96],
        backbone_channel_list=[96, 48, 24, 12],
        backbone_feature_sizes=[[32, 32], [16, 16], [8, 8]],
        fpn_hidden_size=32,
        memory_encoder_hidden_size=32,
        batch_size=2,
        is_training=False,
    ):
        self.parent = parent
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.blocks_per_stage = blocks_per_stage
        self.embed_dim_per_stage = embed_dim_per_stage
        self.backbone_channel_list = backbone_channel_list
        self.backbone_feature_sizes = backbone_feature_sizes
        self.fpn_hidden_size = fpn_hidden_size
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.memory_encoder_hidden_size = memory_encoder_hidden_size

        self.prompt_encoder_tester = Sam2PromptEncoderTester()
        self.mask_decoder_tester = Sam2MaskDecoderTester()

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        backbone_config = Sam2HieraDetConfig(
            hidden_size=self.hidden_size,
            num_channels=self.num_channels,
            image_size=self.image_size,
            patch_stride=self.patch_stride,
            patch_kernel_size=self.patch_kernel_size,
            patch_padding=self.patch_padding,
            blocks_per_stage=self.blocks_per_stage,
            embed_dim_per_stage=self.embed_dim_per_stage,
        )
        vision_config = Sam2VisionConfig(
            backbone_config=backbone_config,
            backbone_channel_list=self.backbone_channel_list,
            backbone_feature_sizes=self.backbone_feature_sizes,
            fpn_hidden_size=self.fpn_hidden_size,
        )

        prompt_encoder_config = self.prompt_encoder_tester.get_config()

        mask_decoder_config = self.mask_decoder_tester.get_config()

        return Sam2Config(
            vision_config=vision_config,
            prompt_encoder_config=prompt_encoder_config,
            mask_decoder_config=mask_decoder_config,
            memory_attention_hidden_size=self.hidden_size,
            memory_encoder_hidden_size=self.memory_encoder_hidden_size,
            image_size=self.image_size,
            mask_downsampler_embed_dim=32,
            memory_fuser_embed_dim=32,
            memory_attention_num_layers=1,
            memory_attention_feed_forward_hidden_size=32,
        )

    def create_and_check_model(self, config, pixel_values):
        model = Sam2Model(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        self.parent.assertEqual(result.iou_scores.shape, (self.batch_size, 1, 3))
        self.parent.assertEqual(result.pred_masks.shape[:3], (self.batch_size, 1, 3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class Sam2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SAM's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (Sam2Model,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": Sam2Model, "mask-generation": Sam2Model} if is_torch_available() else {}
    )
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_torchscript = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Sam2ModelTester(self)
        common_properties = ["initializer_range"]
        self.config_tester = ConfigTester(
            self, config_class=Sam2Config, has_text_modality=False, common_properties=common_properties
        )

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

    # Overriding as attention shape depends on window_size
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
            attentions = outputs.vision_attentions
            expected_num_attentions = sum(self.model_tester.blocks_per_stage)
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.mask_decoder_config.output_attentions = True
            config.vision_config.output_attentions = True
            config.output_attentions = True
            model = model_class._from_config(config, attn_implementation="eager")
            window_size = config.vision_config.backbone_config.window_size_per_stage[0]
            out_dim = self.model_tester.hidden_size
            patch_stride = self.model_tester.patch_stride
            num_windows = (
                self.model_tester.batch_size * (self.model_tester.image_size // (window_size * patch_stride)) ** 2
            )
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.vision_attentions
            self.assertEqual(len(attentions), expected_num_attentions)
            self.assertListEqual(
                list(attentions[0].shape[-4:]),
                [num_windows, window_size, window_size, out_dim],
            )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.vision_attentions
            self.assertEqual(len(attentions), expected_num_attentions)
            self.assertListEqual(
                list(attentions[0].shape[-4:]),
                [num_windows, window_size, window_size, out_dim],
            )

    # Override as Sam2Model has different sub-modules

    def test_sdpa_can_dispatch_composite_models(self):
        """
        Tests if composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are called "SDPAAttention".
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

                vision_encoder_sdpa = getattr(model_sdpa, "vision_encoder")
                mask_decoder_sdpa = getattr(model_sdpa, "mask_decoder")

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(mask_decoder_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(vision_encoder_sdpa.config._attn_implementation == "sdpa")

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(getattr(model_eager, "mask_decoder").config._attn_implementation == "eager")
                self.assertTrue(getattr(model_eager, "vision_encoder").config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if (
                        class_name.endswith("Attention")
                        and getattr(submodule, "config", None)
                        and submodule.config._attn_implementation == "sdpa"
                    ):
                        raise ValueError("The eager model should not have SDPA attention layers")

    # Override as Sam2Model doesn't have hidden states
    def flash_attn_inference_equivalence(self, attn_implementation: str, padding_side: str):
        r"""
        Tests the equivalence between the eager and flash attention implementations.
        This test is only for inference and runs with `dtype=torch.bfloat16`.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_model_classes:
            if (attn_implementation == "flash_attention_2" and not model_class._supports_flash_attn_2) or (
                attn_implementation == "flash_attention_3" and not model_class._supports_flash_attn_3
            ):
                self.skipTest(f"{model_class.__name__} does not support {attn_implementation}")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, dtype=torch.bfloat16, attn_implementation=attn_implementation
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, dtype=torch.bfloat16)
                model.to(torch_device)

                dummy_input = inputs_dict[model.main_input_name][:1]
                if dummy_input.dtype in [torch.float32, torch.float16]:
                    dummy_input = dummy_input.to(torch.bfloat16)

                dummy_attention_mask = inputs_dict.get("attention_mask", None)

                if dummy_attention_mask is not None:
                    dummy_attention_mask = dummy_attention_mask[:1]
                    if padding_side == "left":
                        dummy_attention_mask[:, 1:] = 1
                        dummy_attention_mask[:, :1] = 0
                    else:
                        dummy_attention_mask[:, :-1] = 1
                        dummy_attention_mask[:, -1:] = 0
                if model.config.is_encoder_decoder:
                    decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)[:1]

                    outputs = model(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                    outputs_fa = model_fa(dummy_input, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
                else:
                    outputs = model(dummy_input, output_hidden_states=True)
                    outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = outputs.vision_hidden_states[-1]
                logits_fa = outputs_fa.vision_hidden_states[-1]

                assert torch.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)

                if model.config.is_encoder_decoder:
                    other_inputs = {
                        "decoder_input_ids": decoder_input_ids,
                        "decoder_attention_mask": dummy_attention_mask,
                        "output_hidden_states": True,
                    }
                    if dummy_attention_mask is not None:
                        other_inputs["attention_mask"] = dummy_attention_mask

                    outputs = model(dummy_input, **other_inputs)
                    outputs_fa = model_fa(dummy_input, **other_inputs)
                else:
                    other_inputs = {
                        "output_hidden_states": True,
                    }
                    if dummy_attention_mask is not None:
                        other_inputs["attention_mask"] = dummy_attention_mask

                    outputs = model(dummy_input, **other_inputs)
                    outputs_fa = model_fa(dummy_input, **other_inputs)

                logits = outputs.vision_hidden_states[-1]
                logits_fa = outputs_fa.vision_hidden_states[-1]

                if padding_side == "left":
                    assert torch.allclose(logits_fa[1:], logits[1:], atol=4e-2, rtol=4e-2)

                    # check with inference + dropout
                    model.train()
                    _ = model_fa(dummy_input, **other_inputs)
                else:
                    assert torch.allclose(logits_fa[:-1], logits[:-1], atol=4e-2, rtol=4e-2)

    # Override as diffence slightly higher than the threshold
    def test_batching_equivalence(self, atol=5e-4, rtol=5e-4):
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    @unittest.skip(reason="Sam2Model does not support training")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Hidden_states is tested in sub modules tests")
    def test_hidden_states_output(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/sam2.1-hiera-tiny"
        model = Sam2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    def test_sdpa_can_compile_dynamic(self):
        self.skipTest(reason="SAM2 model can't be compiled dynamic yet")


def prepare_image():
    img_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


def prepare_groceries_image():
    img_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/groceries.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


def prepare_dog_img():
    img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dog-sam.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


def prepare_video():
    video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
    raw_video, _ = load_video(video_url)
    return raw_video


@slow
class Sam2ModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").to(torch.float32)
        self.processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")
        self.model.to(torch_device)
        self.model.eval()

    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        gc.collect()
        backend_empty_cache(torch_device)

    def test_inference_mask_generation_one_point_multimask(self):
        raw_image = prepare_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]

        inputs = self.processor(
            images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        self.assertEqual(outputs.iou_scores.shape, (1, 1, 3))
        self.assertEqual(outputs.pred_masks.shape, (1, 1, 3, 256, 256))
        sorted_indices = torch.argsort(outputs.iou_scores.squeeze(), descending=True)
        scores = outputs.iou_scores.squeeze()[sorted_indices]
        masks_logits = outputs.pred_masks.squeeze()[sorted_indices][0, :3, :3]
        torch.testing.assert_close(
            scores, torch.tensor([0.9547, 0.4932, 0.0427]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            masks_logits,
            torch.tensor(
                [[-24.9288, -41.7466, -31.0128], [-34.5113, -31.1054, -36.5913], [-25.2597, -37.5912, -33.4030]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_one_point_no_multimask(self):
        raw_image = prepare_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]

        inputs = self.processor(
            images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        self.assertEqual(outputs.iou_scores.shape, (1, 1, 1))
        self.assertEqual(outputs.pred_masks.shape, (1, 1, 1, 256, 256))
        scores = outputs.iou_scores.squeeze((0, 1))
        masks_logits = outputs.pred_masks.squeeze((0, 1))[0, :3, :3]
        torch.testing.assert_close(scores, torch.tensor([0.9364]).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            masks_logits,
            torch.tensor(
                [[-7.0462, -13.3857, -9.6419], [-10.4565, -9.7174, -12.3528], [-7.3704, -12.4391, -10.5539]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_batched_images_multi_points(self):
        raw_image1 = prepare_image()
        raw_image2 = prepare_dog_img()
        input_points = [[[[500, 375]]], [[[770, 200], [730, 120]]]]
        input_labels = [[[1]], [[1, 0]]]

        inputs = self.processor(
            images=[raw_image1, raw_image2], input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        self.assertEqual(outputs.iou_scores.shape, (2, 1, 3))
        self.assertEqual(outputs.pred_masks.shape, (2, 1, 3, 256, 256))

        sorted_indices = torch.argsort(outputs.iou_scores[0].squeeze(), descending=True)
        scores1 = outputs.iou_scores[0].squeeze()[sorted_indices]
        masks_logits1 = outputs.pred_masks[0].squeeze()[sorted_indices][0, :3, :3]
        sorted_indices = torch.argsort(outputs.iou_scores[1].squeeze(), descending=True)
        scores2 = outputs.iou_scores[1].squeeze()[sorted_indices]
        masks_logits2 = outputs.pred_masks[1].squeeze()[sorted_indices][0, :3, :3]
        torch.testing.assert_close(
            scores1, torch.tensor([0.9586, 0.4913, 0.0448]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            masks_logits1,
            torch.tensor(
                [[-22.2555, -37.9250, -27.8928], [-30.8681, -27.9519, -32.8032], [-22.4133, -33.9966, -29.7111]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        torch.testing.assert_close(
            scores2, torch.tensor([0.9504, 0.8117, 0.7426]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            masks_logits2,
            torch.tensor(
                [[-13.1182, -17.3217, -14.9651], [-16.2372, -12.7739, -17.6346], [-13.5013, -17.1549, -15.6614]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_batched_images_batched_points_multi_points(self):
        raw_image1 = prepare_image()
        raw_image2 = prepare_groceries_image()
        input_points = [[[[500, 375]], [[650, 750]]], [[[400, 300]], [[630, 300], [550, 300]]]]
        input_labels = [[[1], [1]], [[1], [1, 1]]]
        inputs = self.processor(
            images=[raw_image1, raw_image2], input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(torch_device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        self.assertEqual(outputs.iou_scores.shape, (2, 2, 1))
        self.assertEqual(outputs.pred_masks.shape, (2, 2, 1, 256, 256))
        torch.testing.assert_close(
            outputs.iou_scores,
            torch.tensor([[[0.9500], [0.9718]], [[0.9568], [0.9114]]]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[:, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-5.8131, -11.3020], [-8.6487, -8.0690]]], [[[-4.7731, -8.7606], [-6.2399, -7.0738]]]],
                    [[[[-13.8661, -19.1254], [-20.2477, -14.1636]]], [[[-8.8229, -10.2760], [-11.3797, -8.7189]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_batched_images_batched_boxes(self):
        raw_image1 = prepare_image()
        raw_image2 = prepare_groceries_image()
        input_boxes = [
            [[75, 275, 1725, 850], [425, 600, 700, 875], [1375, 550, 1650, 800], [1240, 675, 1400, 750]],
            [[450, 170, 520, 350], [350, 190, 450, 350], [500, 170, 580, 350], [580, 170, 640, 350]],
        ]
        inputs = self.processor(images=[raw_image1, raw_image2], input_boxes=input_boxes, return_tensors="pt").to(
            torch_device
        )
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        self.assertEqual(outputs.iou_scores.shape, (2, 4, 1))
        self.assertEqual(outputs.pred_masks.shape, (2, 4, 1, 256, 256))
        torch.testing.assert_close(
            outputs.iou_scores,
            torch.tensor([[[0.9873], [0.9264], [0.9496], [0.9208]], [[0.9445], [0.9496], [0.9497], [0.9481]]]).to(
                torch_device
            ),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[:, :, :, :2, :2],
            torch.tensor(
                [
                    [
                        [[[-7.6204, -11.9286], [-8.7747, -10.5662]]],
                        [[[-17.1070, -23.4025], [-20.9608, -19.5600]]],
                        [[[-20.5766, -29.4410], [-26.0739, -24.3225]]],
                        [[[-19.7201, -29.0836], [-24.4915, -23.6377]]],
                    ],
                    [
                        [[[-18.5259, -23.5202], [-25.1906, -17.2518]]],
                        [[[-20.1214, -25.4215], [-25.7877, -19.1169]]],
                        [[[-21.0878, -24.7938], [-27.5625, -19.2650]]],
                        [[[-20.5210, -22.5343], [-26.0968, -17.7544]]],
                    ],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_from_existing_points_and_mask(self):
        raw_image = prepare_image()
        input_points = [[[[500, 375]]]]
        input_labels = [[[1]]]
        original_inputs = self.processor(
            images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(torch_device)
        with torch.no_grad():
            outputs = self.model(**original_inputs)

        # best mask to use as input for new points
        mask_input = outputs.pred_masks[:, :, torch.argmax(outputs.iou_scores)]

        new_input_points = [[[[500, 375], [1125, 625]]]]
        new_input_labels = [[[1, 1]]]
        inputs = self.processor(
            input_points=new_input_points,
            input_labels=new_input_labels,
            original_sizes=original_inputs["original_sizes"],
            return_tensors="pt",
        ).to(torch_device)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                input_masks=mask_input,
                image_embeddings=outputs.image_embeddings,
                multimask_output=False,
            )

        self.assertEqual(outputs.iou_scores.shape, (1, 1, 1))
        self.assertEqual(outputs.pred_masks.shape, (1, 1, 1, 256, 256))
        scores = outputs.iou_scores.squeeze((0, 1))
        masks_logits = outputs.pred_masks.squeeze((0, 1))[0, :3, :3]
        torch.testing.assert_close(scores, torch.tensor([0.9738]).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            masks_logits,
            torch.tensor([[-5.3899, -9.7908, -8.4931], [-5.5144, -8.8731, -8.3000], [-5.5976, -9.9249, -9.0761]]).to(
                torch_device
            ),
            atol=1e-4,
            rtol=1e-4,
        )

        # with negative point
        new_input_points = [[[[500, 375], [1125, 625]]]]
        new_input_labels = [[[1, 0]]]
        inputs = self.processor(
            input_points=new_input_points,
            input_labels=new_input_labels,
            original_sizes=original_inputs["original_sizes"],
            return_tensors="pt",
        ).to(torch_device)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                input_masks=mask_input,
                image_embeddings=outputs.image_embeddings,
                multimask_output=False,
            )
        self.assertEqual(outputs.iou_scores.shape, (1, 1, 1))
        self.assertEqual(outputs.pred_masks.shape, (1, 1, 1, 256, 256))
        scores = outputs.iou_scores.squeeze((0, 1))
        masks_logits = outputs.pred_masks.squeeze((0, 1))[0, :3, :3]
        torch.testing.assert_close(scores, torch.tensor([0.9719]).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            masks_logits,
            torch.tensor(
                [[-15.5081, -21.8641, -18.0479], [-17.4401, -17.4754, -23.6469], [-14.3975, -19.4346, -18.5884]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_dummy_pipeline_generation(self):
        generator = pipeline("mask-generation", model="facebook/sam2.1-hiera-tiny", device=torch_device)
        raw_image = prepare_image()

        _ = generator(raw_image, points_per_batch=64)
