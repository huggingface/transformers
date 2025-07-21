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
    require_torch_sdpa,
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

    from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VisionModel


if is_vision_available():
    from PIL import Image


class Sam2VisionModelTester:
    def __init__(
        self,
        parent,
        hidden_size=12,
        num_channels=3,
        image_size=128,
        patch_kernel_size=7,
        patch_stride=4,
        patch_padding=3,
        batch_size=2,
        dim_mul=2.0,
        stages=[1, 2, 7, 2],
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
        self.stages = stages
        self.dim_mul = dim_mul
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
            stages=self.stages,
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
        output_size = self.image_size // self.patch_stride // (self.dim_mul * len(self.stages))
        output_channels = self.hidden_size * self.dim_mul * len(self.stages)
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
            expected_num_attentions = sum(self.model_tester.stages)
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            window_size = config.backbone_config.window_spec[0]
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

            expected_num_layers = sum(self.model_tester.stages) + 1
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

    @require_torch_sdpa
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
        dim_mul=2.0,
        stages=[1, 2, 7, 2],
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
        self.dim_mul = dim_mul
        self.stages = stages
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
            dim_mul=self.dim_mul,
            stages=self.stages,
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
        self.parent.assertEqual(result.low_res_masks.shape[:3], (self.batch_size, 1, 3))

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
            expected_num_attentions = sum(self.model_tester.stages)
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.mask_decoder_config.output_attentions = True
            config.vision_config.output_attentions = True
            config.output_attentions = True
            model = model_class._from_config(config, attn_implementation="eager")
            window_size = config.vision_config.backbone_config.window_spec[0]
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
    @require_torch_sdpa
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
        This test is only for inference and runs with `torch_dtype=torch.bfloat16`.
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
                    tmpdirname, torch_dtype=torch.bfloat16, attn_implementation=attn_implementation
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.bfloat16)
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
        model_name = "../sam2_hf_implem/sam2.1_tiny_hf"
        model = Sam2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @require_torch_sdpa
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
        self.model = Sam2Model.from_pretrained("yonigozlan/sam2.1_hiera_tiny_hf").to(torch.float32)
        self.video_model = Sam2VideoModel.from_pretrained("yonigozlan/sam2.1_hiera_tiny_hf").to(torch.float32)
        self.processor = Sam2Processor.from_pretrained("yonigozlan/sam2.1_hiera_tiny_hf")
        self.model.to(torch_device)
        self.model.eval()
        self.video_model.to(torch_device)
        self.video_model.eval()

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
        self.assertEqual(outputs.low_res_masks.shape, (1, 1, 3, 256, 256))
        sorted_indices = torch.argsort(outputs.iou_scores.squeeze(), descending=True)
        scores = outputs.iou_scores.squeeze()[sorted_indices]
        masks_logits = outputs.low_res_masks.squeeze()[sorted_indices][0, :3, :3]

        torch.testing.assert_close(
            scores, torch.tensor([0.9546, 0.4937, 0.0428]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            masks_logits,
            torch.tensor(
                [[-25.0963, -41.5728, -30.8723], [-34.7112, -30.7988, -36.4013], [-25.3061, -37.4575, -33.1899]]
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
        self.assertEqual(outputs.low_res_masks.shape, (1, 1, 1, 256, 256))
        scores = outputs.iou_scores.squeeze((0, 1))
        masks_logits = outputs.low_res_masks.squeeze((0, 1))[0, :3, :3]

        torch.testing.assert_close(scores, torch.tensor([0.9366]).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            masks_logits,
            torch.tensor(
                [[-7.1674, -13.4459, -9.6908], [-10.6038, -9.7242, -12.4059], [-7.4478, -12.4997, -10.5906]]
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
        self.assertEqual(outputs.low_res_masks.shape, (2, 1, 3, 256, 256))

        sorted_indices = torch.argsort(outputs.iou_scores[0].squeeze(), descending=True)
        scores1 = outputs.iou_scores[0].squeeze()[sorted_indices]
        masks_logits1 = outputs.low_res_masks[0].squeeze()[sorted_indices][0, :3, :3]
        sorted_indices = torch.argsort(outputs.iou_scores[1].squeeze(), descending=True)
        scores2 = outputs.iou_scores[1].squeeze()[sorted_indices]
        masks_logits2 = outputs.low_res_masks[1].squeeze()[sorted_indices][0, :3, :3]

        torch.testing.assert_close(
            scores1, torch.tensor([0.9584, 0.4898, 0.0445]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            masks_logits1,
            torch.tensor(
                [[-22.4127, -37.7623, -27.7642], [-31.0563, -27.6730, -32.6308], [-22.4559, -33.8773, -29.5238]]
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
                [[-13.1202, -17.3222, -14.9687], [-16.2375, -12.7737, -17.6353], [-13.5025, -17.1528, -15.6627]]
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
        self.assertEqual(outputs.low_res_masks.shape, (2, 2, 1, 256, 256))

        torch.testing.assert_close(
            outputs.iou_scores,
            torch.tensor([[[0.9499], [0.9718]], [[0.9568], [0.9114]]]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.low_res_masks[:, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-5.9315, -11.3817], [-8.7964, -8.0970]]], [[[-4.8636, -8.8059], [-6.3548, -7.0945]]]],
                    [[[[-13.8652, -19.1238], [-20.2494, -14.1600]]], [[[-8.8231, -10.2768], [-11.3808, -8.7182]]]],
                ],
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
        self.assertEqual(outputs.low_res_masks.shape, (2, 4, 1, 256, 256))

        torch.testing.assert_close(
            outputs.iou_scores,
            torch.tensor([[[0.9873], [0.9265], [0.9495], [0.9207]], [[0.9445], [0.9496], [0.9497], [0.9481]]]).to(
                torch_device
            ),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.low_res_masks[:, :, :, :2, :2],
            torch.tensor(
                [
                    [
                        [[[-7.6887, -11.9033], [-8.8828, -10.4974]]],
                        [[[-17.1057, -23.3219], [-21.0064, -19.4283]]],
                        [[[-20.6077, -29.3705], [-26.1830, -24.1720]]],
                        [[[-19.6094, -28.7768], [-24.4176, -23.2746]]],
                    ],
                    [
                        [[[-18.5219, -23.5192], [-25.1876, -17.2496]]],
                        [[[-20.1199, -25.4224], [-25.7887, -19.1165]]],
                        [[[-21.0868, -24.7951], [-27.5652, -19.2626]]],
                        [[[-20.5161, -22.5330], [-26.0963, -17.7497]]],
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
        mask_input = outputs.low_res_masks[:, :, torch.argmax(outputs.iou_scores)]

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
        self.assertEqual(outputs.low_res_masks.shape, (1, 1, 1, 256, 256))
        scores = outputs.iou_scores.squeeze((0, 1))
        masks_logits = outputs.low_res_masks.squeeze((0, 1))[0, :3, :3]
        torch.testing.assert_close(scores, torch.tensor([0.9736]).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            masks_logits,
            torch.tensor([[-5.4097, -9.7417, -8.4445], [-5.5585, -8.8216, -8.2644], [-5.6046, -9.8751, -9.0067]]).to(
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
        self.assertEqual(outputs.low_res_masks.shape, (1, 1, 1, 256, 256))
        scores = outputs.iou_scores.squeeze((0, 1))
        masks_logits = outputs.low_res_masks.squeeze((0, 1))[0, :3, :3]
        torch.testing.assert_close(scores, torch.tensor([0.9720]).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            masks_logits,
            torch.tensor(
                [[-15.5743, -21.8550, -18.0607], [-17.5526, -17.4155, -23.6521], [-14.4471, -19.4647, -18.6332]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_video_one_point(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_points=[[[[210, 350]]]],
            input_labels=[[[1]]],
        )
        outputs = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=False,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )
        low_res_masks = outputs.consolidated_res_masks
        video_res_masks = outputs.video_res_masks
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-21.4113, -21.4113, -22.9685], [-23.3089, -23.3089, -24.2602], [-27.5700, -27.5700, -27.1607]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_async(
            inference_session=inference_session,
            max_frame_num_to_track=2,
        ):
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-21.4113, -21.4113], [-23.3089, -23.3089]]]],
                    [[[[-20.0937, -20.0937], [-21.2233, -21.2233]]]],
                    [[[[-19.9581, -19.9581], [-21.3028, -21.3028]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_video_one_point_propagate_in_video_directly(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_points=[[[[210, 350]]]],
            input_labels=[[[1]]],
        )
        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_async(
            inference_session=inference_session,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=2,
        ):
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-21.4113, -21.4113], [-23.3089, -23.3089]]]],
                    [[[[-20.0937, -20.0937], [-21.2233, -21.2233]]]],
                    [[[[-19.9581, -19.9581], [-21.3028, -21.3028]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_video_multi_points(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_points=[[[[210, 350], [250, 220]]]],
            input_labels=[[[1, 1]]],
        )
        outputs = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=False,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )
        low_res_masks = outputs.consolidated_res_masks
        video_res_masks = outputs.video_res_masks
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-11.1491, -11.1491, -11.4204], [-11.6524, -11.6524, -11.8057], [-12.7825, -12.7825, -12.6707]],
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_async(
            inference_session=inference_session,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=2,
        ):
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-11.1491, -11.1491], [-11.6524, -11.6524]]]],
                    [[[[-15.3764, -15.3764], [-16.0280, -16.0280]]]],
                    [[[[-15.4271, -15.4271], [-16.3561, -16.3561]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_video_one_bb(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_boxes=[[[[300, 0, 500, 400]]]],
        )
        outputs = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=False,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )
        low_res_masks = outputs.consolidated_res_masks
        video_res_masks = outputs.video_res_masks
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-13.1423, -13.1423, -13.6417], [-13.7748, -13.7748, -14.1142], [-15.1950, -15.1950, -15.1751]],
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_async(
            inference_session=inference_session,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=2,
        ):
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-13.1423, -13.1423], [-13.7748, -13.7748]]]],
                    [[[[-14.9965, -14.9965], [-15.7060, -15.7060]]]],
                    [[[[-15.4546, -15.4546], [-16.1641, -16.1641]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_video_one_point_one_bb(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_boxes=[[[[300, 0, 500, 400]]]],
            input_points=[[[[460, 60]]]],
            input_labels=[[[1]]],
        )
        outputs = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=False,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )
        low_res_masks = outputs.consolidated_res_masks
        video_res_masks = outputs.video_res_masks
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-12.3523, -12.3523, -12.8905], [-13.0603, -13.0603, -13.4075], [-14.6503, -14.6503, -14.5686]],
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_async(
            inference_session=inference_session,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=2,
        ):
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-12.3523, -12.3523], [-13.0603, -13.0603]]]],
                    [[[[-15.8182, -15.8182], [-16.4162, -16.4162]]]],
                    [[[[-15.8911, -15.8911], [-16.5963, -16.5963]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_video_multi_objects_multi_points(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_ids = [2, 3]  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_ids,
            input_points=[[[[200, 300], [230, 250], [275, 175]], [[400, 150]]]],
            input_labels=[[[1, 1, 0], [1]]],
        )
        outputs = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=False,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )
        low_res_masks = outputs.consolidated_res_masks
        video_res_masks = outputs.video_res_masks
        self.assertEqual(low_res_masks.shape, (2, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (2, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[:, 0, :2, :2],  # first object
            torch.tensor(
                [[[-12.6303, -12.6303], [-13.3667, -13.3667]], [[-20.3307, -20.3307], [-22.0473, -22.0473]]],
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_async(
            inference_session=inference_session,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=2,
        ):
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 2, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-12.6303, -12.6303], [-13.3667, -13.3667]]], [[[-20.3307, -20.3307], [-22.0473, -22.0473]]]],
                    [[[[-18.5244, -18.5244], [-19.5828, -19.5828]]], [[[-17.5492, -17.5492], [-19.2211, -19.2211]]]],
                    [[[[-14.2723, -14.2723], [-15.4623, -15.4623]]], [[[-18.3153, -18.3153], [-20.0282, -20.0282]]]],
                ],
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_propagate_video_from_mask_input(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # get input_mask
        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_points=[[[[210, 350], [250, 220]]]],
            input_labels=[[[1, 1]]],
        )
        sam2_video_output = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=True,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )

        # set mask as input
        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_masks=sam2_video_output.video_res_masks,
        )
        sam2_video_output = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=False,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )
        low_res_masks = sam2_video_output.consolidated_res_masks
        video_res_masks = sam2_video_output.video_res_masks
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-10.0000, -10.0000, -10.0000], [-10.0000, -10.0000, -10.0000], [-10.0000, -10.0000, -10.0000]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_async(
            inference_session=inference_session,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=2,
        ):
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-10.0000, -10.0000], [-10.0000, -10.0000]]]],
                    [[[[-18.3571, -18.3571], [-19.2278, -19.2278]]]],
                    [[[[-20.3355, -20.3355], [-21.1817, -21.1817]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_propagate_on_streamed_video(self):
        raw_video = prepare_video()

        inference_session = self.processor.init_video_session(inference_device=torch_device)
        video_res_masks = []
        max_frame_num_to_track = 3
        for frame_idx, frame in enumerate(raw_video):
            if frame_idx >= max_frame_num_to_track:
                break
            inputs = self.processor(images=frame, device=torch_device, return_tensors="pt")
            if frame_idx == 0:
                self.processor.add_inputs_to_inference_session(
                    inference_session,
                    frame_idx=0,
                    obj_ids=1,
                    input_points=[[[[210, 350], [250, 220]]]],
                    input_labels=[[[1, 1]]],
                    original_size=inputs.original_sizes[0],
                )
            sam2_video_output = self.video_model(inference_session=inference_session, frame=inputs.pixel_values[0])
            video_res_masks.append(sam2_video_output.video_res_masks)

        video_res_masks = torch.stack(video_res_masks, dim=0)
        self.assertEqual(
            video_res_masks.shape, (max_frame_num_to_track, 1, 1, raw_video.shape[-3], raw_video.shape[-2])
        )
        torch.testing.assert_close(
            video_res_masks[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-11.1491, -11.1491], [-11.6524, -11.6524]]]],
                    [[[[-15.3764, -15.3764], [-16.0280, -16.0280]]]],
                    [[[[-15.4271, -15.4271], [-16.3561, -16.3561]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_dummy_pipeline_generation(self):
        generator = pipeline("mask-generation", model="../sam2_hf_implem/sam2.1_tiny_hf", device=torch_device)
        raw_image = prepare_image()

        _ = generator(raw_image, points_per_batch=64)
