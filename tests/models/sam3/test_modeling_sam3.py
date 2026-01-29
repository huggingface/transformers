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
"""Testing suite for the PyTorch SAM3 model."""

import gc
import platform
import tempfile
import unittest

import requests
import torch

from transformers.models.sam3.configuration_sam3 import Sam3MaskDecoderConfig
from transformers.models.sam3.modeling_sam3 import Sam3MaskDecoder
from transformers.testing_utils import (
    backend_empty_cache,
    require_deterministic_for_xpu,
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


class Sam3MaskDecoderUnitTest(unittest.TestCase):
    def setUp(self):
        self.config = Sam3MaskDecoderConfig(hidden_size=32, num_multiscale_features=3, decoder_num_layers=2)
        self.decoder = Sam3MaskDecoder(self.config)
        self.device = torch.device("cpu")

    def test_single_scale_forward(self):
        import torch

        batch_size = 2
        C, H, W = self.config.hidden_size, 16, 16
        img_embed = torch.randn(batch_size, C, H, W).to(self.device)
        decoder_queries = torch.randn(batch_size, 4, C).to(self.device)
        encoder_hidden_states = torch.randn(batch_size, H * W, C).to(self.device)
        outputs = self.decoder(
            decoder_queries,
            img_embed,
            encoder_hidden_states=encoder_hidden_states,
        )
        self.assertIsNotNone(outputs.pred_masks)

    def test_multi_scale_forward(self):
        import torch

        batch_size = 2
        C, H, W = self.config.hidden_size, 16, 16
        img_embeds = [torch.randn(batch_size, C, H, W).to(self.device) for _ in range(3)]
        decoder_queries = torch.randn(batch_size, 4, C).to(self.device)
        encoder_hidden_states = torch.randn(batch_size, H * W, C).to(self.device)
        outputs = self.decoder(
            decoder_queries,
            img_embeds,
            encoder_hidden_states=encoder_hidden_states,
        )
        self.assertIsNotNone(outputs.pred_masks)


if is_torch_available():
    import torch
    from torch import nn

    from transformers.models.sam3.configuration_sam3 import (
        Sam3Config,
        Sam3DETRDecoderConfig,
        Sam3DETREncoderConfig,
        Sam3GeometryEncoderConfig,
        Sam3MaskDecoderConfig,
        Sam3VisionConfig,
        Sam3ViTConfig,
    )
    from transformers.models.sam3.modeling_sam3 import Sam3Model, Sam3VisionModel
    from transformers.models.sam3.processing_sam3 import Sam3Processor


if is_vision_available():
    from PIL import Image


class Sam3VisionModelTester:
    def __init__(
        self,
        parent,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        num_channels=3,
        image_size=224,
        patch_size=14,
        window_size=8,
        global_attn_indexes=None,
        fpn_hidden_size=32,
        scale_factors=None,
        batch_size=2,
        is_training=False,
    ):
        if global_attn_indexes is None:
            global_attn_indexes = [0, 1]
        if scale_factors is None:
            scale_factors = [4.0, 2.0, 1.0, 0.5]

        self.parent = parent
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.fpn_hidden_size = fpn_hidden_size
        self.scale_factors = scale_factors
        self.batch_size = batch_size
        self.is_training = is_training

    def get_config(self):
        backbone_config = Sam3ViTConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            num_channels=self.num_channels,
            image_size=self.image_size,
            patch_size=self.patch_size,
            window_size=self.window_size,
            global_attn_indexes=self.global_attn_indexes,
        )
        return Sam3VisionConfig(
            backbone_config=backbone_config,
            fpn_hidden_size=self.fpn_hidden_size,
            scale_factors=self.scale_factors,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def create_and_check_model(self, config, pixel_values):
        model = Sam3VisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)

        # Check FPN outputs
        self.parent.assertEqual(len(result.fpn_hidden_states), len(self.scale_factors))
        self.parent.assertEqual(len(result.fpn_position_encoding), len(self.scale_factors))

        # Check last hidden state shape
        expected_seq_len = (self.image_size // self.patch_size) ** 2
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, expected_seq_len, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
@unittest.skipIf(
    platform.system() == "Windows", "safetensors serialization is not supported on Windows for this test."
)
class Sam3VisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Tests for SAM3 Vision Model (ViT backbone + FPN neck).
    """

    all_model_classes = (Sam3VisionModel,) if is_torch_available() else ()

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = Sam3VisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Sam3VisionConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    @unittest.skip(reason="SAM3's vision encoder does not use inputs_embeds")
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
        # Force eager attention to support output attentions
        config._attn_implementation = "eager"

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager")
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # Check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            config.backbone_config.output_attentions = True

            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # For windowed attention, check the attention shape
            # Attention shape: (batch_size, num_heads, seq_len, seq_len) for global attention
            # or windowed shape for local attention
            self.assertIsNotNone(attentions[0])

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            # SAM3VisionModel doesn't return hidden_states in the same way as SAM2
            # It returns last_hidden_state, fpn_hidden_states, and fpn_position_encoding
            self.assertIsNotNone(outputs.last_hidden_state)
            self.assertIsNotNone(outputs.fpn_hidden_states)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            check_hidden_states_output(inputs_dict, config, model_class)

    def test_batching_equivalence(self, atol=5e-4, rtol=5e-4):
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    @unittest.skip(reason="SAM3 model can't be compiled dynamic yet")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="SAM3VisionModel has FPN channel mismatch with flex attention")
    def test_flex_attention_with_grads(self):
        pass


class Sam3ModelTester:
    def __init__(
        self,
        parent,
        num_channels=3,
        image_size=224,  # Keep reasonable size: 224 = 16 * 14
        hidden_size=32,
        patch_size=14,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        window_size=8,  # 224/14 = 16 patches, 16/2 = 8 per window
        global_attn_indexes=None,
        fpn_hidden_size=32,
        scale_factors=None,
        geometry_encoder_hidden_size=32,
        geometry_encoder_num_layers=1,  # Reduced from 2 to 1
        detr_encoder_hidden_size=32,
        detr_encoder_num_layers=1,  # Reduced from 2 to 1
        detr_decoder_hidden_size=32,
        detr_decoder_num_layers=1,  # Reduced from 2 to 1
        detr_decoder_num_queries=5,  # Reduced from 10 to 5
        mask_decoder_hidden_size=32,
        batch_size=2,
        is_training=False,
    ):
        if global_attn_indexes is None:
            global_attn_indexes = [0, 1]
        if scale_factors is None:
            scale_factors = [2.0, 1.0]  # Just 2 scales to reduce params

        self.parent = parent
        self.num_channels = num_channels
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.fpn_hidden_size = fpn_hidden_size
        self.scale_factors = scale_factors
        self.batch_size = batch_size
        self.is_training = is_training

        # Geometry encoder
        self.geometry_encoder_hidden_size = geometry_encoder_hidden_size
        self.geometry_encoder_num_layers = geometry_encoder_num_layers

        # DETR encoder/decoder
        self.detr_encoder_hidden_size = detr_encoder_hidden_size
        self.detr_encoder_num_layers = detr_encoder_num_layers
        self.detr_decoder_hidden_size = detr_decoder_hidden_size
        self.detr_decoder_num_layers = detr_decoder_num_layers
        self.detr_decoder_num_queries = detr_decoder_num_queries

        # Mask decoder
        self.mask_decoder_hidden_size = mask_decoder_hidden_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        # Simple text input (will be processed by text encoder)
        input_ids = torch.randint(0, 1000, (self.batch_size, 16), device=torch_device)
        attention_mask = torch.ones_like(input_ids)

        config = self.get_config()

        return config, pixel_values, input_ids, attention_mask

    def get_config(self):
        backbone_config = Sam3ViTConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            num_channels=self.num_channels,
            image_size=self.image_size,
            patch_size=self.patch_size,
            window_size=self.window_size,
            global_attn_indexes=self.global_attn_indexes,
        )

        vision_config = Sam3VisionConfig(
            backbone_config=backbone_config,
            fpn_hidden_size=self.fpn_hidden_size,
            scale_factors=self.scale_factors,
        )

        # Small text config for testing (instead of default full CLIP model)
        text_config = {
            "vocab_size": 1000,  # Keep at 1000 for stability
            "hidden_size": 32,
            "intermediate_size": 64,
            "projection_dim": 32,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": 4,
            "max_position_embeddings": 32,  # Keep at 32 for stability
            "hidden_act": "gelu",
        }

        geometry_encoder_config = Sam3GeometryEncoderConfig(
            hidden_size=self.geometry_encoder_hidden_size,
            num_layers=self.geometry_encoder_num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            mask_fuser_hidden_size=self.geometry_encoder_hidden_size,  # Match hidden_size to reduce params
            mask_fuser_num_layers=1,  # Reduce from default 2 to 1
        )

        detr_encoder_config = Sam3DETREncoderConfig(
            hidden_size=self.detr_encoder_hidden_size,
            num_layers=self.detr_encoder_num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
        )

        detr_decoder_config = Sam3DETRDecoderConfig(
            hidden_size=self.detr_decoder_hidden_size,
            num_layers=self.detr_decoder_num_layers,
            num_queries=self.detr_decoder_num_queries,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
        )

        mask_decoder_config = Sam3MaskDecoderConfig(
            hidden_size=self.mask_decoder_hidden_size,
            num_upsampling_stages=2,  # Reduced from 3 to 2
        )

        return Sam3Config(
            vision_config=vision_config,
            text_config=text_config,
            geometry_encoder_config=geometry_encoder_config,
            detr_encoder_config=detr_encoder_config,
            detr_decoder_config=detr_decoder_config,
            mask_decoder_config=mask_decoder_config,
        )

    def create_and_check_model(self, config, pixel_values, input_ids, attention_mask):
        model = Sam3Model(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        # Check output shapes
        self.parent.assertIsNotNone(result.pred_masks)
        self.parent.assertIsNotNone(result.pred_boxes)
        self.parent.assertIsNotNone(result.pred_logits)

        # Masks should be [batch_size, num_queries, H, W]
        self.parent.assertEqual(result.pred_masks.shape[0], self.batch_size)
        self.parent.assertEqual(result.pred_masks.shape[1], self.detr_decoder_num_queries)

        # Boxes should be [batch_size, num_queries, 4]
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.detr_decoder_num_queries, 4))

        # Logits should be [batch_size, num_queries]
        self.parent.assertEqual(result.pred_logits.shape, (self.batch_size, self.detr_decoder_num_queries))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, input_ids, attention_mask = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
@unittest.skipIf(
    platform.system() == "Windows", "safetensors serialization is not supported on Windows for this test."
)
class Sam3ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Tests for SAM3 full model.
    """

    all_model_classes = (Sam3Model,) if is_torch_available() else ()
    pipeline_model_mapping = {"mask-generation": Sam3Model} if is_torch_available() else {}

    test_resize_embeddings = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Sam3ModelTester(self)
        common_properties = ["initializer_range"]
        self.config_tester = ConfigTester(
            self, config_class=Sam3Config, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SAM3 does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            # Vision encoder has input embeddings
            self.assertIsInstance(model.vision_encoder.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_batching_equivalence(self, atol=5e-4, rtol=5e-4):
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    # Override as SAM3Model has component-specific attention outputs
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

            # Check that we have the component-specific attention outputs
            # Note: Some may be empty tuples if attentions aren't collected for that component
            self.assertIsNotNone(outputs.vision_attentions)
            self.assertIsNotNone(outputs.detr_encoder_attentions)
            self.assertIsNotNone(outputs.detr_decoder_attentions)
            self.assertIsNotNone(outputs.mask_decoder_attentions)

            # Check vision attentions (from ViT backbone) - should be properly collected
            if outputs.vision_attentions:
                vision_attentions = outputs.vision_attentions
                self.assertEqual(len(vision_attentions), self.model_tester.num_hidden_layers)

            # Check that at least vision attentions are present (others may require different collection mechanism)
            self.assertTrue(
                len(outputs.vision_attentions) > 0,
                "At least vision attentions should be collected when output_attentions=True",
            )

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            for k in config.sub_configs:
                if getattr(config, k) is not None:
                    getattr(config, k).output_attentions = True

            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            # Verify again with config-based setting
            self.assertIsNotNone(outputs.vision_attentions)
            self.assertIsNotNone(outputs.detr_encoder_attentions)
            self.assertIsNotNone(outputs.detr_decoder_attentions)
            self.assertIsNotNone(outputs.mask_decoder_attentions)

    # Override as SAM3Model has component-specific attention/hidden state outputs
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for k in config.sub_configs:
            if getattr(config, k) is not None:
                getattr(config, k).output_hidden_states = True
                getattr(config, k).output_attentions = True

        config.output_hidden_states = True
        config.output_attentions = True
        config._attn_implementation = "eager"

        # Use first model class
        model_class = self.all_model_classes[0]
        model = model_class._from_config(config, attn_implementation="eager")
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)
        outputs = model(**inputs)

        output = outputs[0]

        # SAM3 has component-specific hidden states and attentions
        # Check vision hidden states and attentions
        if outputs.vision_hidden_states is not None and len(outputs.vision_hidden_states) > 0:
            vision_hidden_states = outputs.vision_hidden_states[0]
            vision_hidden_states.retain_grad()

        if outputs.vision_attentions is not None and len(outputs.vision_attentions) > 0:
            vision_attentions = outputs.vision_attentions[0]
            vision_attentions.retain_grad()

        # Check DETR encoder hidden states and attentions
        if outputs.encoder_hidden_states is not None and len(outputs.encoder_hidden_states) > 0:
            encoder_hidden_states = outputs.encoder_hidden_states[0]
            encoder_hidden_states.retain_grad()

        if outputs.detr_encoder_attentions is not None and len(outputs.detr_encoder_attentions) > 0:
            detr_encoder_attentions = outputs.detr_encoder_attentions[0]
            detr_encoder_attentions.retain_grad()

        # Check DETR decoder hidden states and attentions
        if outputs.decoder_hidden_states is not None and len(outputs.decoder_hidden_states) > 0:
            decoder_hidden_states = outputs.decoder_hidden_states[0]
            decoder_hidden_states.retain_grad()

        if outputs.detr_decoder_attentions is not None and len(outputs.detr_decoder_attentions) > 0:
            detr_decoder_attentions = outputs.detr_decoder_attentions[0]
            detr_decoder_attentions.retain_grad()

        # Check mask decoder attentions
        if outputs.mask_decoder_attentions is not None and len(outputs.mask_decoder_attentions) > 0:
            mask_decoder_attentions = outputs.mask_decoder_attentions[0]
            mask_decoder_attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        # Check gradients are not None
        if outputs.vision_hidden_states is not None and len(outputs.vision_hidden_states) > 0:
            self.assertIsNotNone(vision_hidden_states.grad)

        if outputs.vision_attentions is not None and len(outputs.vision_attentions) > 0:
            self.assertIsNotNone(vision_attentions.grad)

        if outputs.encoder_hidden_states is not None and len(outputs.encoder_hidden_states) > 0:
            self.assertIsNotNone(encoder_hidden_states.grad)

        if outputs.detr_encoder_attentions is not None and len(outputs.detr_encoder_attentions) > 0:
            self.assertIsNotNone(detr_encoder_attentions.grad)

        if outputs.decoder_hidden_states is not None and len(outputs.decoder_hidden_states) > 0:
            self.assertIsNotNone(decoder_hidden_states.grad)

        if outputs.detr_decoder_attentions is not None and len(outputs.detr_decoder_attentions) > 0:
            self.assertIsNotNone(detr_decoder_attentions.grad)

        if outputs.mask_decoder_attentions is not None and len(outputs.mask_decoder_attentions) > 0:
            self.assertIsNotNone(mask_decoder_attentions.grad)

    def test_hidden_states_output(self):
        """Test that SAM3 properly outputs component-specific hidden states."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            # Enable hidden states output
            config.output_hidden_states = True
            for k in config.sub_configs:
                if getattr(config, k) is not None:
                    getattr(config, k).output_hidden_states = True

            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            # SAM3 has component-specific hidden states
            # Check vision hidden states
            if outputs.vision_hidden_states is not None:
                vision_hidden_states = outputs.vision_hidden_states
                self.assertIsInstance(vision_hidden_states, (list, tuple))
                # Vision encoder outputs hidden states from each layer
                expected_num_vision_layers = self.model_tester.num_hidden_layers + 1  # +1 for embeddings
                self.assertEqual(len(vision_hidden_states), expected_num_vision_layers)

            # Check DETR encoder hidden states (stored as encoder_hidden_states)
            if outputs.encoder_hidden_states is not None:
                encoder_hidden_states = outputs.encoder_hidden_states
                self.assertIsInstance(encoder_hidden_states, (list, tuple))

            # Check DETR decoder hidden states (stored as decoder_hidden_states)
            if outputs.decoder_hidden_states is not None:
                decoder_hidden_states = outputs.decoder_hidden_states
                self.assertIsInstance(decoder_hidden_states, (list, tuple))

    @unittest.skip(reason="SAM3VisionModel has FPN channel mismatch with flex attention")
    def test_flex_attention_with_grads(self):
        pass

    @unittest.skip(
        reason="Sam3Model creates attention masks from features (with gradients), "
        "which is incompatible with flash attention's expectation of binary masks"
    )
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip(
        reason="Sam3Model creates attention masks from features (with gradients), "
        "which is incompatible with flash attention's expectation of binary masks"
    )
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(
        reason="Sam3Model creates attention masks from features (with gradients), "
        "which is incompatible with flash attention's expectation of binary masks"
    )
    def test_flash_attn_3_inference_equivalence(self):
        pass

    @unittest.skip(
        reason="Sam3Model creates attention masks from features (with gradients), "
        "which is incompatible with flash attention's expectation of binary masks"
    )
    def test_flash_attn_3_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(
        reason="Sam3Model creates attention masks from features (with gradients), "
        "which is incompatible with flash attention's expectation of binary masks"
    )
    def test_flash_attn_kernels_inference_equivalence(self):
        pass

    @unittest.skip(
        reason="Sam3Model creates attention masks from features (with gradients), "
        "which is incompatible with flash attention's expectation of binary masks"
    )
    def test_flash_attn_kernels_mps_inference_equivalence(self):
        pass

    def test_sdpa_can_dispatch_composite_models(self):
        """
        Tests if composite models dispatch correctly on SDPA/eager when requested.
        SAM3 has multiple sub-models: vision_encoder, text_encoder, geometry_encoder,
        detr_encoder, detr_decoder, mask_decoder.
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

                # Get all sub-models that support attention
                vision_encoder_sdpa = getattr(model_sdpa, "vision_encoder")
                text_encoder_sdpa = getattr(model_sdpa, "text_encoder", None)
                detr_encoder_sdpa = getattr(model_sdpa, "detr_encoder", None)
                detr_decoder_sdpa = getattr(model_sdpa, "detr_decoder", None)
                mask_decoder_sdpa = getattr(model_sdpa, "mask_decoder", None)

                # Check that sub-models dispatch to SDPA if they support it
                self.assertTrue(vision_encoder_sdpa.config._attn_implementation == "sdpa")
                if text_encoder_sdpa is not None and hasattr(text_encoder_sdpa, "_supports_sdpa"):
                    # Text encoder from CLIP should support SDPA
                    self.assertTrue(text_encoder_sdpa.config._attn_implementation == "sdpa")
                if detr_encoder_sdpa is not None:
                    self.assertTrue(detr_encoder_sdpa.config._attn_implementation == "sdpa")
                if detr_decoder_sdpa is not None:
                    self.assertTrue(detr_decoder_sdpa.config._attn_implementation == "sdpa")
                if mask_decoder_sdpa is not None:
                    self.assertTrue(mask_decoder_sdpa.config._attn_implementation == "sdpa")

                # Now test with eager
                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)

                self.assertTrue(getattr(model_eager, "vision_encoder").config._attn_implementation == "eager")
                if hasattr(model_eager, "text_encoder"):
                    self.assertTrue(model_eager.text_encoder.config._attn_implementation == "eager")
                if hasattr(model_eager, "detr_encoder"):
                    self.assertTrue(model_eager.detr_encoder.config._attn_implementation == "eager")
                if hasattr(model_eager, "detr_decoder"):
                    self.assertTrue(model_eager.detr_decoder.config._attn_implementation == "eager")
                if hasattr(model_eager, "mask_decoder"):
                    self.assertTrue(model_eager.mask_decoder.config._attn_implementation == "eager")

                # Verify no SDPA layers in eager model
                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if (
                        class_name.endswith("Attention")
                        and getattr(submodule, "config", None)
                        and submodule.config._attn_implementation == "sdpa"
                    ):
                        raise ValueError("The eager model should not have SDPA attention layers")

    def test_forward_with_text_embeds(self):
        """Test that text_embeds parameter works correctly."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            # First get text embeddings
            with torch.no_grad():
                text_embeds = model.get_text_features(
                    input_ids=inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"], return_dict=True
                ).pooler_output

            # Forward with text_embeds (remove input_ids)
            inputs_with_embeds = {
                "pixel_values": inputs_dict["pixel_values"],
                "text_embeds": text_embeds,
            }

            with torch.no_grad():
                outputs_with_embeds = model(**inputs_with_embeds)

            # Forward with input_ids
            with torch.no_grad():
                outputs_with_ids = model(**inputs_dict)

            # Outputs should be very close
            self.assertTrue(torch.allclose(outputs_with_embeds.pred_logits, outputs_with_ids.pred_logits, atol=1e-5))
            self.assertTrue(torch.allclose(outputs_with_embeds.pred_boxes, outputs_with_ids.pred_boxes, atol=1e-5))

    def test_forward_with_both_input_ids_and_text_embeds_raises_error(self):
        """Test that passing both input_ids and text_embeds raises an error."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            # Get text embeddings
            with torch.no_grad():
                text_embeds = model.get_text_features(
                    input_ids=inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"]
                )

            # Try to pass both (should raise error)
            inputs_with_both = {
                "pixel_values": inputs_dict["pixel_values"],
                "input_ids": inputs_dict["input_ids"],
                "text_embeds": text_embeds,
            }

            with self.assertRaises(ValueError):
                model(**inputs_with_both)

    def test_forward_with_vision_embeds(self):
        """Test that vision_embeds parameter works correctly."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            # First get vision embeddings
            with torch.no_grad():
                vision_embeds = model.get_vision_features(pixel_values=inputs_dict["pixel_values"])

            # Forward with vision_embeds (remove pixel_values)
            inputs_with_embeds = {
                "vision_embeds": vision_embeds,
                "input_ids": inputs_dict["input_ids"],
                "attention_mask": inputs_dict["attention_mask"],
            }

            with torch.no_grad():
                outputs_with_embeds = model(**inputs_with_embeds)

            # Forward with pixel_values
            with torch.no_grad():
                outputs_with_pixels = model(**inputs_dict)

            # Outputs should be very close
            self.assertTrue(
                torch.allclose(outputs_with_embeds.pred_logits, outputs_with_pixels.pred_logits, atol=1e-5)
            )
            self.assertTrue(torch.allclose(outputs_with_embeds.pred_boxes, outputs_with_pixels.pred_boxes, atol=1e-5))

    def test_forward_with_both_pixel_values_and_vision_embeds_raises_error(self):
        """Test that passing both pixel_values and vision_embeds raises an error."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            # Get vision embeddings
            with torch.no_grad():
                vision_embeds = model.get_vision_features(pixel_values=inputs_dict["pixel_values"])

            # Try to pass both (should raise error)
            inputs_with_both = {
                "pixel_values": inputs_dict["pixel_values"],
                "vision_embeds": vision_embeds,
                "input_ids": inputs_dict["input_ids"],
                "attention_mask": inputs_dict["attention_mask"],
            }

            with self.assertRaises(ValueError):
                model(**inputs_with_both)

    def test_custom_image_size(self):
        """Test that custom image size can be set and propagates correctly through nested configs."""
        config = self.model_tester.get_config()
        config.image_size = 560

        self.assertEqual(config.image_size, 560)
        self.assertEqual(config.vision_config.image_size, 560)
        self.assertEqual(config.vision_config.backbone_config.image_size, 560)

        # Verify model works with custom size
        model = Sam3Model(config=config).to(torch_device).eval()
        pixel_values = floats_tensor([self.model_tester.batch_size, self.model_tester.num_channels, 560, 560]).to(
            torch_device
        )
        input_ids = torch.randint(0, 1000, (self.model_tester.batch_size, 16), device=torch_device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

        self.assertIsNotNone(outputs.pred_masks)
        self.assertIsNotNone(outputs.pred_boxes)
        self.assertIsNotNone(outputs.pred_logits)

    @unittest.skip(reason="SAM3 model can't be compiled dynamic yet")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(
        reason="SAM3 uses CLIP text encoder which has two attention masks: `causal_attention_mask` and `attention_mask`."
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_model_outputs_equivalence(self):
        """
        Test that tuple and dict outputs are equivalent.
        SAM3 returns complex outputs with component-specific fields, so we need to ensure proper conversion.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (list, tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    # model might return non-tensors objects (e.g. Cache class)
                    elif isinstance(tuple_object, torch.Tensor):
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            # Test with output_hidden_states
            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            # Test with output_attentions if supported
            if self.has_attentions:
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class)
                check_equivalence(
                    model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
                )

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        """Override to ensure input_ids and attention_mask are always present for Sam3Model."""
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        # Sam3Model always requires input_ids and attention_mask for text encoding
        if model_class == Sam3Model:
            if "input_ids" not in inputs_dict or inputs_dict.get("input_ids") is None:
                # Create dummy input_ids if not present
                # Get batch_size from pixel_values or vision_embeds
                if "pixel_values" in inputs_dict and inputs_dict.get("pixel_values") is not None:
                    batch_size = inputs_dict["pixel_values"].shape[0]
                elif "vision_embeds" in inputs_dict and inputs_dict.get("vision_embeds") is not None:
                    vision_embeds = inputs_dict["vision_embeds"]
                    if vision_embeds.fpn_hidden_states is not None and len(vision_embeds.fpn_hidden_states) > 0:
                        batch_size = vision_embeds.fpn_hidden_states[0].shape[0]
                    elif vision_embeds.last_hidden_state is not None:
                        batch_size = vision_embeds.last_hidden_state.shape[0]
                    else:
                        batch_size = 2
                else:
                    batch_size = 2
                config = self.model_tester.get_config()
                # text_config might be a dict or a config object
                if isinstance(config.text_config, dict):
                    vocab_size = config.text_config.get("vocab_size", 1000)
                else:
                    vocab_size = getattr(config.text_config, "vocab_size", 1000)
                inputs_dict["input_ids"] = torch.randint(0, vocab_size, (batch_size, 16), device=torch_device)
            if "attention_mask" not in inputs_dict or inputs_dict.get("attention_mask") is None:
                inputs_dict["attention_mask"] = torch.ones_like(inputs_dict["input_ids"])

        return inputs_dict


def prepare_coco_cat_image():
    """Prepare COCO cat and laptop image (from batched inference notebook)."""
    img_url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


def prepare_coco_kitchen_image():
    """Prepare COCO kitchen scene image (from batched inference notebook)."""
    img_url = "http://images.cocodataset.org/val2017/000000136466.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


@slow
class Sam3ModelIntegrationTest(unittest.TestCase):
    """Integration tests for SAM3 model with real pretrained weights."""

    def setUp(self):
        super().setUp()
        model_name = "facebook/sam3"
        self.model = Sam3Model.from_pretrained(model_name).to(torch.float32)
        self.processor = Sam3Processor.from_pretrained(model_name)
        self.model.to(torch_device)
        self.model.eval()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_inference_text_prompt_only(self):
        """Test inference with text prompt only (from multiway_prompting notebook)."""
        # Example from notebook: "short hair" text prompt
        raw_image = prepare_coco_cat_image()
        text = "ear"

        inputs = self.processor(images=raw_image, text=text, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Check exact output shapes
        self.assertEqual(outputs.pred_masks.shape, (1, 200, 288, 288))
        self.assertEqual(outputs.pred_boxes.shape, (1, 200, 4))
        self.assertEqual(outputs.pred_logits.shape, (1, 200))

        # Check that predictions have reasonable scores (after sigmoid)
        scores = torch.sigmoid(outputs.pred_logits)
        self.assertTrue((scores >= 0).all() and (scores <= 1).all())

        # Check exact values
        sorted_indices = torch.argsort(scores.squeeze(), descending=True)
        top_scores = scores.squeeze()[sorted_indices[:3]]
        top_logits = outputs.pred_logits.squeeze()[sorted_indices[:3]]
        top_idx = sorted_indices[0].item()

        torch.testing.assert_close(
            top_scores, torch.tensor([0.9381, 0.9214, 0.0910]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            top_logits, torch.tensor([2.7182, 2.4618, -2.3020]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            outputs.pred_boxes[0, top_idx],
            torch.tensor([0.4704, 0.2014, 0.5615, 0.3770]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[0, top_idx, :3, :3],
            torch.tensor(
                [[-2.1815, -6.2767, -7.0687], [-5.7988, -10.2704, -10.9379], [-8.5194, -10.7892, -9.9152]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # Test post-processing
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
        )
        self.assertEqual(len(results), 1)
        result = results[0]

        # Check that we have detections
        self.assertGreater(len(result["masks"]), 0)
        self.assertGreater(len(result["boxes"]), 0)
        self.assertGreater(len(result["scores"]), 0)

        # Check exact values for top detection
        top_pp_score = result["scores"][0]
        top_pp_box = result["boxes"][0]

        torch.testing.assert_close(top_pp_score, torch.tensor(0.9210).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            top_pp_box, torch.tensor([402.1755, 90.1420, 459.6165, 156.3702]).to(torch_device), atol=1e-4, rtol=1e-4
        )

    def test_inference_single_box_prompt(self):
        """Test inference with a single bounding box prompt (from batched_inference notebook)."""
        raw_image = prepare_coco_cat_image()
        # Example from notebook: laptop region in image 1
        # Box in xyxy format: [100, 150, 500, 450]
        box_xyxy = [100, 150, 500, 450]
        input_boxes = [[box_xyxy]]

        inputs = self.processor(
            images=raw_image,
            input_boxes=input_boxes,
            input_boxes_labels=[[1]],  # Positive box
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Check exact output shapes
        self.assertEqual(outputs.pred_masks.shape, (1, 200, 288, 288))
        self.assertEqual(outputs.pred_boxes.shape, (1, 200, 4))
        self.assertEqual(outputs.pred_logits.shape, (1, 200))

        # Check exact values
        scores = torch.sigmoid(outputs.pred_logits)
        sorted_indices = torch.argsort(scores.squeeze(), descending=True)
        top_scores = scores.squeeze()[sorted_indices[:3]]
        top_logits = outputs.pred_logits.squeeze()[sorted_indices[:3]]
        top_idx = sorted_indices[0].item()

        torch.testing.assert_close(
            top_scores, torch.tensor([0.9308, 0.1617, 0.1336]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            top_logits, torch.tensor([2.5988, -1.6460, -1.8699]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            outputs.pred_boxes[0, top_idx],
            torch.tensor([0.1631, 0.4140, 0.7510, 0.9931]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[0, top_idx, :3, :3],
            torch.tensor([[-1.8726, -3.5063, -3.7716], [-3.1987, -5.3820, -5.6782], [-3.8850, -5.4164, -5.8604]]).to(
                torch_device
            ),
            atol=1e-4,
            rtol=1e-4,
        )

        # Test post-processing
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
        )
        self.assertEqual(len(results), 1)
        result = results[0]

        # Check that we have detections
        self.assertGreater(len(result["masks"]), 0)

        # Check exact values for top detection
        top_pp_score = result["scores"][0]
        top_pp_box = result["boxes"][0]

        torch.testing.assert_close(top_pp_score, torch.tensor(0.9307).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            top_pp_box, torch.tensor([104.3945, 175.9433, 480.6293, 422.0826]).to(torch_device), atol=1e-4, rtol=1e-4
        )

    def test_inference_multi_box_prompt(self):
        """Test inference with multiple box prompts with positive and negative labels (from batched_inference notebook)."""
        raw_image = prepare_coco_kitchen_image()
        # Example from notebook: multiple positive boxes (dial + button)
        # Dial box (xyxy): [59, 144, 76, 163]
        # Button box (xyxy): [87, 148, 104, 159]
        box1_xyxy = [59, 144, 76, 163]
        box2_xyxy = [87, 148, 104, 159]

        input_boxes = [[box1_xyxy, box2_xyxy]]
        input_boxes_labels = [[1, 1]]  # Both positive

        inputs = self.processor(
            images=raw_image, input_boxes=input_boxes, input_boxes_labels=input_boxes_labels, return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Check exact output shapes
        self.assertEqual(outputs.pred_masks.shape, (1, 200, 288, 288))
        self.assertEqual(outputs.pred_boxes.shape, (1, 200, 4))
        self.assertEqual(outputs.pred_logits.shape, (1, 200))

        # Check exact values
        scores = torch.sigmoid(outputs.pred_logits)
        sorted_indices = torch.argsort(scores.squeeze(), descending=True)
        top_scores = scores.squeeze()[sorted_indices[:3]]
        top_logits = outputs.pred_logits.squeeze()[sorted_indices[:3]]
        top_idx = sorted_indices[0].item()

        torch.testing.assert_close(
            top_scores, torch.tensor([0.9611, 0.9379, 0.8348]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            top_logits, torch.tensor([3.2071, 2.7154, 1.6198]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            outputs.pred_boxes[0, top_idx],
            torch.tensor([0.1757, 0.2888, 0.2296, 0.3259]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[0, top_idx, :3, :3],
            torch.tensor(
                [[-8.6138, -14.5615, -17.9965], [-13.6695, -20.4994, -25.6705], [-14.9681, -23.0616, -17.0045]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # Test post-processing
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
        )
        self.assertEqual(len(results), 1)
        result = results[0]

        # Check that we have detections
        self.assertGreater(len(result["masks"]), 0)

        # Check exact values for top detection
        top_pp_score = result["scores"][0]
        top_pp_box = result["boxes"][0]

        torch.testing.assert_close(top_pp_score, torch.tensor(0.9379).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            top_pp_box, torch.tensor([86.8687, 147.5269, 104.4475, 159.6138]).to(torch_device), atol=1e-4, rtol=1e-4
        )

    def test_inference_combined_prompts(self):
        """Test inference with combined text and geometry prompts (text + negative box from batched_inference notebook)."""
        raw_image = prepare_coco_kitchen_image()
        # Example from notebook: text "handle" + negative box to exclude oven handle
        text = "handle"
        # Negative box covering the oven handle area (xyxy): [40, 183, 318, 204]
        oven_handle_box = [40, 183, 318, 204]

        input_boxes = [[oven_handle_box]]

        inputs = self.processor(
            images=raw_image,
            text=text,
            input_boxes=input_boxes,
            input_boxes_labels=[[0]],  # 0 = negative
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Check exact output shapes
        self.assertEqual(outputs.pred_masks.shape, (1, 200, 288, 288))
        self.assertEqual(outputs.pred_boxes.shape, (1, 200, 4))
        self.assertEqual(outputs.pred_logits.shape, (1, 200))

    def test_inference_batched_images(self):
        """Test batched inference with multiple images (from batched_inference notebook)."""
        # Example from notebook: batch of 2 images with different text prompts
        raw_image1 = prepare_coco_cat_image()
        raw_image2 = prepare_coco_kitchen_image()

        # Batch of 2 images with different text prompts: "ear" for cat, "dial" for kitchen
        inputs = self.processor(images=[raw_image1, raw_image2], text=["ear", "dial"], return_tensors="pt").to(
            torch_device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Check exact output shapes
        self.assertEqual(outputs.pred_masks.shape, (2, 200, 288, 288))
        self.assertEqual(outputs.pred_boxes.shape, (2, 200, 4))
        self.assertEqual(outputs.pred_logits.shape, (2, 200))

        # Check scores are reasonable
        scores = torch.sigmoid(outputs.pred_logits)
        self.assertTrue((scores >= 0).all() and (scores <= 1).all())

        # Check exact values
        sorted_indices_0 = torch.argsort(scores[0], descending=True)
        sorted_indices_1 = torch.argsort(scores[1], descending=True)
        top_scores_0 = scores[0][sorted_indices_0[:3]]
        top_scores_1 = scores[1][sorted_indices_1[:3]]
        top_logits_0 = outputs.pred_logits[0][sorted_indices_0[:3]]
        top_logits_1 = outputs.pred_logits[1][sorted_indices_1[:3]]
        top_idx_0 = sorted_indices_0[0].item()
        top_idx_1 = sorted_indices_1[0].item()

        torch.testing.assert_close(
            top_scores_0, torch.tensor([0.9381, 0.9214, 0.0910]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            top_scores_1, torch.tensor([0.8863, 0.8849, 0.8841]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            top_logits_0, torch.tensor([2.7182, 2.4618, -2.3020]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            top_logits_1, torch.tensor([2.0534, 2.0395, 2.0320]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            outputs.pred_boxes[0, top_idx_0],
            torch.tensor([0.4704, 0.2014, 0.5615, 0.3770]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_boxes[1, top_idx_1],
            torch.tensor([0.6162, 0.2769, 0.6838, 0.3238]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[0, top_idx_0, :3, :3],
            torch.tensor(
                [[-2.1815, -6.2767, -7.0687], [-5.7988, -10.2704, -10.9379], [-8.5194, -10.7892, -9.9152]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[1, top_idx_1, :3, :3],
            torch.tensor(
                [[-7.4371, -13.5898, -13.6496], [-11.8669, -20.6416, -23.0941], [-12.8623, -20.3439, -16.6497]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # Test post-processing
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.3, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
        )
        self.assertEqual(len(results), 2)

        # Check that both have detections
        self.assertGreater(len(results[0]["masks"]), 0)
        self.assertGreater(len(results[1]["masks"]), 0)

        # Check exact values for top detection in each image
        top_pp_score_0 = results[0]["scores"][0]
        top_pp_box_0 = results[0]["boxes"][0]
        top_pp_score_1 = results[1]["scores"][0]
        top_pp_box_1 = results[1]["boxes"][0]

        torch.testing.assert_close(top_pp_score_0, torch.tensor(0.9210).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            top_pp_box_0, torch.tensor([402.1755, 90.1421, 459.6165, 156.3701]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(top_pp_score_1, torch.tensor(0.6641).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            top_pp_box_1, torch.tensor([110.6279, 271.1848, 137.3600, 301.3683]).to(torch_device), atol=1e-4, rtol=1e-4
        )

    def test_inference_batched_mixed_prompts(self):
        """Test batched inference with mixed prompt types (from batched_inference notebook)."""
        # Example from notebook: Image 1 with text "laptop", Image 2 with visual prompt (dial)
        raw_image1 = prepare_coco_cat_image()
        raw_image2 = prepare_coco_kitchen_image()

        # Box for dial in image 2 (xyxy): [59, 144, 76, 163]
        box2_xyxy = [59, 144, 76, 163]

        inputs = self.processor(
            images=[raw_image1, raw_image2],
            text=["laptop", None],  # Only first image has text
            input_boxes=[None, [box2_xyxy]],  # Only second image has box
            input_boxes_labels=[None, [1]],
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Check exact output shapes
        self.assertEqual(outputs.pred_masks.shape, (2, 200, 288, 288))
        self.assertEqual(outputs.pred_boxes.shape, (2, 200, 4))
        self.assertEqual(outputs.pred_logits.shape, (2, 200))

        # Check exact values
        scores = torch.sigmoid(outputs.pred_logits)
        sorted_indices_0 = torch.argsort(scores[0], descending=True)
        sorted_indices_1 = torch.argsort(scores[1], descending=True)
        top_scores_0 = scores[0][sorted_indices_0[:3]]
        top_scores_1 = scores[1][sorted_indices_1[:3]]
        top_logits_0 = outputs.pred_logits[0][sorted_indices_0[:3]]
        top_logits_1 = outputs.pred_logits[1][sorted_indices_1[:3]]
        top_idx_0 = sorted_indices_0[0].item()
        top_idx_1 = sorted_indices_1[0].item()

        torch.testing.assert_close(
            top_scores_0, torch.tensor([0.9756, 0.1352, 0.0701]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            top_scores_1, torch.tensor([0.9683, 0.8310, 0.8222]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            top_logits_0, torch.tensor([3.6865, -1.8555, -2.5854]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            top_logits_1, torch.tensor([3.4183, 1.5929, 1.5315]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            outputs.pred_boxes[0, top_idx_0],
            torch.tensor([-0.0013, 0.0016, 0.4521, 0.9964]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_boxes[1, top_idx_1],
            torch.tensor([0.1774, 0.2876, 0.2296, 0.3261]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[0, top_idx_0, :3, :3],
            torch.tensor([[0.0520, 0.3121, 0.4103], [0.6820, 1.0069, 1.0949], [0.8418, 1.0318, 1.0365]]).to(
                torch_device
            ),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[1, top_idx_1, :3, :3],
            torch.tensor(
                [[-8.7447, -14.3499, -17.5662], [-13.6804, -20.3728, -25.5098], [-15.2996, -22.9116, -17.6658]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # Test post-processing
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.3, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
        )
        self.assertEqual(len(results), 2)

        # Check that both have detections
        self.assertGreater(len(results[0]["masks"]), 0)
        self.assertGreater(len(results[1]["masks"]), 0)

        # Check exact values for top detection in each image
        top_pp_score_0 = results[0]["scores"][0]
        top_pp_box_0 = results[0]["boxes"][0]
        top_pp_score_1 = results[1]["scores"][0]
        top_pp_box_1 = results[1]["boxes"][0]

        torch.testing.assert_close(top_pp_score_0, torch.tensor(0.9655).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            top_pp_box_0, torch.tensor([-0.8481, 0.6668, 289.3758, 423.4723]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(top_pp_score_1, torch.tensor(0.8222).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            top_pp_box_1, torch.tensor([168.9376, 137.3257, 191.7281, 161.3243]).to(torch_device), atol=1e-4, rtol=1e-4
        )

    # TODO add exact values
    def test_semantic_segmentation_output(self):
        """Test that semantic segmentation output is produced."""
        raw_image = prepare_coco_cat_image()
        inputs = self.processor(images=raw_image, text="ear", return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Check exact semantic segmentation output shape
        self.assertEqual(outputs.semantic_seg.shape, (1, 1, 288, 288))
        # Check that semantic seg has same spatial size as pred_masks
        self.assertEqual(outputs.semantic_seg.shape[-2:], outputs.pred_masks.shape[-2:])

    @require_deterministic_for_xpu
    def test_efficient_multi_prompt_single_image(self):
        """Test efficient inference with multiple prompts on a single image using get_vision_features."""
        raw_image = prepare_coco_cat_image()

        # Pre-compute vision embeddings once
        img_inputs = self.processor(images=raw_image, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            vision_embeds = self.model.get_vision_features(pixel_values=img_inputs.pixel_values)

        # Run multiple text prompts efficiently
        text_prompts = ["ear", "eye"]
        all_results = []

        for prompt in text_prompts:
            text_inputs = self.processor(text=prompt, return_tensors="pt").to(torch_device)
            with torch.no_grad():
                outputs = self.model(vision_embeds=vision_embeds, **text_inputs)

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=img_inputs.get("original_sizes").tolist(),
            )[0]
            all_results.append(results)

        # Check that we get results for both prompts
        self.assertEqual(len(all_results), 2)

        # Verify outputs are equivalent to running with pixel_values directly
        text_inputs = self.processor(text="ear", return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs_with_embeds = self.model(vision_embeds=vision_embeds, **text_inputs)

        inputs_direct = self.processor(images=raw_image, text="ear", return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs_direct = self.model(**inputs_direct)

        # Outputs should be identical
        torch.testing.assert_close(outputs_with_embeds.pred_logits, outputs_direct.pred_logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(outputs_with_embeds.pred_boxes, outputs_direct.pred_boxes, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(outputs_with_embeds.pred_masks, outputs_direct.pred_masks, atol=1e-5, rtol=1e-5)

    @require_deterministic_for_xpu
    def test_efficient_single_prompt_multi_images(self):
        """Test efficient inference with same prompt on multiple images using get_text_features."""
        raw_image1 = prepare_coco_cat_image()
        raw_image2 = prepare_coco_kitchen_image()

        # Pre-compute text embeddings once
        text_prompt = "handle"
        text_inputs = self.processor(text=text_prompt, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**text_inputs)

        # Run inference on multiple images reusing text embeddings
        # Note: attention_mask must be passed along with text_embeds for proper masking
        images = [raw_image1, raw_image2]
        all_results = []

        for image in images:
            img_inputs = self.processor(images=image, return_tensors="pt").to(torch_device)
            with torch.no_grad():
                outputs = self.model(
                    text_embeds=text_embeds,
                    attention_mask=text_inputs.attention_mask,
                    **img_inputs,
                )

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=img_inputs.get("original_sizes").tolist(),
            )[0]
            all_results.append(results)

        # Check that we get results for both images
        self.assertEqual(len(all_results), 2)

        # Verify outputs are equivalent to running with input_ids directly
        img_inputs = self.processor(images=raw_image2, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs_with_embeds = self.model(
                text_embeds=text_embeds,
                attention_mask=text_inputs.attention_mask,
                **img_inputs,
            )

        inputs_direct = self.processor(images=raw_image2, text=text_prompt, return_tensors="pt").to(torch_device)
        with torch.no_grad():
            outputs_direct = self.model(**inputs_direct)

        # Outputs should be identical
        torch.testing.assert_close(outputs_with_embeds.pred_logits, outputs_direct.pred_logits, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(outputs_with_embeds.pred_boxes, outputs_direct.pred_boxes, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(outputs_with_embeds.pred_masks, outputs_direct.pred_masks, atol=1e-5, rtol=1e-5)
