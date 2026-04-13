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
"""Testing suite for the PyTorch SAM3-LiteText model."""

import unittest

from transformers.testing_utils import (
    require_torch,
    torch_device,
)
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers.models.sam3.configuration_sam3 import (
        Sam3DETRDecoderConfig,
        Sam3DETREncoderConfig,
        Sam3GeometryEncoderConfig,
        Sam3MaskDecoderConfig,
        Sam3VisionConfig,
        Sam3ViTConfig,
    )
    from transformers.models.sam3_lite_text.configuration_sam3_lite_text import (
        Sam3LiteTextConfig,
        Sam3LiteTextMobileCLIPConfig,
    )
    from transformers.models.sam3_lite_text.modeling_sam3_lite_text import Sam3LiteTextModel


class Sam3LiteTextModelTester:
    def __init__(
        self,
        parent,
        num_channels=3,
        image_size=224,
        hidden_size=32,
        patch_size=14,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        window_size=8,
        global_attn_indexes=None,
        fpn_hidden_size=32,
        scale_factors=None,
        # MobileCLIP text encoder (small)
        text_hidden_size=32,
        text_num_hidden_layers=1,
        text_num_attention_heads=2,
        text_intermediate_size=64,
        text_context_length=8,
        text_vocab_size=1000,
        # Other components
        geometry_encoder_hidden_size=32,
        geometry_encoder_num_layers=1,
        detr_encoder_hidden_size=32,
        detr_encoder_num_layers=1,
        detr_decoder_hidden_size=32,
        detr_decoder_num_layers=1,
        detr_decoder_num_queries=5,
        mask_decoder_hidden_size=32,
        batch_size=2,
    ):
        if global_attn_indexes is None:
            global_attn_indexes = [0, 1]
        if scale_factors is None:
            scale_factors = [2.0, 1.0]

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

        self.text_hidden_size = text_hidden_size
        self.text_num_hidden_layers = text_num_hidden_layers
        self.text_num_attention_heads = text_num_attention_heads
        self.text_intermediate_size = text_intermediate_size
        self.text_context_length = text_context_length
        self.text_vocab_size = text_vocab_size

        self.geometry_encoder_hidden_size = geometry_encoder_hidden_size
        self.geometry_encoder_num_layers = geometry_encoder_num_layers
        self.detr_encoder_hidden_size = detr_encoder_hidden_size
        self.detr_encoder_num_layers = detr_encoder_num_layers
        self.detr_decoder_hidden_size = detr_decoder_hidden_size
        self.detr_decoder_num_layers = detr_decoder_num_layers
        self.detr_decoder_num_queries = detr_decoder_num_queries
        self.mask_decoder_hidden_size = mask_decoder_hidden_size

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

        text_config = Sam3LiteTextMobileCLIPConfig(
            hidden_size=self.text_hidden_size,
            num_hidden_layers=self.text_num_hidden_layers,
            num_attention_heads=self.text_num_attention_heads,
            intermediate_size=self.text_intermediate_size,
            context_length=self.text_context_length,
            vocab_size=self.text_vocab_size,
            projection_dim=self.text_hidden_size,
            kernel_size=3,
        )

        geometry_encoder_config = Sam3GeometryEncoderConfig(
            hidden_size=self.geometry_encoder_hidden_size,
            num_layers=self.geometry_encoder_num_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
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
            num_upsampling_stages=2,
        )

        return Sam3LiteTextConfig(
            vision_config=vision_config,
            text_config=text_config,
            geometry_encoder_config=geometry_encoder_config,
            detr_encoder_config=detr_encoder_config,
            detr_decoder_config=detr_decoder_config,
            mask_decoder_config=mask_decoder_config,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        input_ids = torch.randint(
            0, self.text_vocab_size, (self.batch_size, self.text_context_length), device=torch_device
        )
        attention_mask = torch.ones_like(input_ids)

        config = self.get_config()
        return config, pixel_values, input_ids, attention_mask

    def create_and_check_model(self, config, pixel_values, input_ids, attention_mask):
        model = Sam3LiteTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        self.parent.assertIsNotNone(result.pred_masks)
        self.parent.assertIsNotNone(result.pred_boxes)
        self.parent.assertIsNotNone(result.pred_logits)

        self.parent.assertEqual(result.pred_masks.shape[0], self.batch_size)
        self.parent.assertEqual(result.pred_masks.shape[1], self.detr_decoder_num_queries)
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.detr_decoder_num_queries, 4))
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
class Sam3LiteTextModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Sam3LiteTextModel,) if is_torch_available() else ()
    pipeline_model_mapping = {"mask-generation": Sam3LiteTextModel} if is_torch_available() else {}

    test_resize_embeddings = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Sam3LiteTextModelTester(self)
        common_properties = ["initializer_range"]
        self.config_tester = ConfigTester(
            self, config_class=Sam3LiteTextConfig, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        # Skip composite config roundtrip test: the generated sub_configs has CLIPTextConfig
        # for text_config (inherited from Sam3Config), but we use Sam3LiteTextMobileCLIPConfig.
        # The modular converter cannot override sub_configs with generated types.
        # Individual config tests still run below.
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    @unittest.skip(reason="SAM3-LiteText does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.vision_encoder.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_batching_equivalence(self, atol=5e-4, rtol=5e-4):
        super().test_batching_equivalence(atol=atol, rtol=rtol)

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

            self.assertIsNotNone(outputs.vision_attentions)
            self.assertIsNotNone(outputs.detr_encoder_attentions)
            self.assertIsNotNone(outputs.detr_decoder_attentions)
            self.assertIsNotNone(outputs.mask_decoder_attentions)

            if outputs.vision_attentions:
                self.assertEqual(len(outputs.vision_attentions), self.model_tester.num_hidden_layers)

            self.assertTrue(
                len(outputs.vision_attentions) > 0,
                "At least vision attentions should be collected when output_attentions=True",
            )

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for k in config.sub_configs:
            if getattr(config, k) is not None:
                getattr(config, k).output_hidden_states = True
                getattr(config, k).output_attentions = True

        config.output_hidden_states = True
        config.output_attentions = True
        config._attn_implementation = "eager"

        model_class = self.all_model_classes[0]
        model = model_class._from_config(config, attn_implementation="eager")
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)
        outputs = model(**inputs)

        output = outputs[0]

        if outputs.vision_hidden_states is not None and len(outputs.vision_hidden_states) > 0:
            outputs.vision_hidden_states[0].retain_grad()

        if outputs.vision_attentions is not None and len(outputs.vision_attentions) > 0:
            outputs.vision_attentions[0].retain_grad()

        output.sum().backward(retain_graph=True)

    def test_text_encoder(self):
        """Test that the MobileCLIP text encoder produces correct output shapes."""
        config = self.model_tester.get_config()
        model = Sam3LiteTextModel(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = torch.randint(
            0, self.model_tester.text_vocab_size, (2, self.model_tester.text_context_length), device=torch_device
        )
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            text_features = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        self.assertIsNotNone(text_features.last_hidden_state)
        self.assertEqual(
            text_features.last_hidden_state.shape,
            (2, self.model_tester.text_context_length, self.model_tester.text_hidden_size),
        )
        # pooler_output is projected to DETR hidden size
        self.assertIsNotNone(text_features.pooler_output)
        self.assertEqual(
            text_features.pooler_output.shape,
            (2, self.model_tester.text_context_length, self.model_tester.detr_encoder_hidden_size),
        )

    @unittest.skip(reason="SAM3-LiteText can't be compiled dynamic yet")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="SAM3-LiteText has FPN channel mismatch with flex attention")
    def test_flex_attention_with_grads(self):
        pass

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        for k in config.sub_configs:
            if (subconfig := getattr(config, k, None)) is not None:
                subconfig.output_hidden_states = True
                for sk in getattr(subconfig, "sub_configs", {}):
                    if (subsubconfig := getattr(subconfig, sk, None)) is not None:
                        subsubconfig.output_hidden_states = True
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertIsNotNone(outputs.vision_hidden_states)
            self.assertTrue(len(outputs.vision_hidden_states) > 0)

    @unittest.skip(
        reason="SAM3-LiteText uses component-specific hidden states, training test expects generic hidden_states"
    )
    def test_training(self):
        pass

    @unittest.skip(reason="SAM3-LiteText uses component-specific hidden states")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SAM3-LiteText uses component-specific hidden states")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skip(reason="SAM3-LiteText uses component-specific hidden states")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_24_fp32_pad_left_output_attentions(self):
        pass

    @unittest.skip(reason="MobileCLIP text encoder does not output attentions")
    def test_get_text_features_attentions(self):
        pass

    @unittest.skip(reason="MobileCLIP text encoder does not output hidden states via standard interface")
    def test_get_text_features_hidden_states(self):
        pass

    def test_eager_matches_sdpa_inference(self, *args, **kwargs):
        self.skipTest("MobileCLIP text encoder uses custom attention without SDPA support")

    @unittest.skip(reason="SDPA pad_left not supported")
    def test_eager_matches_sdpa_inference_01_fp16_pad_left(self):
        pass

    @unittest.skip(reason="SDPA pad_left not supported")
    def test_eager_matches_sdpa_inference_09_fp32_pad_left(self):
        pass

    @unittest.skip(reason="SDPA pad_left not supported")
    def test_eager_matches_sdpa_inference_17_bf16_pad_left(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_00_fp16_pad_left_sdpa_kernels(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_02_fp16_pad_left_no_attn_mask_sdpa_kernels(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_03_fp16_pad_left_no_attn_mask(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_04_fp16_pad_right_sdpa_kernels(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_05_fp16_pad_right(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_06_fp16_pad_right_no_attn_mask_sdpa_kernels(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_07_fp16_pad_right_no_attn_mask(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_08_fp32_pad_left_sdpa_kernels(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_10_fp32_pad_left_no_attn_mask_sdpa_kernels(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_11_fp32_pad_left_no_attn_mask(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_12_fp32_pad_right_sdpa_kernels(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_13_fp32_pad_right(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_14_fp32_pad_right_no_attn_mask_sdpa_kernels(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_15_fp32_pad_right_no_attn_mask(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_16_bf16_pad_left_sdpa_kernels(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_18_bf16_pad_left_no_attn_mask_sdpa_kernels(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_19_bf16_pad_left_no_attn_mask(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_20_bf16_pad_right_sdpa_kernels(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_21_bf16_pad_right(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_22_bf16_pad_right_no_attn_mask_sdpa_kernels(self):
        pass

    @unittest.skip(reason="SDPA not supported for composite text encoder")
    def test_eager_matches_sdpa_inference_23_bf16_pad_right_no_attn_mask(self):
        pass
