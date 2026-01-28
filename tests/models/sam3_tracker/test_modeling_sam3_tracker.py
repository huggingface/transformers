# Copyright 2025 the HuggingFace Team. All rights reserved.
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
    Sam3TrackerConfig,
    Sam3TrackerMaskDecoderConfig,
    Sam3TrackerPromptEncoderConfig,
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

    from transformers import Sam3TrackerModel, Sam3TrackerProcessor, Sam3VisionConfig, Sam3ViTConfig


if is_vision_available():
    from PIL import Image


class Sam3TrackerPromptEncoderTester:
    def __init__(
        self,
        hidden_size=32,
        input_image_size=128,
        patch_size=16,
        mask_input_channels=8,
        num_point_embeddings=4,
        hidden_act="gelu",
        is_training=True,
    ):
        self.hidden_size = hidden_size
        self.input_image_size = input_image_size
        self.patch_size = patch_size
        self.mask_input_channels = mask_input_channels
        self.num_point_embeddings = num_point_embeddings
        self.hidden_act = hidden_act
        self.is_training = is_training

    def get_config(self):
        return Sam3TrackerPromptEncoderConfig(
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


class Sam3TrackerMaskDecoderTester:
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
        is_training=True,
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
        self.is_training = is_training

    def get_config(self):
        return Sam3TrackerMaskDecoderConfig(
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


class Sam3TrackerModelTester:
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
        backbone_feature_sizes=[[32, 32], [16, 16], [8, 8]],
        memory_encoder_hidden_size=32,
        batch_size=2,
        is_training=True,
    ):
        if global_attn_indexes is None:
            global_attn_indexes = [0, 1]
        if scale_factors is None:
            scale_factors = [2.0, 1.0, 0.5]  # 3 scales to match backbone_feature_sizes

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
        self.backbone_feature_sizes = backbone_feature_sizes
        self.batch_size = batch_size
        self.is_training = is_training
        self.memory_encoder_hidden_size = memory_encoder_hidden_size
        self.prompt_encoder_tester = Sam3TrackerPromptEncoderTester()
        self.mask_decoder_tester = Sam3TrackerMaskDecoderTester()

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

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
            backbone_feature_sizes=self.backbone_feature_sizes,
        )

        prompt_encoder_config = self.prompt_encoder_tester.get_config()

        mask_decoder_config = self.mask_decoder_tester.get_config()

        return Sam3TrackerConfig(
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
        model = Sam3TrackerModel(config=config)
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
class Sam3TrackerModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SAM's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (Sam3TrackerModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": Sam3TrackerModel, "mask-generation": Sam3TrackerModel} if is_torch_available() else {}
    )

    test_resize_embeddings = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Sam3TrackerModelTester(self)
        common_properties = ["initializer_range"]
        self.config_tester = ConfigTester(
            self, config_class=Sam3TrackerConfig, has_text_modality=False, common_properties=common_properties
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

    # Overriding as Sam3TrackerModel returns vision_attentions
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
            expected_num_attentions = self.model_tester.num_hidden_layers
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.mask_decoder_config.output_attentions = True
            config.vision_config.output_attentions = True
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.vision_attentions
            self.assertEqual(len(attentions), expected_num_attentions)

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

    # Override as Sam3TrackerModel has different sub-modules
    def test_sdpa_can_dispatch_composite_models(self):
        """
        Tests if composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are called "SDPAAttention".
        In contrast to the above test, this one checks if the "config._attn_implementation" is a dict after the model
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

    # Override as Sam3TrackerModel doesn't have hidden states
    def flash_attn_inference_equivalence(
        self, attn_implementation: str, padding_side: str, atol: float = 4e-2, rtol: float = 4e-2
    ):
        r"""
        Tests the equivalence between the eager and flash attention implementations.
        This test is only for inference and runs with `dtype=torch.bfloat16`.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        # TODO take a look at this
        # head size needs to be a multiple of 8 but needs more adjustments than our current `_prepare_config_headdim`
        if attn_implementation != "flash_attention_2":
            self.skipTest(
                reason="Model fails for every other FA implementation than FA2 due to dim incompatibilities."
            )

        for model_class in self.all_model_classes:
            if not getattr(model_class, "_supports_flash_attn"):
                self.skipTest(f"{model_class.__name__} does not support Flash Attention")

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

                assert torch.allclose(logits_fa, logits, atol=atol, rtol=rtol)

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
                    assert torch.allclose(logits_fa[1:], logits[1:], atol=atol, rtol=rtol)

                    # check with inference + dropout
                    model.train()
                    _ = model_fa(dummy_input, **other_inputs)
                else:
                    assert torch.allclose(logits_fa[:-1], logits[:-1], atol=atol, rtol=rtol)

    # Override as difference slightly higher than the threshold
    def test_batching_equivalence(self, atol=5e-4, rtol=5e-4):
        super().test_batching_equivalence(atol=atol, rtol=rtol)

    @unittest.skip(reason="Hidden_states is tested in sub modules tests")
    def test_hidden_states_output(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/sam2.1-hiera-tiny"
        model = Sam3TrackerModel.from_pretrained(model_name)
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
class Sam3TrackerModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        checkpoint_path = "facebook/sam3"
        self.model = Sam3TrackerModel.from_pretrained(checkpoint_path).to(torch.float32)
        self.processor = Sam3TrackerProcessor.from_pretrained(checkpoint_path)
        self.model.to(torch_device)
        self.model.eval()

    def tearDown(self):
        super().tearDown()
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
        self.assertEqual(outputs.pred_masks.shape, (1, 1, 3, 288, 288))
        sorted_indices = torch.argsort(outputs.iou_scores.squeeze(), descending=True)
        scores = outputs.iou_scores.squeeze()[sorted_indices]
        masks_logits = outputs.pred_masks.squeeze()[sorted_indices][0, :3, :3]
        torch.testing.assert_close(
            scores,
            torch.tensor([0.9106, 0.5326, 0.0379]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            masks_logits,
            torch.tensor(
                [
                    [-18.9093, -31.1757, -23.6851],
                    [-20.3388, -31.0213, -29.8815],
                    [-20.7554, -29.4530, -30.1776],
                ]
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
        self.assertEqual(outputs.pred_masks.shape, (1, 1, 1, 288, 288))
        scores = outputs.iou_scores.squeeze((0, 1))
        masks_logits = outputs.pred_masks.squeeze((0, 1))[0, :3, :3]
        torch.testing.assert_close(scores, torch.tensor([0.9474]).to(torch_device), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            masks_logits,
            torch.tensor(
                [
                    [-8.1500, -12.3282, -9.6828],
                    [-9.0512, -11.6470, -11.6363],
                    [-9.2391, -11.9863, -12.4858],
                ]
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
        self.assertEqual(outputs.pred_masks.shape, (2, 1, 3, 288, 288))

        sorted_indices = torch.argsort(outputs.iou_scores[0].squeeze(), descending=True)
        scores1 = outputs.iou_scores[0].squeeze()[sorted_indices]
        masks_logits1 = outputs.pred_masks[0].squeeze()[sorted_indices][0, :3, :3]
        sorted_indices = torch.argsort(outputs.iou_scores[1].squeeze(), descending=True)
        scores2 = outputs.iou_scores[1].squeeze()[sorted_indices]
        masks_logits2 = outputs.pred_masks[1].squeeze()[sorted_indices][0, :3, :3]
        torch.testing.assert_close(
            scores1,
            torch.tensor([0.8837, 0.5837, 0.0372]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            masks_logits1,
            torch.tensor(
                [
                    [-19.4976, -32.4384, -24.2687],
                    [-20.9939, -32.2782, -31.2067],
                    [-21.2991, -30.3071, -31.1489],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        torch.testing.assert_close(
            scores2,
            torch.tensor([0.7675, 0.7505, 0.5348]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            masks_logits2,
            torch.tensor(
                [
                    [-10.3051, -9.9056, -10.5699],
                    [-8.8009, -11.1684, -10.7158],
                    [-9.6653, -10.9755, -10.3231],
                ]
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
        self.assertEqual(outputs.pred_masks.shape, (2, 2, 1, 288, 288))
        torch.testing.assert_close(
            outputs.iou_scores,
            torch.tensor([[[0.9370], [0.9425]], [[0.9734], [0.9262]]]).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[:, :, :, :2, :2],
            torch.tensor(
                [
                    [
                        [[[-7.6936, -11.7077], [-8.6289, -11.0604]]],
                        [[[-6.2675, -9.9616], [-6.5427, -9.0548]]],
                    ],
                    [
                        [[[-10.3143, -13.0117], [-10.2967, -12.3099]]],
                        [[[-9.1198, -10.1437], [-8.2902, -10.6460]]],
                    ],
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
        self.assertEqual(outputs.pred_masks.shape, (2, 4, 1, 288, 288))
        torch.testing.assert_close(
            outputs.iou_scores,
            torch.tensor(
                [
                    [[0.9862], [0.9666], [0.9588], [0.9331]],
                    [[0.9757], [0.9838], [0.9785], [0.9755]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            outputs.pred_masks[:, :, :, :2, :2],
            torch.tensor(
                [
                    [
                        [[[-12.5972, -19.5327], [-12.4126, -18.3935]]],
                        [[[-20.2715, -31.6163], [-22.3341, -27.6888]]],
                        [[[-20.9112, -31.4296], [-22.9174, -26.5892]]],
                        [[[-23.6995, -37.8614], [-26.3752, -31.1497]]],
                    ],
                    [
                        [[[-21.7436, -29.5702], [-24.3507, -25.5635]]],
                        [[[-28.0691, -38.6044], [-31.3014, -33.8172]]],
                        [[[-25.3085, -33.9384], [-27.7918, -30.1258]]],
                        [[[-26.7339, -36.4405], [-28.8027, -31.8549]]],
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
        self.assertEqual(outputs.pred_masks.shape, (1, 1, 1, 288, 288))
        torch.testing.assert_close(
            outputs.iou_scores, torch.tensor([[[0.9809]]]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            outputs.pred_masks[:, :, 0, :3, :3],
            torch.tensor(
                [
                    [
                        [
                            [-5.3111, -7.4920, -5.5444],
                            [-4.7685, -6.3513, -6.2969],
                            [-4.8471, -5.1722, -6.5492],
                        ]
                    ]
                ]
            ).to(torch_device),
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
        self.assertEqual(outputs.pred_masks.shape, (1, 1, 1, 288, 288))
        torch.testing.assert_close(
            outputs.iou_scores, torch.tensor([[[0.9625]]]).to(torch_device), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            outputs.pred_masks[:, :, 0, :3, :3],
            torch.tensor(
                [
                    [
                        [
                            [-13.4726, -19.9250, -16.3620],
                            [-13.5886, -18.7266, -17.6766],
                            [-14.6962, -19.3814, -19.9888],
                        ]
                    ]
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_dummy_pipeline_generation(self):
        generator = pipeline("mask-generation", model="facebook/sam3", device=torch_device)
        raw_image = prepare_image()

        _ = generator(raw_image, points_per_batch=64)
