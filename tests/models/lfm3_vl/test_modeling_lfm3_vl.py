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
"""Testing suite for the PyTorch LFM3-VL model."""

import unittest

from transformers import (
    AutoProcessor,
    Lfm3Config,
    Lfm3VlConfig,
    Lfm3VlForConditionalGeneration,
    Lfm3VlModel,
    Lfm3VlVisionConfig,
    is_torch_available,
)
from transformers.models.lfm3_vl.modeling_lfm3_vl import Lfm3VlVisionModel
from transformers.testing_utils import Expectations, require_torch, require_torch_accelerator, slow, torch_device

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor, ids_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch
    import torch.nn.functional as F


class Lfm3VlVisionText2TextModelTester(VLMModelTester):
    base_model_class = Lfm3VlModel
    config_class = Lfm3VlConfig
    text_config_class = Lfm3Config
    vision_config_class = Lfm3VlVisionConfig
    conditional_generation_class = Lfm3VlForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("vision_start_token_id", 5)
        kwargs.setdefault("vision_end_token_id", 6)
        # one 2x2 patch grid per image, merged into a single language token
        kwargs.setdefault("image_size", 4)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("temporal_patch_size", 1)
        kwargs.setdefault("spatial_merge_size", 2)
        kwargs.setdefault("num_image_tokens", 1)
        kwargs.setdefault("in_channels", 3)
        # head_dim 16: the FA4 kernels need at least 16
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("num_key_value_heads", 1)
        kwargs.setdefault("layer_types", ["conv", "full_attention"])
        kwargs.setdefault("conv_L_cache", 3)
        kwargs.setdefault("projector_hidden_size", 32)
        kwargs.setdefault(
            "rope_parameters",
            {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.5,
                },
            },
        )
        super().__init__(parent, **kwargs)

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        patches_per_image = (self.image_size // self.patch_size) ** 2
        return floats_tensor(
            [batch_size * patches_per_image, self.in_channels * self.temporal_patch_size * self.patch_size**2]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        # stray vision ids in the random text would be placeholders without features, which the model refuses
        for token_id in (
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        ):
            input_ids[input_ids == token_id] = self.bos_token_id
        input_ids[:, 0] = self.vision_start_token_id
        input_ids[:, 1] = self.image_token_id
        input_ids[:, 2] = self.vision_end_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        grid = self.image_size // self.patch_size
        return {"image_grid_thw": torch.tensor([[1, grid, grid]] * batch_size, device=torch_device)}


@require_torch
class Lfm3VlModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Lfm3VlVisionText2TextModelTester

    def _get_conv_state_shape(self, batch_size: int, config):
        return (batch_size, config.hidden_size, config.conv_L_cache)

    def test_attention_outputs(self):
        """Only the attention layers of the language model record attentions, the conv layers do not."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        seq_len = self.model_tester.seq_length
        num_attention_layers = sum(layer_type != "conv" for layer_type in config.text_config.layer_types)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), num_attention_layers)
            self.assertListEqual(
                list(attentions[0].shape[-3:]), [config.text_config.num_attention_heads, seq_len, seq_len]
            )

    @property
    def _special_token_ids(self):
        tester = self.model_tester
        return {tester.image_token_id, tester.video_token_id, tester.vision_start_token_id, tester.vision_end_token_id}

    def _text_input_ids(self, batch_size: int, seq_length: int, config):
        input_ids = ids_tensor([batch_size, seq_length], config.text_config.vocab_size)
        reserved = self._special_token_ids | {config.text_config.pad_token_id}
        for token_id in reserved:
            input_ids[input_ids == token_id] = max(reserved) + 1
        return input_ids

    def _image_features_get_expected_num_attentions(self, model_tester=None):
        return self.model_tester.layer_types.count("full_attention")

    def _video_features_get_expected_num_attentions(self, model_tester=None):
        return self.model_tester.layer_types.count("full_attention")

    def _image_features_get_expected_num_hidden_states(self, model_tester=None):
        return self.model_tester.num_hidden_layers + 1

    def _video_features_get_expected_num_hidden_states(self, model_tester=None):
        return self.model_tester.num_hidden_layers + 1

    def _random_patches(self, config, grid_thw):
        patch_dim = (
            config.vision_config.in_channels
            * config.vision_config.temporal_patch_size
            * config.vision_config.patch_size**2
        )
        return floats_tensor([int(grid_thw.prod(-1).sum()), patch_dim])

    def test_video_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        merge = config.vision_config.spatial_merge_size
        video_grid_thw = torch.tensor([[3, 2, 4]], device=torch_device)
        tokens_per_frame = int(video_grid_thw[0, 1:].prod()) // merge**2
        pixel_values_videos = self._random_patches(config, video_grid_thw)

        input_ids = self._text_input_ids(1, 30, config)
        position = 2
        for _ in range(int(video_grid_thw[0, 0])):
            input_ids[0, position] = config.vision_start_token_id
            input_ids[0, position + 1 : position + 1 + tokens_per_frame] = config.video_token_id
            input_ids[0, position + 1 + tokens_per_frame] = config.vision_end_token_id
            position += tokens_per_frame + 3

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids, pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw
                )
            self.assertEqual(
                outputs.video_hidden_states.shape,
                (int(video_grid_thw[0, 0]) * tokens_per_frame, config.text_config.hidden_size),
            )
            short_input_ids = input_ids.clone()
            short_input_ids[0, 3] = config.vision_end_token_id + 1
            with self.assertRaises(ValueError):
                model(
                    input_ids=short_input_ids, pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw
                )

    def test_image_and_video_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        merge = config.vision_config.spatial_merge_size
        video_grid_thw = torch.tensor([[2, 2, 2]], device=torch_device)
        tokens_per_frame = int(video_grid_thw[0, 1:].prod()) // merge**2
        pixel_values_videos = self._random_patches(config, video_grid_thw)

        input_ids = inputs["input_ids"].clone()
        input_ids[input_ids == config.video_token_id] = config.vision_end_token_id + 1
        input_ids[0, 4] = config.video_token_id
        input_ids[0, 5] = config.video_token_id
        self.assertEqual(int((input_ids == config.video_token_id).sum()), 2 * tokens_per_frame)

        model = Lfm3VlForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values=inputs["pixel_values"],
                image_grid_thw=inputs["image_grid_thw"],
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
        self.assertEqual(outputs.logits.shape[:2], input_ids.shape)
        with torch.no_grad():
            embeds = model.get_input_embeddings()(input_ids)
            embeds[input_ids == config.image_token_id] = outputs.image_hidden_states
            embeds[input_ids == config.video_token_id] = outputs.video_hidden_states
            expected = model.lm_head(model.model.language_model(inputs_embeds=embeds).last_hidden_state)
        torch.testing.assert_close(outputs.logits, expected)

    def test_vision_tower_packs_frames_like_single_frames(self):
        """Every frame is an independent sequence, so a packed batch must match the frames encoded one by one."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        vision_tower = Lfm3VlVisionModel(config.vision_config).to(torch_device).eval()
        grid_thw = torch.tensor([[2, 2, 4], [1, 4, 2]], device=torch_device)
        pixel_values = self._random_patches(config, grid_thw)

        with torch.no_grad():
            packed = vision_tower(pixel_values, grid_thw=grid_thw).last_hidden_state
            offset = 0
            for temporal, height, width in grid_thw.tolist():
                for _ in range(temporal):
                    frame = pixel_values[offset : offset + height * width]
                    single = vision_tower(
                        frame, grid_thw=torch.tensor([[1, height, width]], device=torch_device)
                    ).last_hidden_state
                    torch.testing.assert_close(packed[offset : offset + height * width], single, rtol=1e-4, atol=1e-4)
                    offset += height * width

    def test_vision_conv_runs_over_the_raster_grid(self):
        """Patches arrive grouped by merge block; the depthwise convolution must see the raster grid of each frame."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        vision_tower = Lfm3VlVisionModel(config.vision_config).to(torch_device).eval()
        conv_layer = next(layer.conv for layer in vision_tower.layers if not layer.is_attention_layer)
        merge = config.vision_config.spatial_merge_size
        height, width = 4, 6
        # inputs that carry signal through the two projections and the gate, so the tolerances below are meaningful
        hidden_states = torch.randn(height * width, config.vision_config.hidden_size, device=torch_device) * 8

        with torch.no_grad():
            output = conv_layer(hidden_states, frame_shapes=[[height, width]])
            # reference: reorder to raster, gate, convolve the (height, width) grid, gate again
            raster = (
                hidden_states.view(height // merge, width // merge, merge, merge, -1)
                .permute(0, 2, 1, 3, 4)
                .reshape(height, width, -1)
            )
            B, C, x = conv_layer.in_proj(raster).chunk(3, dim=-1)
            gated = (B * x).permute(2, 0, 1).unsqueeze(0)
            convolved = F.conv2d(gated, conv_layer.spatial_conv.weight, padding=1, groups=gated.shape[1])
            expected = conv_layer.out_proj(C * convolved[0].permute(1, 2, 0))
            expected = expected.view(height // merge, merge, width // merge, merge, -1).permute(0, 2, 1, 3, 4)
        self.assertGreater(output.abs().max().item(), 1e-3)
        torch.testing.assert_close(output, expected.reshape(height * width, -1), rtol=1e-5, atol=1e-6)

    @unittest.skip("The vision tower splits frames by host-side sizes, which `torch.export` can't trace")
    def test_torch_export_dynamic(self):
        pass

    @unittest.skip("The vision tower splits frames by host-side sizes, which `torch.export` can't trace")
    def test_torch_export_static(self):
        pass

    @unittest.skip(
        "The tiny text model has no sliding-attention layer, whose `rope_parameters` entry these tests scale"
    )
    def test_model_rope_scaling_frequencies(self):
        pass

    @unittest.skip(
        "The tiny text model has no sliding-attention layer, whose `rope_parameters` entry these tests scale"
    )
    def test_model_rope_scaling_from_config_0_linear(self):
        pass

    @unittest.skip(
        "The tiny text model has no sliding-attention layer, whose `rope_parameters` entry these tests scale"
    )
    def test_model_rope_scaling_from_config_1_dynamic(self):
        pass

    @unittest.skip(
        "The tiny text model has no sliding-attention layer, whose `rope_parameters` entry these tests scale"
    )
    def test_model_rope_scaling_from_config_2_yarn(self):
        pass


@require_torch_accelerator
@slow
class Lfm3VlIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "LiquidAI/LFM3-VL-2B"

    def setUp(self):
        super().setUp()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        from transformers.image_utils import load_image

        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/coco_sample.png"
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
        self.inputs = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

    def test_model_logits(self):
        model = Lfm3VlForConditionalGeneration.from_pretrained(
            self.model_id, device_map=torch_device, dtype=torch.float32
        )
        self.assertEqual(int((self.inputs["input_ids"] == self.processor.image_token_id).sum()), 300)
        with torch.no_grad():
            logits = model(**self.inputs).logits.float()

        expectations = Expectations(
            {
                (None, None): [
                    -8.934641,
                    -8.601767,
                    -9.073932,
                    -6.786697,
                    -7.16684,
                    -6.331409,
                    -7.234934,
                    -3.480811,
                    1.642203,
                    0.016106,
                    4.17064,
                    1.260395,
                    4.941868,
                    -0.93643,
                    -2.70716,
                    0.525768,
                ],
            }
        )
        expected_slice = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(logits[0, -1, :16], expected_slice, rtol=1e-3, atol=1e-3)

    def test_model_greedy_generation(self):
        model = Lfm3VlForConditionalGeneration.from_pretrained(
            self.model_id, device_map=torch_device, dtype=torch.bfloat16
        )
        generated_ids = model.generate(**self.inputs, max_new_tokens=24, do_sample=False)
        text = self.processor.batch_decode(
            generated_ids[:, self.inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]

        expectations = Expectations(
            {
                (None, None): (
                    "The user wants a one-sentence description of the image. I need to observe carefully: "
                    "two tabby cats lying on"
                ),
            }
        )
        self.assertEqual(text, expectations.get_expectation())
