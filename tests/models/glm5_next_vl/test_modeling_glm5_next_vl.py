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
"""Testing suite for the PyTorch Glm5NextVL model."""

import unittest

from transformers import (
    Glm5NextVLConfig,
    Glm5NextVLForConditionalGeneration,
    Glm5NextVLModel,
    is_torch_available,
)
from transformers.models.glm5_next_vl.configuration_glm5_next_vl import Glm5NextVLTextConfig
from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class Glm5NextVLVisionText2TextModelTester(VLMModelTester):
    base_model_class = Glm5NextVLModel
    config_class = Glm5NextVLConfig
    text_config_class = Glm5NextVLTextConfig
    vision_config_class = GlmOcrVisionConfig
    conditional_generation_class = Glm5NextVLForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("video_start_token_id", 3)
        kwargs.setdefault("video_end_token_id", 4)
        kwargs.setdefault("image_start_token_id", 5)
        kwargs.setdefault("image_end_token_id", 6)
        kwargs.setdefault("image_token_id", 7)
        kwargs.setdefault("video_token_id", 8)
        kwargs.setdefault("image_size", 112)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("num_image_tokens", 64)
        kwargs.setdefault("seq_length", 64 + 7)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 16)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("num_experts_per_tok", 4)
        kwargs.setdefault("n_routed_experts", 8)
        kwargs.setdefault("num_local_experts", 8)
        kwargs.setdefault("linear_num_heads", 2)
        kwargs.setdefault("linear_head_dim", 16)
        kwargs.setdefault("linear_conv_kernel_dim", 2)
        kwargs.setdefault("v_head_dim", 16)
        kwargs.setdefault("qk_rope_head_dim", 0)
        kwargs.setdefault("qk_nope_head_dim", 64)
        kwargs.setdefault("q_lora_rank", 32)
        kwargs.setdefault("kv_lora_rank", 16)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("spatial_merge_size", 1)
        kwargs.setdefault("temporal_patch_size", 2)
        kwargs.setdefault("hidden_size", 48)
        kwargs.setdefault("intermediate_size", 16)
        kwargs.setdefault("mlp_layer_types", ["dense", "sparse"])
        # TODO: add indexer stuff when finished training
        kwargs.setdefault("layer_types", ["linear_attention", "full_attention"])
        super().__init__(parent, **kwargs)

    def create_pixel_values(self):
        return floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels * (self.patch_size**2) * self.temporal_patch_size,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        # Clear any accidental special tokens first
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.video_start_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_start_token_id] = self.pad_token_id
        input_ids[input_ids == self.video_end_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_end_token_id] = self.pad_token_id
        # Place image tokens with image start/end prefix/suffix
        input_ids[:, 0] = self.image_start_token_id
        input_ids[:, 1 : 1 + self.num_image_tokens] = self.image_token_id
        input_ids[:, 1 + self.num_image_tokens] = self.image_end_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[:, 1 : 1 + self.num_image_tokens] = 1
        patches_per_side = self.image_size // self.patch_size
        return {
            "image_grid_thw": torch.tensor(
                [[1, patches_per_side, patches_per_side]] * self.batch_size, device=torch_device
            ),
            "mm_token_type_ids": mm_token_type_ids,
        }

    def get_vision_config(self):
        return self.vision_config_class(
            depth=self.depth,
            hidden_act=self.hidden_act,
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            out_hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            patch_size=self.patch_size,
            spatial_merge_size=self.spatial_merge_size,
            temporal_patch_size=self.temporal_patch_size,
        )

    def get_config(self):
        return self.config_class(
            text_config=self.get_text_config(),
            vision_config=self.get_vision_config(),
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            video_start_token_id=self.video_start_token_id,
            video_end_token_id=self.video_end_token_id,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
        )


@require_torch
class Glm5NextVLModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Glm5NextVLVisionText2TextModelTester
    test_all_params_have_gradient = False  # MoE
    model_split_percents = [0.5, 0.8, 0.9]

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        """Override similar to GLM4V: images shaped as (bs*patch_len, dim) so we can't slice to batches in generate"""
        config, inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size)
        _, full_inputs = self.model_tester.prepare_config_and_inputs_for_common()

        num_patches = int(inputs_dict["image_grid_thw"].prod(-1).sum().item())
        inputs_dict["pixel_values"] = full_inputs["pixel_values"][:num_patches]

        return config, inputs_dict

    def _get_conv_state_shape(self, batch_size: int, config):
        return (batch_size, 3 * config.linear_num_heads * config.linear_head_dim, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.linear_num_heads, config.linear_head_dim, config.linear_head_dim)

    def _get_attention_shape(self, batch_size: int, seq_length: int, config):
        # (batch, head, seq_length, head_features)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        return expected_key_shape, expected_value_shape

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        """Override to account for the difference in MHC and the final state shapes"""
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (output_length - prompt_length))

        # When `output_hidden_states=True`, each iteration of generate appends the hidden states corresponding to the
        # new token(s)
        # NOTE: `StaticCache` may have different lengths on different layers, if this test starts failing add more
        # elaborate checks
        for generated_length, iter_hidden_states in enumerate(hidden_states):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length

            # We have raw MHC shapes until the final one which is collapsed
            mhc_shape = (batch_size, model_input_length, config.hc_mult, config.hidden_size)
            final_shape = (batch_size, model_input_length, config.hidden_size)
            expected_shapes = [mhc_shape] * (len(iter_hidden_states) - 1)
            expected_shapes.append(final_shape)

            # check hidden size
            self.assertListEqual(
                [state.shape for state in iter_hidden_states],
                expected_shapes,
            )

    def test_attention_outputs(self):
        """Needs to be overwritten as GLM5 Next VL alternates between attention layers and KDA layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        config.return_dict = True
        text_config = config.get_text_config()

        # Force eager attention to support output attentions.
        text_config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True

            model = model_class._from_config(config, attn_implementation="eager")
            config = model.config
            text_config = config.get_text_config()

            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs.attentions
            self.assertEqual(
                len(attentions),
                sum(layer == "full_attention" for layer in text_config.layer_types),
            )

            # Check that output_attentions also works through config.
            del inputs_dict["output_attentions"]
            text_config.output_attentions = True

            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs.attentions
            self.assertEqual(
                len(attentions),
                sum(layer == "full_attention" for layer in text_config.layer_types),
            )
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [text_config.num_attention_heads, seq_len, seq_len],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine.
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True

            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(
                len(self_attentions),
                sum(layer == "full_attention" for layer in text_config.layer_types),
            )
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [text_config.num_attention_heads, seq_len, seq_len],
            )

    def test_hidden_states_output(self):
        """Override to account for the difference in MHC and the final state shapes"""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config.output_hidden_states = True

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            text_config = model.config.get_text_config()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), text_config.num_hidden_layers + 1)

            batch_size, seq_len = inputs_dict["input_ids"].shape

            # Raw MHC shapes
            for layer_hidden_states in hidden_states[:-1]:
                self.assertEqual(
                    layer_hidden_states.shape,
                    (
                        batch_size,
                        seq_len,
                        text_config.hc_mult,
                        text_config.hidden_size,
                    ),
                )

            # Final output is standard again
            self.assertEqual(
                hidden_states[-1].shape,
                (
                    batch_size,
                    seq_len,
                    text_config.hidden_size,
                ),
            )

    @unittest.skip("MLA creates different head dims which avoids invoking the FA backend")
    def test_sdpa_can_dispatch_on_flash(self):
        pass


@require_torch_accelerator
@slow
@unittest.skip(reason="No model weights yet, add after release")
class Glm5NextVLIntegrationTest(unittest.TestCase):
    pass
