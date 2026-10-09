# Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Qwen3.5 model."""

import copy
import tempfile
import unittest

from parameterized import parameterized

from transformers import DataCollatorWithFlattening, is_torch_available
from transformers.testing_utils import (
    require_causal_conv1d,
    require_flash_linear_attention,
    require_torch,
    require_torch_gpu,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import floats_tensor, ids_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        Qwen3_5MoeConfig,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeModel,
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeTextModel,
        Qwen3_5MoeVisionConfig,
    )


class Qwen3_5MoeTextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Qwen3_5MoeTextModel
        causal_lm_class = Qwen3_5MoeForCausalLM

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.hidden_act = "silu"
        self.layer_types = ["full_attention", "linear_attention"]
        self.linear_conv_kernel_dim = 2
        self.linear_key_head_dim = 16
        self.linear_value_head_dim = 16
        self.linear_num_key_heads = 4
        self.linear_num_value_heads = 8
        self.rope_parameters = {
            "rope_type": "default",
            "rope_theta": 10_000,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        }
        self.head_dim = 32


@require_torch
class Qwen3_5MoeTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Qwen3_5MoeTextModelTester
    config_class = Qwen3_5MoeTextConfig

    def _get_conv_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        num_k_heads = config.linear_num_key_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim
        intermediate_size = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

        return (batch_size, intermediate_size, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim

        return (batch_size, num_v_heads, head_k_dim, head_v_dim)

    def test_attention_outputs(self):
        "Needs to be overwritten as Qwen3.5 Moe alternates between attention layers and gated deltanet layers."
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

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
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])

    @unittest.skip("Intentionally not reversable (no changes) as only load time within a VLM depends on this")
    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        pass

    @require_causal_conv1d
    @require_flash_linear_attention
    @require_torch_gpu
    def test_padding_free_matches_padded_fast_path_regression(self):
        torch.manual_seed(0)
        config = self.model_tester.get_config()
        model = Qwen3_5MoeForCausalLM(config).to(torch_device).eval()

        data_collator = DataCollatorWithFlattening(
            return_tensors="pt", return_seq_idx=True, return_flash_attn_kwargs=True
        )
        test_cases = [
            (
                torch.tensor([[0, 0, 0, 1, 2, 3], [0, 0, 0, 0, 4, 5]], device=torch_device),
                torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]], dtype=torch.long, device=torch_device),
                [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}],
            ),
            (
                torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 6]], device=torch_device),
                torch.tensor([[0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1]], dtype=torch.long, device=torch_device),
                [{"input_ids": [1, 2, 3, 4, 5]}, {"input_ids": [6]}],
            ),
        ]

        for padded_input_ids, attention_mask, features in test_cases:
            position_ids = ((attention_mask == 1).long().cumsum(dim=1) - 1) * (attention_mask == 1).long()
            padding_free_batch = data_collator(features)
            padding_free_batch = {
                key: value.to(torch_device) if torch.is_tensor(value) else value
                for key, value in padding_free_batch.items()
            }

            with torch.no_grad():
                res_padded = model(
                    input_ids=padded_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                res_padfree = model(**padding_free_batch, use_cache=False)

            logits_padded = res_padded.logits[attention_mask.bool()]
            logits_padfree = res_padfree.logits[0]

            torch.testing.assert_close(logits_padded, logits_padfree, atol=1e-5, rtol=1e-5)


class Qwen3_5MoeVisionText2TextModelTester(VLMModelTester):
    base_model_class = Qwen3_5MoeModel
    config_class = Qwen3_5MoeConfig
    text_config_class = Qwen3_5MoeTextConfig
    vision_config_class = Qwen3_5MoeVisionConfig
    conditional_generation_class = Qwen3_5MoeForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bos_token_id", 0)
        kwargs.setdefault("eos_token_id", 1)
        kwargs.setdefault("pad_token_id", 2)
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("vision_start_token_id", 5)
        kwargs.setdefault("vision_end_token_id", 6)
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("num_image_tokens", 32)
        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("head_dim", 32)
        kwargs.setdefault("intermediate_size", 37)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("layer_types", ["full_attention", "linear_attention"])
        kwargs.setdefault(
            "rope_parameters",
            {"rope_type": "default", "rope_theta": 10_000, "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        )
        kwargs.setdefault("linear_conv_kernel_dim", 2)
        kwargs.setdefault("linear_key_head_dim", 16)
        kwargs.setdefault("linear_value_head_dim", 16)
        kwargs.setdefault("linear_num_key_heads", 4)
        kwargs.setdefault("linear_num_value_heads", 8)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("shared_expert_intermediate_size", 36)
        kwargs.setdefault("num_experts_per_tok", 2)
        kwargs.setdefault("num_experts", 8)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("spatial_merge_size", 1)
        kwargs.setdefault("temporal_patch_size", 2)
        kwargs.setdefault("num_position_embeddings", 16)
        kwargs.setdefault("vision_hidden_act", "gelu_pytorch_tanh")
        kwargs.setdefault("vision_intermediate_size", 32)
        super().__init__(parent, **kwargs)

        self.in_channels = self.num_channels
        self.out_hidden_size = self.hidden_size

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor(
            [
                batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels * (self.patch_size**2) * self.temporal_patch_size,
            ]
        )

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        }

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[:, -1] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id
        input_ids[:, self.num_image_tokens - 1] = self.vision_start_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        return {
            "image_grid_thw": torch.tensor([[1, 1, 1]] * batch_size, device=torch_device),
            "mm_token_type_ids": mm_token_type_ids,
        }

    def get_vision_config(self):
        vision_config = super().get_vision_config()
        vision_config.hidden_act = self.vision_hidden_act
        vision_config.intermediate_size = self.vision_intermediate_size
        return vision_config

    def get_config(self):
        return self.config_class(
            text_config=self.get_text_config().to_dict(),
            vision_config=self.get_vision_config().to_dict(),
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )


@require_torch
class Qwen3_5MoeModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Qwen3_5MoeVisionText2TextModelTester

    @parameterized.expand([("from_pretrained",), ("from_config",)])
    def test_automodelforcausallm(self, loader: str) -> None:
        """`AutoModelForCausalLM` must unwrap the text sub-config for composite-to-text-only mappings."""
        config = self.model_tester.get_config()
        self.assertIsInstance(config, Qwen3_5MoeConfig, msg="Test setup expects the composite Qwen3_5MoeConfig.")

        if loader == "from_config":
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config)
        else:
            full_model = Qwen3_5MoeForConditionalGeneration(config)
            with tempfile.TemporaryDirectory() as tmp_dir:
                full_model.save_pretrained(tmp_dir)
                model = AutoModelForCausalLM.from_pretrained(tmp_dir)

        self.assertIsInstance(model, Qwen3_5MoeForCausalLM)
        self.assertIsInstance(model.config, Qwen3_5MoeTextConfig)

    def test_automodelforcausallm_dtype(self) -> None:
        """`AutoModelForCausalLM` must honor a concrete `dtype`, overriding the saved composite dtype (#46459)."""
        config = self.model_tester.get_config()
        # The saved dtype (bf16) must differ from the requested one (fp32) to exercise the bug.
        full_model = Qwen3_5MoeForConditionalGeneration(config).to(torch.bfloat16)

        with tempfile.TemporaryDirectory() as tmp_dir:
            full_model.save_pretrained(tmp_dir)

            # #46459 regression: a concrete `dtype` must win over the saved bf16 composite dtype.
            model = AutoModelForCausalLM.from_pretrained(tmp_dir, dtype=torch.float32)
            self.assertIsInstance(model, Qwen3_5MoeForCausalLM)
            self.assertEqual(next(model.parameters()).dtype, torch.float32)

            # Default behavior is unchanged: `auto` and no dtype still load in the checkpoint's saved bf16.
            model_auto = AutoModelForCausalLM.from_pretrained(tmp_dir, dtype="auto")
            self.assertEqual(next(model_auto.parameters()).dtype, torch.bfloat16)
            model_default = AutoModelForCausalLM.from_pretrained(tmp_dir)
            self.assertEqual(next(model_default.parameters()).dtype, torch.bfloat16)

            # The legacy `torch_dtype` alias is honored the same way.
            model_legacy = AutoModelForCausalLM.from_pretrained(tmp_dir, torch_dtype=torch.float32)
            self.assertEqual(next(model_legacy.parameters()).dtype, torch.float32)

            # Non-regression guard: loading the whole VLM already honored `dtype` (this path does not
            # hit the text-config swap), and must keep doing so before and after the fix.
            vlm = AutoModelForImageTextToText.from_pretrained(tmp_dir, dtype=torch.float32)
            self.assertEqual(vlm.config.dtype, torch.float32)
            self.assertEqual(vlm.config.text_config.dtype, torch.float32)
            self.assertEqual(vlm.config.vision_config.dtype, torch.float32)
            # Check the actual weights load in fp32, not just the config metadata.
            self.assertTrue(all(param.dtype == torch.float32 for param in vlm.parameters()))

    @unittest.skip(
        "Conversion only for the `CausalLM` loading from saved `ConditionalLM`, doesn't apply to simple VLM"
    )
    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        pass

    def _get_conv_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        num_k_heads = config.linear_num_key_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim
        intermediate_size = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

        return (batch_size, intermediate_size, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        num_v_heads = config.linear_num_value_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim

        return (batch_size, num_v_heads, head_k_dim, head_v_dim)

    def test_attention_outputs(self):
        "Needs to be overwritten as Qwen3.5 Moe alternates between attention layers and gated deltanet layers."
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

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
            self.assertEqual(
                len(attentions), sum(layer == "full_attention" for layer in config.text_config.layer_types)
            )

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.text_config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(
                len(attentions), sum(layer == "full_attention" for layer in config.text_config.layer_types)
            )
            self.assertListEqual(
                list(attentions[0].shape[-3:]), [config.text_config.num_attention_heads, seq_len, seq_len]
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
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
                len(self_attentions), sum(layer == "full_attention" for layer in config.text_config.layer_types)
            )
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]), [config.text_config.num_attention_heads, seq_len, seq_len]
            )

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs throw an explicit error when image count mismatches image-token count in text.
        Also checks multi-image cases where one prompt has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**input_dict)
            curr_input_dict = copy.deepcopy(input_dict)

            patch_size = config.vision_config.patch_size
            one_img_length = (self.model_tester.image_size**2) // (patch_size**2)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-one_img_length:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            if hasattr(model.base_model, "rope_deltas"):
                model.base_model.rope_deltas = None

            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            mm_token_type_ids = curr_input_dict["mm_token_type_ids"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=torch.cat([mm_token_type_ids, mm_token_type_ids], dim=0),
                )

            if hasattr(model.base_model, "rope_deltas"):
                model.base_model.rope_deltas = None

            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            mm_token_type_ids = torch.cat(
                [curr_input_dict["mm_token_type_ids"][:1], curr_input_dict["mm_token_type_ids"][:1]], dim=0
            )
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )

    def test_image_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        bsz = self.model_tester.batch_size
        channels = config.vision_config.in_channels
        temporal_patch = config.vision_config.temporal_patch_size
        patch_size = config.vision_config.patch_size
        num_images = 2

        input_ids = ids_tensor([bsz, self.model_tester.seq_length], self.model_tester.vocab_size)
        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_end_token_id] = self.model_tester.pad_token_id

        patches_per_image = 1
        pixel_values = floats_tensor(
            [
                bsz * num_images * patches_per_image,
                channels * temporal_patch * (patch_size**2),
            ]
        )
        image_grid_thw = torch.tensor([[1, 1, 1]] * (bsz * num_images), device=torch_device)
        self.assertEqual(pixel_values.shape[0], image_grid_thw.prod(dim=1).sum().item())

        insertion_point = 0
        tokens_per_image = 3  # vision_start + image_token + vision_end
        required_seq_length = insertion_point + num_images * tokens_per_image
        self.assertLessEqual(required_seq_length, input_ids.shape[1])

        for b in range(bsz):
            for image_idx in range(num_images):
                image_start = insertion_point + image_idx * tokens_per_image
                input_ids[b, image_start] = self.model_tester.vision_start_token_id
                input_ids[b, image_start + 1] = self.model_tester.image_token_id
                input_ids[b, image_start + 2] = self.model_tester.vision_end_token_id

        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.model_tester.image_token_id] = 1

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.assertIsNotNone(outputs)

    def test_video_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        bsz = self.model_tester.batch_size
        channels = config.vision_config.in_channels
        temporal_patch = config.vision_config.temporal_patch_size
        patch_size = config.vision_config.patch_size

        input_ids = ids_tensor([bsz, self.model_tester.seq_length], self.model_tester.vocab_size)

        frames = 4
        num_video = 2
        frame_timestamp_tokens = 5
        patch_h = self.model_tester.image_size // patch_size
        patch_w = self.model_tester.image_size // patch_size
        patch_t = frames // temporal_patch
        patches_per_video = patch_t * patch_h * patch_w
        patches_per_frame = patch_h * patch_w
        pixel_values_videos = floats_tensor(
            [
                bsz * num_video * patches_per_video,
                channels * temporal_patch * (patch_size**2),
            ]
        )

        video_grid_thw = torch.tensor([[patch_t, patch_h, patch_w]] * (bsz * num_video), device=torch_device)
        self.assertEqual(pixel_values_videos.shape[0], video_grid_thw.prod(dim=1).sum().item())

        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_end_token_id] = self.model_tester.pad_token_id

        insertion_point = 0
        tokens_per_frame = frame_timestamp_tokens + 1 + patches_per_frame + 1
        tokens_per_video = patch_t * tokens_per_frame
        required_seq_length = insertion_point + num_video * tokens_per_video
        if required_seq_length > input_ids.shape[1]:
            pad_extension = torch.full(
                (bsz, required_seq_length - input_ids.shape[1]),
                self.model_tester.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, pad_extension], dim=1)

        timestamp_start_token_id = self.model_tester.vision_end_token_id + 1
        self.assertLessEqual(timestamp_start_token_id + frame_timestamp_tokens, self.model_tester.vocab_size)
        timestamp_token_ids = torch.arange(
            timestamp_start_token_id,
            timestamp_start_token_id + frame_timestamp_tokens,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

        self.assertLessEqual(required_seq_length, input_ids.shape[1])
        for b in range(bsz):
            for video_idx in range(num_video):
                video_start = insertion_point + video_idx * tokens_per_video
                for frame_idx in range(patch_t):
                    frame_start = video_start + frame_idx * tokens_per_frame
                    input_ids[b, frame_start : frame_start + frame_timestamp_tokens] = timestamp_token_ids

                    vision_start_pos = frame_start + frame_timestamp_tokens
                    input_ids[b, vision_start_pos] = self.model_tester.vision_start_token_id

                    frame_token_start = vision_start_pos + 1
                    frame_token_end = frame_token_start + patches_per_frame
                    input_ids[b, frame_token_start:frame_token_end] = self.model_tester.video_token_id
                    input_ids[b, frame_token_end] = self.model_tester.vision_end_token_id

        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.model_tester.video_token_id] = 2

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.assertIsNotNone(outputs)
