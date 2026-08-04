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
"""Tests for the native MiniMax-VL-01 PyTorch model."""

import unittest

import pytest

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    CLIPVisionConfig,
    MiniMaxConfig,
    MiniMaxVL01Config,
    MiniMaxVL01ForConditionalGeneration,
    MiniMaxVL01Model,
    is_torch_available,
)
from transformers.testing_utils import require_torch, torch_device

from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers.models.minimax.modeling_minimax import MiniMaxCache
    from transformers.models.minimax_vl_01.modeling_minimax_vl_01 import MiniMaxVL01TextCache


class MiniMaxVL01VisionText2TextModelTester(VLMModelTester):
    base_model_class = MiniMaxVL01Model
    config_class = MiniMaxVL01Config
    conditional_generation_class = MiniMaxVL01ForConditionalGeneration
    text_config_class = MiniMaxConfig
    vision_config_class = CLIPVisionConfig

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("vocab_size", 97)
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("intermediate_size", 16)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 64)
        kwargs.setdefault("num_experts", 2)
        kwargs.setdefault("num_experts_per_tok", 2)
        kwargs.setdefault("layer_types", ["linear_attention", "full_attention"])
        kwargs.setdefault("block_size", 4)
        kwargs.setdefault(
            "rope_parameters",
            {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.5},
        )
        kwargs.setdefault("image_size", 8)
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("projection_dim", 32)
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("num_patches_per_image", 2)
        # Four base-image patch tokens plus four any-resolution patch tokens and two row newlines.
        kwargs.setdefault("num_image_tokens", 10)
        super().__init__(parent, **kwargs)

    def get_vision_config(self):
        return CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            projection_dim=self.hidden_size,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_channels=self.num_channels,
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_act="gelu",
        )

    def get_config(self):
        config = MiniMaxVL01Config(
            text_config=self.get_text_config(),
            vision_config=self.get_vision_config(),
            image_token_index=self.image_token_id,
            image_grid_pinpoints=[[self.image_size, self.image_size]],
            projector_hidden_act="gelu",
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
        )
        config.text_config._attn_implementation = "eager"
        return config

    def create_attention_mask(self, input_ids):
        return torch.ones_like(input_ids, device=torch_device)

    def create_pixel_values(self):
        return floats_tensor(
            [
                self.batch_size,
                self.num_patches_per_image,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        return {"image_sizes": torch.tensor([[self.image_size, self.image_size]] * self.batch_size)}


@require_torch
class MiniMaxVL01ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MiniMaxVL01VisionText2TextModelTester
    skip_test_image_features_output_shape = True

    def test_reverse_loading_mapping(self):
        # The released checkpoint prefixes target the conditional model's `model` subtree, not the bare base model.
        super().test_reverse_loading_mapping(skip_base_model=True)

    def test_text_cache_resolves_first_full_attention_layer(self):
        cache = MiniMaxVL01TextCache()
        key_states = torch.zeros(2, 2, 5, 8)
        cache.update(key_states, key_states, layer_idx=7)

        self.assertEqual(cache._get_attention_layer_idx(0), 7)
        self.assertEqual(cache._get_attention_layer_idx(7), 7)

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_past_key_values
    ):
        text_config = config.get_text_config(decoder=True)
        self.assertIsInstance(attentions, tuple)
        self.assertEqual(len(attentions), output_length - prompt_length)
        use_cache = decoder_past_key_values is not None
        head_dim = text_config.head_dim or text_config.hidden_size // text_config.num_attention_heads

        for generated_length, iteration_attentions in enumerate(attentions):
            self.assertIsInstance(iteration_attentions, tuple)
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length
            full_attention_shape = (
                batch_size,
                text_config.num_attention_heads,
                model_input_length,
                prompt_length + generated_length,
            )
            recurrent_state_shape = (
                batch_size,
                text_config.num_attention_heads,
                head_dim,
                head_dim,
            )
            for layer_type, layer_attention in zip(text_config.layer_types, iteration_attentions):
                expected_shape = full_attention_shape if layer_type == "full_attention" else recurrent_state_shape
                self.assertEqual(tuple(layer_attention.shape), expected_shape)

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        text_config = config.get_text_config(decoder=True)
        self.assertIsInstance(past_key_values, MiniMaxCache)
        head_dim = text_config.head_dim or text_config.hidden_size // text_config.num_attention_heads
        full_attention_shape = (
            batch_size,
            text_config.num_key_value_heads,
            seq_length,
            head_dim,
        )
        recurrent_state_shape = (
            batch_size,
            text_config.num_attention_heads,
            head_dim,
            head_dim,
        )

        for layer_idx, layer_type in enumerate(text_config.layer_types):
            if layer_type == "full_attention":
                self.assertEqual(tuple(past_key_values.layers[layer_idx].keys.shape), full_attention_shape)
                self.assertEqual(tuple(past_key_values.layers[layer_idx].values.shape), full_attention_shape)
            else:
                self.assertEqual(tuple(past_key_values.linear_cache[layer_idx].shape), recurrent_state_shape)

    def _check_caches_are_equal(self, cache1: MiniMaxCache, cache2: MiniMaxCache):
        self.assertIsInstance(cache1, MiniMaxCache)
        self.assertIsInstance(cache2, MiniMaxCache)
        self.assertEqual(len(cache1), len(cache2))

        for layer_idx in range(len(cache1)):
            if layer_idx < len(cache1.layers) and cache1.layers[layer_idx].is_initialized:
                torch.testing.assert_close(cache1.layers[layer_idx].keys, cache2.layers[layer_idx].keys)
                torch.testing.assert_close(cache1.layers[layer_idx].values, cache2.layers[layer_idx].values)
            if layer_idx < len(cache1.linear_cache) and not isinstance(cache1.linear_cache[layer_idx], list):
                torch.testing.assert_close(cache1.linear_cache[layer_idx], cache2.linear_cache[layer_idx])

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

    @pytest.mark.xfail(reason="The multimodal projector is not exercised by text-only checkpointing tests.")
    def test_training_gradient_checkpointing(self):
        super().test_training_gradient_checkpointing()

    @pytest.mark.xfail(reason="The multimodal projector is not exercised by text-only checkpointing tests.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        super().test_training_gradient_checkpointing_use_reentrant_false()

    @pytest.mark.xfail(reason="The multimodal projector is not exercised by text-only checkpointing tests.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        super().test_training_gradient_checkpointing_use_reentrant_true()

    @unittest.skip("Mixed full and Lightning attention uses the MiniMax-specific attention output contract.")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Lightning-attention outputs are recurrent states and are not part of the logits gradient path.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("MiniMaxCache does not implement cache cropping for assisted decoding.")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip("MiniMaxCache does not implement cache cropping for assisted decoding.")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("MiniMaxCache does not implement cache cropping for assisted decoding.")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip("MiniMaxCache does not implement cache cropping for assisted decoding.")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    def _single_image_inputs(self):
        config = self.model_tester.get_config()
        pixel_values = floats_tensor([1, 2, 3, 8, 8]).to(torch_device)
        input_ids = torch.tensor([[3] * 10 + [7, 8]], device=torch_device)
        return config, {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": pixel_values,
            "image_sizes": torch.tensor([[8, 8]], device=torch_device),
        }

    def test_flat_batched_and_list_pixel_values_match(self):
        config, inputs = self._single_image_inputs()
        model = MiniMaxVL01ForConditionalGeneration(config).to(torch_device).eval()

        pixel_values_5d = inputs["pixel_values"]
        pixel_values_4d = pixel_values_5d.flatten(0, 1)
        list_of_patch_batches = [pixel_values_5d[0]]
        list_of_patches = list(pixel_values_5d[0].unbind(0))

        with torch.no_grad():
            reference = model(**inputs).logits
            for pixel_values in (pixel_values_4d, list_of_patch_batches, list_of_patches):
                candidate_inputs = {**inputs, "pixel_values": pixel_values}
                candidate = model(**candidate_inputs).logits
                torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-5)

    def test_attention_mask_is_applied_to_loss(self):
        config = self.model_tester.get_config()
        model = MiniMaxVL01ForConditionalGeneration(config).to(torch_device).eval()
        input_ids = torch.tensor([[5, 6, 7, 8, 9]], device=torch_device)
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]], device=torch_device)
        labels = input_ids.clone()

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        shifted_mask = attention_mask[:, 1:].bool()
        shifted_logits = output.logits[:, :-1][shifted_mask]
        shifted_labels = labels[:, 1:][shifted_mask]
        expected_loss = torch.nn.functional.cross_entropy(shifted_logits, shifted_labels)
        torch.testing.assert_close(output.loss, expected_loss)

    def test_empty_pixel_tensor_uses_text_only_path(self):
        config = self.model_tester.get_config()
        model = MiniMaxVL01ForConditionalGeneration(config).to(torch_device).eval()
        input_ids = torch.tensor([[5, 6, 7, 8]], device=torch_device)

        with torch.no_grad():
            text_only = model(input_ids=input_ids).logits
            with_empty_pixels = model(
                input_ids=input_ids,
                pixel_values=torch.empty((0, 3, 8, 8), device=torch_device),
                image_sizes=torch.empty((0, 2), dtype=torch.long, device=torch_device),
            ).logits

        torch.testing.assert_close(with_empty_pixels, text_only)

    def test_hybrid_cache_prefill_decode_matches_full_forward(self):
        config = self.model_tester.get_config()
        model = MiniMaxVL01ForConditionalGeneration(config).to(torch_device).eval()
        prompt_ids = torch.tensor([[5, 6, 7, 8]], device=torch_device)
        next_id = torch.tensor([[9]], device=torch_device)

        with torch.no_grad():
            prefill = model(input_ids=prompt_ids, use_cache=True)
            decoded = model(input_ids=next_id, past_key_values=prefill.past_key_values, use_cache=True)
            full = model(input_ids=torch.cat((prompt_ids, next_id), dim=-1), use_cache=False)

        self.assertIsInstance(prefill.past_key_values, MiniMaxCache)
        torch.testing.assert_close(decoded.logits[:, -1], full.logits[:, -1], rtol=1e-4, atol=1e-4)

    def test_multimodal_greedy_generation_uses_cache(self):
        config, inputs = self._single_image_inputs()
        config.eos_token_id = None
        config.text_config.eos_token_id = None
        model = MiniMaxVL01ForConditionalGeneration(config).to(torch_device).eval()

        generated = model.generate(**inputs, do_sample=False, max_new_tokens=2)

        self.assertEqual(generated.shape, (1, inputs["input_ids"].shape[1] + 2))
        torch.testing.assert_close(generated[:, : inputs["input_ids"].shape[1]], inputs["input_ids"])

    def test_auto_model_mappings(self):
        config = self.model_tester.get_config()

        self.assertIsInstance(AutoModel.from_config(config), MiniMaxVL01Model)
        self.assertIsInstance(AutoModelForCausalLM.from_config(config), MiniMaxVL01ForConditionalGeneration)
        self.assertIsInstance(AutoModelForImageTextToText.from_config(config), MiniMaxVL01ForConditionalGeneration)


if __name__ == "__main__":
    unittest.main()
