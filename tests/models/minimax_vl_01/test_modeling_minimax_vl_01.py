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

import tempfile
import unittest
from unittest.mock import Mock

from parameterized import parameterized

from transformers import (
    CLIPVisionConfig,
    MiniMaxVL01Config,
    MiniMaxVL01ForConditionalGeneration,
    MiniMaxVL01Model,
    MiniMaxVL01TextConfig,
    is_torch_available,
)
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers.models.minimax_vl_01.modeling_minimax_vl_01 import (
        MiniMaxVL01TextCache,
        MiniMaxVL01TextLightningAttention,
    )


class MiniMaxVL01VisionText2TextModelTester(VLMModelTester):
    base_model_class = MiniMaxVL01Model
    config_class = MiniMaxVL01Config
    conditional_generation_class = MiniMaxVL01ForConditionalGeneration
    text_config_class = MiniMaxVL01TextConfig
    vision_config_class = CLIPVisionConfig

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 2)
        kwargs.setdefault("vocab_size", 97)
        kwargs.setdefault("intermediate_size", 16)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 64)
        kwargs.setdefault("num_experts", 2)
        kwargs.setdefault("layer_types", ["linear_attention", "full_attention"])
        kwargs.setdefault("block_size", 4)
        kwargs.setdefault(
            "rope_parameters",
            {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.5},
        )
        kwargs.setdefault("num_patches_per_image", 2)
        patches_per_side = kwargs.get("image_size", 8) // kwargs.get("patch_size", 4)
        # The thumbnail and crop each contribute a patch grid, with one newline per crop row.
        kwargs.setdefault("num_image_tokens", patches_per_side**2 + patches_per_side * (patches_per_side + 1))
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
        config = super().get_config()
        config.image_grid_pinpoints = [[self.image_size, self.image_size]]
        return config

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor(
            [
                batch_size,
                self.num_patches_per_image,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {"image_sizes": torch.tensor([[self.image_size, self.image_size]] * batch_size)}


@require_torch
class MiniMaxVL01ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MiniMaxVL01VisionText2TextModelTester
    skip_test_image_features_output_shape = True
    # CLIP's pooling LayerNorm is unused: VL-01 consumes encoder hidden states instead of pooled output.
    test_all_params_have_gradient = False

    def test_reverse_loading_mapping(self):
        # The released checkpoint prefixes target the conditional model's `model` subtree, not the bare base model.
        super().test_reverse_loading_mapping(skip_base_model=True)

    # Generation never calls `batch_repeat_interleave`, and the beam-search tests do not compare cached and uncached
    # outputs, so the common tests cannot catch a regression in these MiniMaxCache fixes.
    def test_text_cache_batch_repeat_interleave_handles_final_full_attention_layer(self):
        cache = MiniMaxVL01TextCache()
        recurrent_state = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)
        full_attention_state = torch.tensor([10.0, 20.0]).reshape(2, 1, 1, 1)
        cache.set_linear_cache(layer_idx=0, linear_cache=recurrent_state)
        cache.update(full_attention_state, full_attention_state, layer_idx=1)

        cache.batch_repeat_interleave(2)

        expected_recurrent = torch.tensor([1.0, 1.0, 2.0, 2.0]).reshape(4, 1, 1, 1)
        expected_full = torch.tensor([10.0, 10.0, 20.0, 20.0]).reshape(4, 1, 1, 1)
        torch.testing.assert_close(cache.linear_cache[0], expected_recurrent)
        torch.testing.assert_close(cache.layers[1].keys, expected_full)
        torch.testing.assert_close(cache.layers[1].values, expected_full)

    @parameterized.expand([("batch_select_indices",), ("reorder_cache",)])
    def test_text_cache_selection_reorders_recurrent_and_full_attention_layers(self, method_name):
        cache = MiniMaxVL01TextCache()
        recurrent_state = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)
        full_attention_state = torch.tensor([10.0, 20.0]).reshape(2, 1, 1, 1)
        cache.set_linear_cache(layer_idx=0, linear_cache=recurrent_state)
        cache.update(full_attention_state, full_attention_state, layer_idx=1)

        getattr(cache, method_name)(torch.tensor([1, 0]))

        expected_recurrent = torch.tensor([2.0, 1.0]).reshape(2, 1, 1, 1)
        expected_full = torch.tensor([20.0, 10.0]).reshape(2, 1, 1, 1)
        torch.testing.assert_close(cache.linear_cache[0], expected_recurrent)
        torch.testing.assert_close(cache.layers[1].keys, expected_full)
        torch.testing.assert_close(cache.layers[1].values, expected_full)

    def test_text_lightning_attention_slope_handles_single_layer(self):
        config = self.model_tester.get_text_config()
        config.num_hidden_layers = 1
        config.layer_types = ["linear_attention"]
        attention = MiniMaxVL01TextLightningAttention(config, layer_idx=0)
        slope_rate = attention.get_slope_rate(device=torch_device)

        self.assertEqual(slope_rate.shape, (config.num_attention_heads, 1, 1))
        self.assertEqual(slope_rate.dtype, torch.float32)
        self.assertEqual(slope_rate.device.type, torch.device(torch_device).type)
        self.assertTrue(torch.isfinite(slope_rate).all())

    def test_text_lightning_attention_source_padding_and_fp32_cache(self):
        # The common padding tests cannot catch a missing value mask: tiny test models zero the pad-token embedding,
        # so padded positions carry no signal. They also run in float32, which hides a low-precision recurrent state.
        config = MiniMaxVL01TextConfig(
            vocab_size=97,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            num_local_experts=2,
            num_experts_per_tok=2,
            layer_types=["linear_attention"] * 2,
            block_size=2,
        )
        attention = MiniMaxVL01TextLightningAttention(config, layer_idx=0).to(dtype=torch.float16)
        with torch.no_grad():
            identity = torch.eye(config.hidden_size, dtype=torch.float16)
            attention.qkv_proj.weight.copy_(torch.cat((identity, 0.5 * identity, 0.25 * identity)))
            attention.output_gate.weight.zero_()
            attention.out_proj.weight.copy_(identity)
            attention.norm.weight.fill_(1)

        attention.get_slope_rate = Mock(wraps=attention.get_slope_rate)
        attention.decay_factors = Mock(wraps=attention.decay_factors)

        hidden_states = torch.tensor([[[1.0, 0.5, -0.5, 2.0], [2.0, -1.0, 0.25, 0.75]]], dtype=torch.float16)
        attention_mask = torch.tensor([[1, 0]])
        cache = MiniMaxVL01TextCache()

        output, recurrent_state = attention(
            hidden_states,
            position_embeddings=None,
            attention_mask=attention_mask,
            past_key_values=cache,
        )
        changed_hidden_states = hidden_states.clone()
        changed_hidden_states[:, 1] += 3
        changed_cache = MiniMaxVL01TextCache()
        changed_output, changed_state = attention(
            changed_hidden_states,
            position_embeddings=None,
            attention_mask=attention_mask,
            past_key_values=changed_cache,
        )

        # A masked source cannot affect the unmasked output or the recurrent state used for decoding.
        torch.testing.assert_close(output[:, 0], changed_output[:, 0])
        torch.testing.assert_close(recurrent_state, changed_state)
        self.assertEqual(recurrent_state.dtype, torch.float32)
        self.assertEqual(cache.linear_cache[0].dtype, torch.float32)
        self.assertGreater(output[0, 1].abs().max().item(), 0)

        decode_hidden_states = torch.tensor([[[0.75, -0.25, 1.25, 0.5]]], dtype=torch.float16)
        output, recurrent_state = attention(
            decode_hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=cache,
        )
        changed_output, changed_state = attention(
            decode_hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=changed_cache,
        )
        torch.testing.assert_close(output, changed_output)
        torch.testing.assert_close(recurrent_state, changed_state)
        self.assertEqual(recurrent_state.dtype, torch.float32)
        self.assertEqual(cache.linear_cache[0].dtype, torch.float32)

        attention.get_slope_rate.assert_not_called()
        attention.decay_factors.assert_not_called()
        attention.half()
        attention.get_slope_rate.assert_called_once()
        attention.decay_factors.assert_called_once()
        attention(
            decode_hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=cache,
        )
        attention.get_slope_rate.assert_called_once()
        attention.decay_factors.assert_called_once()
        for name in ("slope_rate", "query_decay", "key_decay", "diagonal_decay"):
            self.assertEqual(getattr(attention, name).dtype, torch.float32)

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
        self.assertIsInstance(past_key_values, MiniMaxVL01TextCache)
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

    def _check_caches_are_equal(self, cache1: MiniMaxVL01TextCache, cache2: MiniMaxVL01TextCache):
        self.assertIsInstance(cache1, MiniMaxVL01TextCache)
        self.assertIsInstance(cache2, MiniMaxVL01TextCache)
        self.assertEqual(len(cache1), len(cache2))

        for layer_idx in range(len(cache1)):
            if layer_idx < len(cache1.layers) and cache1.layers[layer_idx].is_initialized:
                torch.testing.assert_close(cache1.layers[layer_idx].keys, cache2.layers[layer_idx].keys)
                torch.testing.assert_close(cache1.layers[layer_idx].values, cache2.layers[layer_idx].values)
            if layer_idx < len(cache1.linear_cache) and not isinstance(cache1.linear_cache[layer_idx], list):
                torch.testing.assert_close(cache1.linear_cache[layer_idx], cache2.linear_cache[layer_idx])

    @unittest.skip("Mixed full and Lightning attention uses the MiniMax-specific attention output contract.")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Lightning-attention outputs are recurrent states and are not part of the logits gradient path.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("Lightning attention carries its recurrent state across packed sequences, like MiniMax.")
    def attention_mask_padding_matches_padding_free_with_position_ids(
        self, attn_implementation: str, fa_kwargs: bool = False
    ):
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


@slow
@require_torch
class MiniMaxVL01IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    # TODO: replace with a tiny MiniMax-VL-01 checkpoint on the Hub, as MiniMax uses `hf-internal-testing/MiniMax-tiny`.
    @staticmethod
    def _dummy_model():
        config = MiniMaxVL01Config(
            text_config=MiniMaxVL01TextConfig(
                vocab_size=97,
                hidden_size=32,
                intermediate_size=16,
                num_hidden_layers=8,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                hidden_act="silu",
                max_position_embeddings=64,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                num_local_experts=4,
                num_experts_per_tok=2,
                block_size=4,
                layer_types=["linear_attention"] * 7 + ["full_attention"],
                rope_parameters={"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 0.5},
            ),
            vision_config=CLIPVisionConfig(
                hidden_size=16,
                intermediate_size=32,
                projection_dim=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_channels=3,
                image_size=8,
                patch_size=4,
                hidden_act="gelu",
            ),
            image_token_index=3,
            image_grid_pinpoints=[[8, 8]],
            pad_token_id=0,
        )
        torch.manual_seed(0)
        model = MiniMaxVL01ForConditionalGeneration(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            return MiniMaxVL01ForConditionalGeneration.from_pretrained(
                tmpdir, dtype=torch.float32, attn_implementation="eager"
            ).to(torch_device)

    @staticmethod
    def _image_inputs():
        pixel_values = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32, device=torch_device).reshape(4, 3, 8, 8)
        return {
            "input_ids": torch.tensor([[3] * 10 + [5, 6, 7], [3] * 10 + [8, 9, 10]], device=torch_device),
            "attention_mask": torch.ones(2, 13, dtype=torch.long, device=torch_device),
            "pixel_values": pixel_values / pixel_values.numel(),
            "image_sizes": torch.tensor([[8, 8], [8, 8]], device=torch_device),
        }

    def test_small_model_logits_batched(self):
        model = self._dummy_model()
        logits = model(**self._image_inputs(), logits_to_keep=3).logits
        expected_slice = torch.tensor(
            [
                [[-0.101953, 0.019430, 0.007517], [0.181610, 0.051723, -0.147496], [0.308670, -0.122409, -0.094961]],
                [[0.153198, 0.031881, 0.072750], [0.045287, -0.029863, -0.156537], [-0.031201, -0.085948, 0.008906]],
            ],
            device=torch_device,
        )
        torch.testing.assert_close(logits[:, -3:, -3:], expected_slice, atol=1e-4, rtol=1e-4)

    def test_small_model_generation(self):
        model = self._dummy_model()
        inputs = self._image_inputs()
        generated_ids = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        self.assertEqual(
            generated_ids[:, inputs["input_ids"].shape[1] :].tolist(),
            [[94, 93, 63, 89, 8], [23, 85, 93, 63, 38]],
        )


if __name__ == "__main__":
    unittest.main()
